import matplotlib
matplotlib.use('Agg')
# import pickle
import copy
import argparse
import time
import datetime
import numpy as np
from io import StringIO
import itertools
import os
import sys

from ff.handlers.serialize import serialize_handlers
from ff.handlers.deserialize import deserialize

from multiprocessing import Process, Pipe

from jax.config import config as jax_config
# this always needs to be set
jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS

from fe import dataset

from fe import loss, bar
from fe.pdb_writer import PDBWriter

import configparser
import grpc

from training import trainer
from training import service_pb2_grpc

from ff.handlers import bonded, nonbonded, openmm_deserializer


from training import water_box

from timemachine.integrator import langevin_coefficients
from timemachine.lib import ops, custom_ops


def convert_uIC50_to_kJ_per_mole(amount_in_uM):
    return 0.593*np.log(amount_in_uM*1e-6)*4.18

def setup_system(
    ff_handlers,
    guest_mol,
    host_system,
    host_coords):

    host_fns, host_masses = openmm_deserializer.deserialize_system(host_system)

    guest_masses = np.array([a.GetMass() for a in guest_mol.GetAtoms()], dtype=np.float64)

    num_guest_atoms = len(guest_masses)
    num_host_atoms = len(host_masses)

    final_gradients = []

    for item in host_fns: 

        if item[0] == 'LennardJones':
            host_lj_params = item[1]
        elif item[0] == 'Charges':
            continue
            # host_charge_params = item[1]
        elif item[0] == 'Exclusions':
            host_exclusions = item[1]
        else:
            final_gradients.append((item[0], item[1]))


    guest_exclusion_idxs, guest_scales = nonbonded.generate_exclusion_idxs(
        guest_mol,
        scale12=1.0,
        scale13=1.0,
        scale14=0.5
    )

    guest_exclusion_idxs += num_host_atoms
    guest_lj_exclusion_scales = guest_scales
    # guest_charge_exclusion_scales = guest_scales

    host_exclusion_idxs = host_exclusions[0]
    host_lj_exclusion_scales = host_exclusions[1]
    # host_charge_exclusion_scales = host_exclusions[2]

    combined_exclusion_idxs = np.concatenate([host_exclusion_idxs, guest_exclusion_idxs])
    combined_lj_exclusion_scales = np.concatenate([host_lj_exclusion_scales, guest_lj_exclusion_scales])
    # combined_charge_exclusion_scales = np.concatenate([host_charge_exclusion_scales, guest_charge_exclusion_scales])

    for handle in ff_handlers:
        results = handle.parameterize(guest_mol)

        if isinstance(handle, bonded.HarmonicBondHandler):
            bond_idxs, (bond_params, _) = results
            bond_idxs += num_host_atoms
            # bind potentials
            final_gradients.append(("HarmonicBond", (bond_idxs, bond_params)))
        elif isinstance(handle, bonded.HarmonicAngleHandler):
            angle_idxs, (angle_params, _) = results
            angle_idxs += num_host_atoms
            final_gradients.append(("HarmonicAngle", (angle_idxs, angle_params)))
        elif isinstance(handle, bonded.ProperTorsionHandler):
            torsion_idxs, (torsion_params, _) = results
            torsion_idxs += num_host_atoms
            final_gradients.append(("PeriodicTorsion", (torsion_idxs, torsion_params)))
        elif isinstance(handle, bonded.ImproperTorsionHandler):
            torsion_idxs, (torsion_params, _) = results
            torsion_idxs += num_host_atoms
            final_gradients.append(("PeriodicTorsion", (torsion_idxs, torsion_params)))
        elif isinstance(handle, nonbonded.LennardJonesHandler):
            guest_lj_params, _ = results
            combined_lj_params = np.concatenate([host_lj_params, guest_lj_params])
        else:
            print("skipping", handle)
            pass

    host_conf = np.array(host_coords)

    conformer = guest_mol.GetConformer(0)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    guest_conf = guest_conf/10 # convert to md_units

    x0 = np.concatenate([host_conf, guest_conf]) # combined geometry

    # print("x0", x0)
    # assert 0

    # v0 = np.zeros_like(x0)

    N_C = num_host_atoms + num_guest_atoms
    N_A = num_host_atoms

    cutoff = 100000.0

    combined_lambda_plane_idxs = np.zeros(N_C, dtype=np.int32)
    combined_lambda_offset_idxs = np.zeros(N_C, dtype=np.int32)
    combined_lambda_offset_idxs[num_host_atoms:] = 1

    # print(combined_lambda_plane_idxs)
    # print(combined_lambda_offset_idxs)

    # assert 0

    final_gradients.append((
        'LennardJones', (
        # np.asarray(combined_charge_params),
        combined_exclusion_idxs,
        # combined_charge_exclusion_scales,
        combined_lj_exclusion_scales,
        combined_lambda_plane_idxs,
        combined_lambda_offset_idxs,
        cutoff,
        np.asarray(combined_lj_params),
        )
    ))

    combined_masses = np.concatenate([host_masses, guest_masses])


    return x0, combined_masses, final_gradients

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Absolute Hydration Free Energy Script')

    ligand_sdf = "/home/yutong/Downloads/ligands_40.sdf"

    suppl = Chem.SDMolSupplier(ligand_sdf, removeHs=False)

    all_guest_mols = []

    data = []

    for guest_mol in suppl:
        break

    forcefield = "ff/params/smirnoff_1_1_0_ccc.py"

    ff_raw = open(forcefield, "r").read()

    ff_handlers = deserialize(ff_raw)

    host_system, host_coords, box = water_box.get_water_box(2.0)

    # print(box)
    # print(host_coords)
    # assert 0

    x0, combined_masses, final_gradients = setup_system(
        ff_handlers,
        guest_mol,
        host_system,
        host_coords
    )

    v0 = np.zeros_like(x0)

    # bind final_gradients
    bps = []
    pots = []

    lamb = 0.5
    for name, args in final_gradients:
        print("---potential---", name)
        params = args[-1]
        op_fn = getattr(ops, name)
        potential = op_fn(*args[:-1], precision=np.float32)
        pots.append(potential) # (ytz) needed for binding, else python decides to GC this
        du_dx, du_dp, du_dl, u = potential.execute(x0, params, box, lamb)
        # print(du_dx)
        # print(du_dp, du_dl, u)
        bp = custom_ops.BoundPotential(potential, params)
        bps.append(bp)

    print("bps", bps)
    # for bp in bps:
        # bp.execute_host()

    dt = 1.5e-3

    ca, cbs, ccs = langevin_coefficients(
        temperature=300.0,
        dt=dt,
        friction=1.0,
        masses=combined_masses
    )

    cbs *= -1

    seed = 2020

    intg = custom_ops.LangevinIntegrator(
        dt,
        ca,
        cbs,
        ccs,
        seed
    )

    obs = []

    ctxt = custom_ops.Context(
        x0,
        v0,
        box,
        lamb,
        intg,
        bps,
        obs
    )

    for step in range(10000):
        # print(step)
        ctxt.step()

    print("final_coords", ctxt.get_x_t())

    # integrator = system.Integrator(
    #     steps=n_steps,
    #     dt=1.5e-3,
    #     temperature=300.0,
    #     friction=50,
    #     masses=combined_masses,
    #     # lamb=np.zeros(n_steps),
    #     seed=42
    # )



    # print(combined_masses)

    assert 0

    # return x0, combined_masses, final_gradients

    assert 0
        # elif isinstance(handle, nonbonded.SimpleChargeHandler):
        #     guest_charge_params, _ = results
        #     combined_charge_params = np.concatenate([host_charge_params, guest_charge_params])
        # elif isinstance(handle, nonbonded.GBSAHandler):
        #     guest_gb_params, _ = results
        #     combined_gb_params = np.concatenate([host_gb_params, guest_gb_params])
        # elif isinstance(handle, nonbonded.AM1BCCHandler):
        #     guest_charge_params, _ = results
        #     combined_charge_params = np.concatenate([host_charge_params, guest_charge_params])
        # elif isinstance(handle, nonbonded.AM1CCCHandler):
        #     guest_charge_params, _ = results
        #     combined_charge_params = np.concatenate([host_charge_params, guest_charge_params])
        # else:
        #     raise Exception("Unknown Handler", handle)


    print(ff_handlers)


        # mol_dG = -1*convert_uIC50_to_kJ_per_mole(float(mol.GetProp(general_cfg['bind_prop'])))
        # data.append((mol, mol_dG))

    # parser.add_argument('--config_file', type=str, required=True, help='Location of config file.')

    # args = parser.parse_args()

    # print("Launch Time:", datetime.datetime.now())

    # config = configparser.ConfigParser()
    # config.read(args.config_file)
    # print("Config Settings:")
    # config.write(sys.stdout)

    # general_cfg = config['general']

    # if not os.path.exists(general_cfg['out_dir']):
    #     os.makedirs(general_cfg['out_dir'])

    # suppl = Chem.SDMolSupplier(general_cfg['ligand_sdf'], removeHs=False)

    # all_guest_mols = []

    # data = []

    # for guest_idx, mol in enumerate(suppl):
    #     mol_dG = -1*convert_uIC50_to_kJ_per_mole(float(mol.GetProp(general_cfg['bind_prop'])))
    #     data.append((mol, mol_dG))

    # full_dataset = dataset.Dataset(data)
    # train_frac = float(general_cfg['train_frac'])
    # train_dataset, test_dataset = full_dataset.split(train_frac)

    # # process the host first
    # host_pdbfile = general_cfg['protein_pdb']

    # stage_dGs = []

    # ff_raw = open(general_cfg['forcefield'], "r").read()
    # ff_handlers = deserialize(ff_raw)

    # worker_address_list = []
    # for address in config['workers']['hosts'].split(','):
    #     worker_address_list.append(address)

    # stubs = []

    # for address in worker_address_list:
    #     print("connecting to", address)
    #     channel = grpc.insecure_channel(address,
    #         options = [
    #             ('grpc.max_send_message_length', 500 * 1024 * 1024),
    #             ('grpc.max_receive_message_length', 500 * 1024 * 1024)
    #         ]
    #     )

    #     stub = service_pb2_grpc.WorkerStub(channel)
    #     stubs.append(stub)

    # intg_cfg = config['integrator']
    # lr_config = config['learning_rates']
    # restr_config = config['restraints']

    # lambda_schedule = {}

    # for stage_str, v in config['lambda_schedule'].items():

    #     stage = int(stage_str)
    #     stage_schedule = np.array([float(x) for x in v.split(',')])

    #     assert stage not in lambda_schedule

    #     if stage == 0 or stage == 1:
    #         # stage 0 must be monotonically decreasing
    #         assert np.all(np.diff(stage_schedule) > 0)
    #     else:
    #         raise Exception("unknown stage")
    #         # stage 1 and 2 must be monotonically increasing
    #         # assert np.all(np.diff(stage_schedule) > 0)
    #     lambda_schedule[stage] = stage_schedule

    # learning_rates = {}
    # for k, v in config['learning_rates'].items():
    #     learning_rates[k] = np.array([float(x) for x in v.split(',')])

    # engine = trainer.Trainer(
    #     host_pdbfile, 
    #     stubs,
    #     worker_address_list,
    #     ff_handlers,
    #     lambda_schedule,
    #     int(general_cfg['du_dl_cutoff']),
    #     float(restr_config['search_radius']),
    #     float(restr_config['force_constant']),
    #     int(general_cfg['n_frames']),
    #     int(intg_cfg['steps']),
    #     float(intg_cfg['dt']),
    #     float(intg_cfg['temperature']),
    #     float(intg_cfg['friction']),
    #     learning_rates,
    #     general_cfg['precision']
    # )

    # for epoch in range(100):

    #     print("Starting Epoch", epoch, datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    #     epoch_dir = os.path.join(general_cfg['out_dir'], "epoch_"+str(epoch))

    #     if not os.path.exists(epoch_dir):
    #         os.makedirs(epoch_dir)

    #     epoch_params = serialize_handlers(ff_handlers)
    #     with open(os.path.join(epoch_dir, "start_epoch_params.py"), 'w') as fh:
    #         fh.write(epoch_params)

    #     for mol, experiment_dG in test_dataset.data:
    #         print("test mol", mol.GetProp("_Name"), "Smiles:", Chem.MolToSmiles(mol))
    #         mol_dir = os.path.join(epoch_dir, "test_mol_"+mol.GetProp("_Name"))
    #         start_time = time.time()
    #         dG, ci, loss = engine.run_mol(mol, inference=True, run_dir=mol_dir, experiment_dG=experiment_dG)
    #         print(mol.GetProp("_Name"), "test loss", loss, "pred_dG", dG, "exp_dG", experiment_dG, "time", time.time() - start_time, "ci 95% (mean, lower, upper)", ci.value, ci.lower_bound, ci.upper_bound)

    #     train_dataset.shuffle()

    #     for mol, experiment_dG in train_dataset.data:
    #         print("train mol", mol.GetProp("_Name"), "Smiles:", Chem.MolToSmiles(mol))
    #         mol_dir = os.path.join(epoch_dir, "train_mol_"+mol.GetProp("_Name"))
    #         start_time = time.time()
    #         dG, ci, loss = engine.run_mol(mol, inference=False, run_dir=mol_dir, experiment_dG=experiment_dG)
    #         print(mol.GetProp("_Name"), "train loss", loss, "pred_dG", dG, "exp_dG", experiment_dG, "time", time.time() - start_time, "ci 95% (mean, lower, upper)", ci.value, ci.lower_bound, ci.upper_bound)

    #     epoch_params = serialize_handlers(ff_handlers)
    #     with open(os.path.join(epoch_dir, "end_epoch_params.py"), 'w') as fh:
    #         fh.write(epoch_params)
