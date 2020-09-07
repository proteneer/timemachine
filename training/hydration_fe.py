# import matplotlib
# matplotlib.use('Agg')
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
import jax
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
from training import service_pb2
from training import service_pb2_grpc

from ff.handlers import bonded, nonbonded, openmm_deserializer


from training import water_box

from timemachine.integrator import langevin_coefficients
from timemachine.lib import ops, custom_ops

from matplotlib import pyplot as plt

import pickle

def convert_uIC50_to_kJ_per_mole(amount_in_uM):
    return 0.593*np.log(amount_in_uM*1e-6)*4.18


def concat_with_vjps(p_a, p_b, vjp_a, vjp_b):
    """
    Returns the combined parameters p_c, and a vjp_fn that can take in adjoint with shape
    of p_c and returns adjoints of primitives of p_a and p_b.

    i.e. 
       vjp_a            
    A' -----> A 
                \ vjp_c
                 +-----> C
       vjp_b    /
    B' -----> B

    """
    p_c, vjp_c = jax.vjp(jnp.concatenate, [p_a, p_b])
    adjoints = np.random.randn(*p_c.shape)

    def adjoint_fn(p_c):
        ad_a, ad_b = vjp_c(p_c)[0]
        if vjp_a is not None:
            ad_a = vjp_a(ad_a)
        else:
            ad_a = None

        if vjp_b is not None:
            ad_b = vjp_b(ad_b)
        else:
            ad_b = None

        return ad_b[0]

    return p_c, adjoint_fn

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
    
    handler_vjp_fns = {}

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
            guest_lj_params, guest_lj_vjp_fn = results
            combined_lj_params, handler_vjp_fn = concat_with_vjps(
                host_lj_params,
                guest_lj_params,
                None,
                guest_lj_vjp_fn
            )

            # move to outside of if later
            handler_vjp_fns[handle] = handler_vjp_fn
            # combined_lj_params = np.concatenate([host_lj_params, guest_lj_params])
        else:
            print("skipping", handle)
            pass



    host_conf = np.array(host_coords)

    conformer = guest_mol.GetConformer(0)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    guest_conf = guest_conf/10 # convert to md_units
    guest_conf -= np.array([[1.0, 1.0, 1.0]]) # displace to recenter

    x0 = np.concatenate([host_conf, guest_conf]) # combined geometry

    N_C = num_host_atoms + num_guest_atoms
    N_A = num_host_atoms

    cutoff = 100000.0

    combined_lambda_plane_idxs = np.zeros(N_C, dtype=np.int32)
    combined_lambda_offset_idxs = np.zeros(N_C, dtype=np.int32)
    combined_lambda_offset_idxs[num_host_atoms:] = 1

    final_gradients.append((
        'LennardJones', (
        combined_exclusion_idxs,
        combined_lj_exclusion_scales,
        combined_lambda_plane_idxs,
        combined_lambda_offset_idxs,
        cutoff,
        np.asarray(combined_lj_params),
        )
    ))

    combined_masses = np.concatenate([host_masses, guest_masses])

    return x0, combined_masses, final_gradients, handler_vjp_fns

def simulate(
    guest_mol,
    ff_handlers,
    stubs,
    n_frames):

    box_width = 3.0
    host_system, host_coords, box, host_pdbfile = water_box.get_water_box(box_width)

    # print(box)
    # print(host_coords)
    # assert 0

    # janky
    combined_pdb = Chem.CombineMols(Chem.MolFromPDBFile(host_pdbfile, removeHs=False), guest_mol)

    x0, combined_masses, final_gradients, handler_vjp_fns = setup_system(
        ff_handlers,
        guest_mol,
        host_system,
        host_coords
    )

    v0 = np.zeros_like(x0)

    simulate_futures = []
    lambda_schedule = np.concatenate([
        np.linspace(0.0, 0.6, 40, endpoint=False),
        np.linspace(0.6, 1.5, 20, endpoint=False),
        np.linspace(1.5, 5.5, 20, endpoint=True)
    ])

    # lambda_schedule = np.array([0.0, 0.5, 15.0])

    for lamb_idx, lamb in enumerate(lambda_schedule):

        bps = []
        pots = []

        for name, args in final_gradients:
            params = args[-1]
            op_fn = getattr(ops, name)
            potential = op_fn(*args[:-1], precision=np.float32)
            pots.append(potential) # (ytz) needed for binding, else python decides to GC this
            du_dx, du_dp, du_dl, u = potential.execute(x0, params, box, lamb)
            bp = custom_ops.BoundPotential(potential, params)
            bps.append(bp)

        dt = 1.5e-3

        ca, cbs, ccs = langevin_coefficients(
            temperature=300.0,
            dt=dt,
            friction=1.0,
            masses=combined_masses
        )
        cbs *= -1

        seed = np.random.randint(150000)

        intg_args = (dt, ca, cbs, ccs, seed)

        complex_system = system.System(
            x0,
            v0,
            box,
            final_gradients,
            intg_args
        )

        # endpoint lambda
        if lamb_idx == 0 or lamb_idx == len(lambda_schedule) - 1:
            observe_du_dl_freq = 5000 # this is analytically zero.
            observe_du_dp_freq = 25
        else:
            observe_du_dl_freq = 25 # this is analytically zero.
            observe_du_dp_freq = 0

        request = service_pb2.SimulateRequest(
            system=pickle.dumps(complex_system),
            lamb=lamb,
            prep_steps=5000,
            # prod_steps=100000,
            prod_steps=100000,
            observe_du_dl_freq=observe_du_dl_freq,
            observe_du_dp_freq=observe_du_dp_freq,
            precision="single",
            n_frames=n_frames,
        )

        stub = stubs[lamb_idx % len(stubs)]

        # launch asynchronously
        response_future = stub.Simulate.future(request)
        simulate_futures.append(response_future)

    lj_du_dps = []

    du_dls = []

    for lamb_idx, (lamb, future) in enumerate(zip(lambda_schedule, simulate_futures)):
        response = future.result()
        energies = pickle.loads(response.energies)

        if n_frames > 0:
            frames = pickle.loads(response.frames)
            combined_pdb_str = StringIO(Chem.MolToPDBBlock(combined_pdb))

            out_file = "simulation_"+str(lamb_idx)+".pdb"
            pdb_writer = PDBWriter(combined_pdb_str, out_file)
            pdb_writer.write_header(box)
            for x in frames:
                x = recenter(x, box)
                pdb_writer.write(x*10)
            pdb_writer.close()

        du_dl = pickle.loads(response.avg_du_dls)
        du_dls.append(du_dl)

        if lamb_idx == 0 or lamb_idx == len(lambda_schedule) - 1:
            lj_du_dps.append(pickle.loads(response.avg_du_dps)[0])

        print("lamb", lamb, "avg_du_dl", du_dl)


    # expected = -150

    predicted = np.trapz(du_dls, lambda_schedule)

    print("dG", predicted)
    # loss = np.abs(predicted - expected)
    # loss_grad = np.sign(predicted - expected)

    # lj_du_dp = loss_grad*(lj_du_dps[0] - lj_du_dps[1])
    lj_du_dp = (lj_du_dps[0] - lj_du_dps[1])
    print("lj_du_dp", lj_du_dp)

    for h, vjp_fn in handler_vjp_fns.items():

        # if isinstance(h, nonbonded.SimpleChargeHandler):
        #     # disable training to SimpleCharges
        #     assert 0
        #     h.params -= charge_gradients*self.learning_rates['charge']
        # elif isinstance(h, nonbonded.AM1CCCHandler):
        #     charge_gradients = vjp_fn(sum_charge_derivs)
        #     if np.any(np.isnan(charge_gradients)) or np.any(np.isinf(charge_gradients)) or np.any(np.amax(np.abs(charge_gradients)) > 10000.0):
        #         print("Skipping Fatal Charge Derivatives:", charge_gradients)
        #     else:
        #         charge_scale_factor = np.amax(np.abs(charge_gradients))/self.learning_rates['charge']
        #         h.params -= charge_gradients/charge_scale_factor
        if isinstance(h, nonbonded.LennardJonesHandler):
            lj_grads = np.asarray(vjp_fn(lj_du_dp)).copy()
            print("before", lj_grads)
            lj_grads[np.isnan(lj_grads)] = 0.0
            clip = 0.003
            # clipped grads
            lj_grads = np.clip(lj_grads, -clip, clip)
            print("after", lj_grads)

            h.params -= lj_grads
            # assert 0

            # if np.any(np.isnan(lj_gradients)) or np.any(np.isinf(lj_gradients)) or np.any(np.amax(np.abs(lj_gradients)) > 10000.0):
            #     print("Skipping Fatal LJ Derivatives:", lj_gradients)
            # else:
            #     lj_sig_scale = np.amax(np.abs(lj_gradients[:, 0]))/self.learning_rates['lj'][0]
            #     lj_eps_scale = np.amax(np.abs(lj_gradients[:, 1]))/self.learning_rates['lj'][1]
            #     lj_scale_factor = np.array([lj_sig_scale, lj_eps_scale])
            #     h.params -= lj_gradients/lj_scale_factor


def recenter(conf, box):

    new_coords = []

    periodicBoxSize = box

    for atom in conf:
        diff = np.array([0., 0., 0.])
        diff += periodicBoxSize[2]*np.floor(atom[2]/periodicBoxSize[2][2]);
        diff += periodicBoxSize[1]*np.floor((atom[1]-diff[1])/periodicBoxSize[1][1]);
        diff += periodicBoxSize[0]*np.floor((atom[0]-diff[0])/periodicBoxSize[0][0]);
        new_coords.append(atom - diff)

    return np.array(new_coords)


from fe import math_utils, system


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Absolute Hydration Free Energy Script')

    parser.add_argument('--n_workers', type=int, help='number of workers')
    parser.add_argument('--n_frames', type=int, help='number of frames')
    parser.add_argument('--ligand_sdf', type=str, help='ligand sdf file path')

    args = parser.parse_args()

    suppl = Chem.SDMolSupplier(args.ligand_sdf, removeHs=False)

    all_guest_mols = []

    data = []

    for guest_mol in suppl:
        break

    forcefield = "ff/params/smirnoff_1_1_0_ccc.py"


    stubs = []

    address_list = []
    for idx in range(args.n_workers):
        address_list.append("0.0.0.0:"+str(5000+idx))

    for address in address_list:
        print("connecting to", address)
        channel = grpc.insecure_channel(address,
            options = [
                ('grpc.max_send_message_length', 500 * 1024 * 1024),
                ('grpc.max_receive_message_length', 500 * 1024 * 1024)
            ]
        )

        stub = service_pb2_grpc.WorkerStub(channel)
        stubs.append(stub)


    ff_raw = open(forcefield, "r").read()

    ff_handlers = deserialize(ff_raw)

    for epoch in range(100):
        print("=====epoch====", epoch)
        simulate(
            guest_mol,
            ff_handlers,
            stubs,
            args.n_frames
        )
