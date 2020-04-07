import copy
import argparse
import time
import numpy as np
from io import StringIO
import itertools
import os
import sys

from timemachine.integrator import langevin_coefficients

from jax.config import config as jax_config
# this always needs to be set
jax_config.update("jax_enable_x64", True)

from scipy.stats import special_ortho_group
import jax
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

from simtk.openmm import app
from simtk.openmm.app import PDBFile

from timemachine.lib import custom_ops, ops
from fe.utils import to_md_units, write
from fe import math_utils

from multiprocessing import Process, Pipe
from matplotlib import pyplot as plt

from jax.experimental import optimizers

from fe import simulation
from fe import loss, bar
from fe.pdb_writer import PDBWriter

from ff import forcefield
from ff import system
from ff import openmm_converter

def com(conf):
    return np.sum(conf, axis=0)/conf.shape[0]

def recenter(conf, true_com, scale_factor=1):
    mol_com = np.sum(conf, axis=0)/conf.shape[0]
    centered = conf - mol_com  # centered to origin
    return true_com + centered/scale_factor 

from hilbertcurve.hilbertcurve import HilbertCurve

def hilbert_sort(conf):
    hc = HilbertCurve(16, 3)
    int_confs = (conf*1000).astype(np.int64)+10000
    dists = []
    for xyz in int_confs.tolist():
        dist = hc.distance_from_coordinates(xyz)
        dists.append(dist)
    perm = np.argsort(dists)
    return perm

def get_masses(m):
    masses = []
    for a in m.GetAtoms():
        masses.append(a.GetMass())
    return masses

import jax.numpy as jnp

exp_grad = jax.grad(bar.EXP)

def exp_grad_filter(all_du_dls_raw, schedule, true_dG):
    """
    Compute the derivative of exp weighted free energy with respect
    to the input work values
    """
 
    all_du_dls = []
    for conf_idx, du_dl in enumerate(all_du_dls_raw):
        if du_dl is not None:
            all_du_dls.append(du_dl)
            print("conf_idx", conf_idx, "dG", math_utils.trapz(du_dl, schedule))
        else:
            print("conf_idx", conf_idx, "dG None")

    all_du_dls = jnp.array(all_du_dls)
    work_insertion = math_utils.trapz(all_du_dls, schedule) # integral from 0 to inf

    kT = 2.479
    work_insertion /= kT

    grads = exp_grad(work_insertion)

    return grads

def error_fn(all_du_dls_raw, schedule, true_dG):

    all_du_dls = []
    for conf_idx, du_dl in enumerate(all_du_dls_raw):
        if du_dl is not None:
            all_du_dls.append(du_dl)
            print("conf_idx", conf_idx, "dG", math_utils.trapz(du_dl, schedule))
        else:
            print("conf_idx", conf_idx, "dG None")

    all_du_dls = jnp.array(all_du_dls)
    work_insertion = math_utils.trapz(all_du_dls, schedule) # integral from 0 to inf

    kT = 2.479
    work_insertion /= kT

    pred_dG = bar.EXP(work_insertion)
    pred_dG *= kT

    print("pred_dG", pred_dG, "true_dG", true_dG)
    return jnp.abs(pred_dG - true_dG)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Quick Test')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--precision', type=str, required=True)    
    parser.add_argument('--complex_pdb', type=str, required=True)
    parser.add_argument('--ligand_sdf', type=str, required=True)
    parser.add_argument('--num_gpus', type=int, required=True)
    parser.add_argument('--num_trials', type=int, required=True)
    parser.add_argument('--forcefield', type=str, required=True)
    args = parser.parse_args()

    assert os.path.isdir(args.out_dir)

    if args.precision == 'single':
        precision = np.float32
    elif args.precision == 'double':
        precision = np.float64
    else:
        raise Exception("precision must be either single or double")

    suppl = Chem.SDMolSupplier(args.ligand_sdf, removeHs=False)

    all_guest_mols = []
    for guest_mol in suppl:
        all_guest_mols.append(guest_mol)
    
    np.random.seed(123)

    perm = np.arange(len(all_guest_mols))
    np.random.shuffle(perm)

    print("======Picking Mol=======", perm[0])

    guest_mol = all_guest_mols[perm[0]]

    num_gpus = args.num_gpus
    all_du_dls = []



    lr = 5e-4
    # opt_init, opt_update, get_params = optimizers.adam(lr)
    opt_init, opt_update, get_params = optimizers.sgd(lr)

    host_pdb_file = args.complex_pdb
    host_pdb = app.PDBFile(host_pdb_file)
    host_conf = []
    for x,y,z in host_pdb.positions:
        host_conf.append([to_md_units(x),to_md_units(y),to_md_units(z)])
    host_conf = np.array(host_conf)
    host_name = "complex"

    conformer = guest_mol.GetConformer(0)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    guest_conf = guest_conf/10 # convert to md_units
    x0 = np.concatenate([host_conf, guest_conf]) # combined geometry
    # thermostat will bring velocities to the correct temperature
    v0 = np.zeros_like(x0)

    # set up the system
    amber_ff = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')
    host_system = amber_ff.createSystem(host_pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False)

    cutoff = 1.25

    host_system = openmm_converter.deserialize_system(host_system, cutoff=cutoff)
    num_host_atoms = len(host_system.masses)

    print("num_host_atoms", num_host_atoms)

    open_ff = forcefield.Forcefield(args.forcefield)
    nrg_fns = open_ff.parameterize(guest_mol, cutoff=cutoff)
    guest_masses = get_masses(guest_mol)
    guest_system = system.System(nrg_fns, open_ff.params, open_ff.param_groups, guest_masses)

    combined_system = host_system.merge(guest_system)

    temperature = 300
    dt = 1.5e-3
    friction = 91

    ca, cbs, ccs = langevin_coefficients(
        temperature,
        dt,
        friction,
        np.array(combined_system.masses)
    )

    cbs *= -1

    print("Integrator coefficients:")
    print("ca", ca)
    print("cbs", cbs)
    print("ccs", ccs)

    # insertion only
    Ts = [
        1000, # fast insertion
        5000, # slow insertion
    ]

    complete_T = np.sum(Ts)

    complete_lambda = np.concatenate([
        np.linspace(cutoff, 0.3, Ts[0]),
        np.linspace(0.3,   0.0,  Ts[1]),
    ])

    complete_cas = np.concatenate([
        np.linspace(0.00, ca, Ts[0]),
        np.linspace(ca, ca, Ts[1]),
    ])

    complete_dts = np.concatenate([
        np.linspace(dt, dt, Ts[0]),
        np.linspace(dt, dt, Ts[1]),
    ])

    lambda_idxs = np.zeros(len(combined_system.masses), dtype=np.int32)
    lambda_idxs[num_host_atoms:] = 1

    sim = simulation.Simulation(
        combined_system,
        complete_dts,
        complete_cas,
        cbs,
        ccs,
        complete_lambda,
        lambda_idxs,
        precision
    )

    initial_params = sim.system.params

    opt_state = opt_init(initial_params)

    num_epochs = 100

    for epoch in range(num_epochs):
        epoch_params = get_params(opt_state)

        # deepy and openff param at start
        epoch_ff_params = copy.deepcopy(open_ff)
        epoch_ff_params.params = epoch_params[len(host_system.params):]
        fname = "epoch_"+str(epoch)+"_params"
        fpath = os.path.join(args.out_dir, fname)
        epoch_ff_params.save(fpath)

        sim.system.params = np.asarray(epoch_params)

        all_args = []
        child_conns = []
        parent_conns = []
        processes = []

        np.random.seed(2020)
        for trial_idx in range(args.num_trials):

            if epoch % 10 == 0:
                combined_pdb = Chem.CombineMols(Chem.MolFromPDBFile(host_pdb_file, removeHs=False), guest_mol)
                combined_pdb_str = StringIO(Chem.MolToPDBBlock(combined_pdb))
                out_file = os.path.join(args.out_dir, "epoch_"+str(epoch)+"_insertion_"+str(trial_idx)+".pdb")
                writer = PDBWriter(combined_pdb_str, out_file)
            else:
                writer = None

            parent_conn, child_conn = Pipe()
            parent_conns.append(parent_conn)

            intg_seed = np.random.randint(np.iinfo(np.int32).max)

            all_args.append([x0, v0, intg_seed, writer, child_conn, trial_idx % args.num_gpus])

        for arg in all_args:
            p = Process(target=sim.run_forward_and_backward, args=arg)
            p.daemon = True
            processes.append(p)
            p.start()

        all_du_dls = []
        for pc in parent_conns:
            du_dls = pc.recv()
            all_du_dls.append(du_dls)

        loss_grad_fn = jax.grad(error_fn, argnums=(0,)) 
      
        for du_dls in all_du_dls:
            if du_dls is None:
                continue
            insertion = du_dls
            plt.plot(complete_lambda, insertion)

            plt.ylabel("du_dl")
            plt.xlabel("lambda")

        axes = plt.gca()
        axes.set_xlim([0, 2])
        plt.savefig(os.path.join(args.out_dir, "bkwd_epoch_"+str(epoch)+"_du_dls"))

        true_dG = -26.61024 # -6.36 * 4.184
        error = error_fn(all_du_dls, complete_lambda, true_dG)
        work_grads = exp_grad_filter(all_du_dls, complete_lambda, true_dG)

        print("---EPOCH", epoch, "---- LOSS", error)

        error_grad = loss_grad_fn(all_du_dls, complete_lambda, true_dG)
        all_du_dl_adjoints = error_grad[0]

        # this needs to be rescaled by number of conformers (eg. 100 conformers with 0.01 each will fail)
        work_grad_cutoff = 1e-2
        # send everything at once

        assert len(parent_conns) == len(all_du_dl_adjoints)
        assert len(all_du_dl_adjoints) == len(work_grads)
        assert len(work_grads) ==  len(all_du_dls)

        picked_conformers = []
        unstable_conformers = []

        print("work grads", work_grads)

        for conf_idx, (pc, du_dl_adjoints, wg, du_dls) in enumerate(zip(parent_conns, all_du_dl_adjoints, work_grads, all_du_dls)):
            if du_dls is None:
                unstable_conformers.append(conf_idx)
            if wg < work_grad_cutoff or du_dls is None:
                pc.send(None)
            else:
                picked_conformers.append(conf_idx)
                pc.send(du_dl_adjoints)

        print("picked conformers:", picked_conformers)
        print("unstable conformers:", unstable_conformers)

        # receive everything at once
        all_dl_dps = []
        for conf_idx, (pc, wg, du_dls) in enumerate(zip(parent_conns, work_grads, all_du_dls)):
            if wg > work_grad_cutoff and du_dls is not None:
                dl_dp = pc.recv()
                all_dl_dps.append(dl_dp)

        # terminate all the processes
        for p_idx, p in enumerate(processes):
            p.join()

        all_dl_dps = np.array(all_dl_dps)
        all_dl_dps = np.sum(all_dl_dps, axis=0)

        allowed_groups = {
            # 7: 0.5,
            14: 0.5, # small_molecule charge
            # 12: 1e-2, # GB atomic radii
            # 13: 1e-2 # GB scale factor
        }

        filtered_grad = []
        for g_idx, (g, gp) in enumerate(zip(all_dl_dps, sim.system.param_groups)):
            if gp in allowed_groups:
                pf = allowed_groups[gp]
                filtered_grad.append(g*pf)
                if g != 0:
                    print("derivs", g_idx, '\t group', gp, '\t', g, '\t adjusted to', g*pf, '\t old val', sim.system.params[g_idx])
            else:
                filtered_grad.append(0)

        filtered_grad = np.array(filtered_grad)
        opt_state = opt_update(epoch, filtered_grad, opt_state)