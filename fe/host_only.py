import copy
import argparse
import time
import numpy as np
from io import StringIO
import itertools
import os
import sys
import pickle

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

from rdkit.Chem import rdMolAlign

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

def exp_grad_filter(all_du_dls_raw, T, schedule, true_dG):
    """
    Compute the derivative of exp weighted free energy with respect
    to the input work values
    """
    all_du_dls = []
    for du_dl in all_du_dls_raw:
        if du_dl is not None:
            all_du_dls.append(du_dl)
        else:
            all_du_dls.append(np.zeros_like(schedule))
    all_du_dls = jnp.array(all_du_dls)

    bkwd = all_du_dls[:, T:]
    bkwd_sched = schedule[T:]

    dG_bkwd = math_utils.trapz(bkwd, bkwd_sched) # integral from 0 to inf
    dG_bkwd = -dG_bkwd

    kT = 2.479
    dG_bkwd /= kT

    grads = exp_grad(dG_bkwd)

    return grads

def error_fn(all_du_dls_raw, T, schedule, true_dG):

    print("schedule", schedule)
    print("T", T)

    all_du_dls = []
    for conf_idx, du_dl in enumerate(all_du_dls_raw):
        if du_dl is not None:
            all_du_dls.append(du_dl)
            print("conf_idx", conf_idx, "dG", math_utils.trapz(du_dl[T:], schedule[T:]))
        else:
            print("conf_idx", conf_idx, "dG None")

    all_du_dls = jnp.array(all_du_dls)

    fwd = all_du_dls
    fwd_sched = schedule

    dG_fwd = math_utils.trapz(fwd, fwd_sched) # integral from inf to 0
    # this is in kJ/mol, inputs to BAR needs to be in 1/kT.
    kT = 2.479
    dG_fwd /= kT

    pred_dG = bar.EXP(dG_fwd)
    pred_dG *= kT

    print("dG_fwd", pred_dG, "true_dG", true_dG)
    return jnp.abs(pred_dG - true_dG)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Quick Test')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--precision', type=str, required=True)    
    parser.add_argument('--complex_pdb', type=str, required=True)
    parser.add_argument('--num_gpus', type=int, required=True)
    parser.add_argument('--num_conformers', type=int, required=True)
    parser.add_argument('--forcefield', type=str, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()

    assert os.path.isdir(args.out_dir)


    if args.precision == 'single':
        precision = np.float32
    elif args.precision == 'double':
        precision = np.float64
    else:
        raise Exception("precision must be either single or double")

    num_gpus = args.num_gpus
    all_du_dls = []

    cutoff = 100.0

    Ts = [
        5000, # minimization (with thermal noise lol)
        50000, # equilibration
    ]

    lambda_offset = Ts[0]+Ts[1]

    np.random.seed(args.seed)

    host_pdb_file = args.complex_pdb
    host_pdb = app.PDBFile(host_pdb_file)
    host_conf = []
    for x,y,z in host_pdb.positions:
        host_conf.append([to_md_units(x),to_md_units(y),to_md_units(z)])
    host_conf = np.array(host_conf)
    host_name = "complex"

    # set up the system
    amber_ff = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')
    host_system = amber_ff.createSystem(host_pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False)

    host_system = openmm_converter.deserialize_system(host_system, cutoff=cutoff)
    num_host_atoms = len(host_system.masses)

    temperature = 300
    dt = 1.5e-3
    friction = 91

    ca, cbs, ccs = langevin_coefficients(temperature, dt, friction, np.array(host_system.masses))

    cbs *= -1 # sign convention

    complete_lambda = np.concatenate([
        np.linspace(cutoff, cutoff, Ts[0]),
        np.linspace(cutoff, cutoff, Ts[1])
    ])

    complete_cas = np.concatenate([
        np.linspace(0.00, ca, Ts[0]),
        np.linspace(ca, ca, Ts[1])
    ])

    complete_dts = np.concatenate([
        np.linspace(1e-9, dt, Ts[0]),
        np.linspace(dt, dt, Ts[1])
    ])

    lambda_idxs = np.zeros(len(host_system.masses), dtype=np.int32)
    lambda_idxs[num_host_atoms:] = 1

    sim = simulation.Simulation(
        host_system,
        complete_dts,
        complete_cas,
        cbs,
        ccs,
        complete_lambda,
        lambda_idxs,
        precision,
        args.seed
    )

    initial_params = sim.system.params

    out_file = os.path.join(args.out_dir, "iconf_"+str(0)+".pdb")
    writer = PDBWriter(open(args.complex_pdb), out_file)
    x0 = host_conf
    v0 = np.zeros_like(x0)
    # writer can be None if we don't care about vis

    parent_conn, child_conn = Pipe()
    args = (x0, v0, 0 % num_gpus, writer, child_conn)
    
    p = Process(target=sim.run_forward_and_backward, args=args)
    p.start()
    parent_conn.recv()
    parent_conn.send(None)
    p.join()