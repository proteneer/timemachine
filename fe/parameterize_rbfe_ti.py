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
from fe import linear_mixer
from fe import atom_mapping
from fe.pdb_writer import PDBWriter

from ff import forcefield
from ff import system
from ff import openmm_converter
import jax.numpy as jnp


from rdkit.Chem import rdFMCS

def convert_uIC50_to_kJ_per_mole(amount_in_uM):
    return 0.593*np.log(amount_in_uM*1e-6)*4.18

def get_masses(m):
    masses = []
    for a in m.GetAtoms():
        masses.append(a.GetMass())
    return masses

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Quick Test')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--precision', type=str, required=True)    
    parser.add_argument('--complex_pdb', type=str, required=True)
    parser.add_argument('--ligand_sdf', type=str, required=True)
    parser.add_argument('--num_gpus', type=int, required=True)
    parser.add_argument('--forcefield', type=str, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--cutoff', type=float, required=True)
    parser.add_argument('--num_windows', type=int, required=True)
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
    for guest_idx, guest_mol in enumerate(suppl):
        all_guest_mols.append(guest_mol)

    all_guest_mols = all_guest_mols[:2]

    a_to_b_map = atom_mapping.mcs_map(*all_guest_mols)

    c = atom_mapping.mcs_map(*all_guest_mols)
    d = atom_mapping.mcs_map(*all_guest_mols)
    e = atom_mapping.mcs_map(*all_guest_mols)

    assert c == d

    assert d == e
    open_ff = forcefield.Forcefield(args.forcefield)
    
    all_nrg_fns = []
    mol_a = all_guest_mols[0]
    mol_b = all_guest_mols[1]

    lhs_nrg_fns = {}
    rhs_nrg_fns = {}

    a_masses = get_masses(mol_a)
    b_masses = get_masses(mol_b)

    combined_masses = np.concatenate([a_masses, b_masses])

    a_nrg_fns = open_ff.parameterize(mol_a, cutoff=args.cutoff, am1=False)
    b_nrg_fns = open_ff.parameterize(mol_b, cutoff=args.cutoff, am1=False)

    a_bond_idxs, a_bond_param_idxs = a_nrg_fns['HarmonicBond']
    b_bond_idxs, b_bond_param_idxs = b_nrg_fns['HarmonicBond']

    lm = linear_mixer.LinearMixer(mol_a.GetNumAtoms(), a_to_b_map)

    lhs_bond_idxs, lhs_bond_param_idxs, rhs_bond_idxs, rhs_bond_param_idxs = lm.mix_arbitrary_bonds(
        a_bond_idxs, a_bond_param_idxs,
        b_bond_idxs, b_bond_param_idxs
    )

    lhs_nrg_fns['HarmonicBond'] = (lhs_bond_idxs, lhs_bond_param_idxs)
    rhs_nrg_fns['HarmonicBond'] = (rhs_bond_idxs, rhs_bond_param_idxs)

    a_angle_idxs, a_angle_param_idxs = a_nrg_fns['HarmonicAngle']
    b_angle_idxs, b_angle_param_idxs = b_nrg_fns['HarmonicAngle']

    lhs_angle_idxs, lhs_angle_param_idxs, rhs_angle_idxs, rhs_angle_param_idxs = lm.mix_arbitrary_bonds(
        a_angle_idxs, a_angle_param_idxs,
        b_angle_idxs, b_angle_param_idxs
    )

    lhs_nrg_fns['HarmonicAngle'] = (lhs_angle_idxs, lhs_angle_param_idxs)
    rhs_nrg_fns['HarmonicAngle'] = (rhs_angle_idxs, rhs_angle_param_idxs)

    a_torsion_idxs, a_torsion_param_idxs = a_nrg_fns['PeriodicTorsion']
    b_torsion_idxs, b_torsion_param_idxs = b_nrg_fns['PeriodicTorsion']

    lhs_torsion_idxs, lhs_torsion_param_idxs, rhs_torsion_idxs, rhs_torsion_param_idxs = lm.mix_arbitrary_bonds(
        a_torsion_idxs, a_torsion_param_idxs,
        b_torsion_idxs, b_torsion_param_idxs
    )

    lhs_nrg_fns['PeriodicTorsion'] = (lhs_torsion_idxs, lhs_torsion_param_idxs)
    rhs_nrg_fns['PeriodicTorsion'] = (rhs_torsion_idxs, rhs_torsion_param_idxs)

    lambda_plane_idxs, lambda_offset_idxs = lm.mix_lambda_planes(mol_a.GetNumAtoms(), mol_b.GetNumAtoms())

    print(lambda_plane_idxs)
    print(lambda_offset_idxs)

    assert lambda_offset_idxs[26] == 1
    assert lambda_offset_idxs[35] == 1
    assert lambda_offset_idxs[36] == 1

    assert lambda_offset_idxs[18+mol_a.GetNumAtoms()] == 1
    assert lambda_offset_idxs[19+mol_a.GetNumAtoms()] == 1
    assert lambda_offset_idxs[20+mol_a.GetNumAtoms()] == 1
    assert lambda_offset_idxs[32+mol_a.GetNumAtoms()] == 1

    a_es_param_idxs, a_lj_param_idxs, a_exc_idxs, a_es_exc_param_idxs, a_lj_exc_param_idxs, a_cutoff = a_nrg_fns['Nonbonded']
    b_es_param_idxs, b_lj_param_idxs, b_exc_idxs, b_es_exc_param_idxs, b_lj_exc_param_idxs, b_cutoff = b_nrg_fns['Nonbonded']

    assert a_cutoff == args.cutoff
    assert a_cutoff == b_cutoff

    lhs_es_param_idxs, rhs_es_param_idxs = lm.mix_nonbonded_parameters(a_es_param_idxs, b_es_param_idxs)
    lhs_lj_param_idxs, rhs_lj_param_idxs = lm.mix_nonbonded_parameters(a_lj_param_idxs, b_lj_param_idxs)

    (_,            lhs_lj_exc_param_idxs), (           _, rhs_lj_exc_param_idxs) = lm.mix_exclusions(a_exc_idxs, a_lj_exc_param_idxs, b_exc_idxs, b_lj_exc_param_idxs)
    (lhs_exc_idxs, lhs_es_exc_param_idxs), (rhs_exc_idxs, rhs_es_exc_param_idxs) = lm.mix_exclusions(a_exc_idxs, a_es_exc_param_idxs, b_exc_idxs, b_es_exc_param_idxs)

    for exc, param in zip(rhs_exc_idxs, rhs_lj_exc_param_idxs):
        src, dst = exc
        if src == 2 or dst == 2:
            print("!!!", src, dst, param)

    assert (26, 15 + mol_a.GetNumAtoms()) in lhs_exc_idxs
    assert (26, 16 + mol_a.GetNumAtoms()) in lhs_exc_idxs
    assert (26, 17 + mol_a.GetNumAtoms()) in lhs_exc_idxs
    assert (26, 3 + mol_a.GetNumAtoms()) in lhs_exc_idxs
    assert (19, 26) in lhs_exc_idxs
    assert (20, 26) in lhs_exc_idxs
    assert (21, 26) in lhs_exc_idxs
    # assert (19, 26) in lhs_exc_idxs

    # print("OKAY")
    # assert 0

    lhs_exc_idxs = np.array(lhs_exc_idxs, dtype=np.int32)
    rhs_exc_idxs = np.array(rhs_exc_idxs, dtype=np.int32)

    lhs_es_exc_param_idxs = np.array(lhs_es_exc_param_idxs, dtype=np.int32) 
    rhs_es_exc_param_idxs = np.array(rhs_es_exc_param_idxs, dtype=np.int32) 
    lhs_lj_exc_param_idxs = np.array(lhs_lj_exc_param_idxs, dtype=np.int32) 
    rhs_lj_exc_param_idxs = np.array(rhs_lj_exc_param_idxs, dtype=np.int32)

    lhs_nrg_fns['Nonbonded'] = (lhs_es_param_idxs, lhs_lj_param_idxs, lhs_exc_idxs, lhs_es_exc_param_idxs, lhs_lj_exc_param_idxs, lambda_plane_idxs, lambda_offset_idxs, a_cutoff)
    rhs_nrg_fns['Nonbonded'] = (rhs_es_param_idxs, rhs_lj_param_idxs, rhs_exc_idxs, rhs_es_exc_param_idxs, rhs_lj_exc_param_idxs, lambda_plane_idxs, lambda_offset_idxs, b_cutoff)

    a_gb_args = a_nrg_fns['GBSA']
    b_gb_args = b_nrg_fns['GBSA']

    a_gb_charges, a_gb_radii, a_gb_scales = a_gb_args[:3]
    b_gb_charges, b_gb_radii, b_gb_scales = b_gb_args[:3]

    assert a_gb_args[3:] == b_gb_args[3:]

    lhs_gb_charges, rhs_gb_charges = lm.mix_nonbonded_parameters(a_gb_charges, b_gb_charges)
    lhs_gb_radii, rhs_gb_radii = lm.mix_nonbonded_parameters(a_gb_radii, b_gb_radii)
    lhs_gb_scales, rhs_gb_scales = lm.mix_nonbonded_parameters(a_gb_scales, b_gb_scales)

    lhs_nrg_fns['GBSA'] = (lhs_gb_charges, lhs_gb_radii, lhs_gb_scales, lambda_plane_idxs, lambda_offset_idxs, *a_gb_args[3:])
    rhs_nrg_fns['GBSA'] = (rhs_gb_charges, rhs_gb_radii, rhs_gb_scales, lambda_plane_idxs, lambda_offset_idxs, *a_gb_args[3:])

    lhs_dual_system = system.System(lhs_nrg_fns, open_ff.params, open_ff.param_groups, combined_masses)
    rhs_dual_system = system.System(rhs_nrg_fns, open_ff.params, open_ff.param_groups, combined_masses)

    host_pdb_file = args.complex_pdb
    host_pdb = app.PDBFile(host_pdb_file)
    amber_ff = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')
    host_system = amber_ff.createSystem(host_pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False)

    host_system = openmm_converter.deserialize_system(host_system, cutoff=args.cutoff)
    lhs_combined_system = host_system.merge(lhs_dual_system)
    rhs_combined_system = host_system.merge(rhs_dual_system)

    for _ in range(100):

        temperature = 300
        # dt = 1.5e-3
        dt = 1.5e-3
        friction = 40

        np.testing.assert_array_equal(lhs_combined_system.masses, rhs_combined_system.masses)

        masses = np.array(lhs_combined_system.masses)
        ca, cbs, ccs = langevin_coefficients(
            temperature,
            dt,
            friction,
            masses
        )

        cbs *= -1

        print("Integrator coefficients:")
        print("ca", ca)
        print("cbs", cbs)
        print("ccs", ccs)
     
        complete_T = 12000
        equil_T = 2000

        print("CUTOFF", args.cutoff)

        ti_lambdas = np.linspace(0, 1, args.num_windows)
        # ti_lambdas = np.ones(args.num_windows)*0.2

        # ti_lambdas = np.array([0.001, 0.01, 0.1])

        all_du_dls = []
        # all_args = []

        all_processes = []
        all_pcs = []


        for lambda_idx, lamb in enumerate(ti_lambdas):

            complete_lambda = np.zeros(complete_T) + lamb
            complete_cas = np.ones(complete_T)*ca
            complete_dts = np.concatenate([
                np.linspace(0, dt, equil_T),
                np.ones(complete_T-equil_T)*dt
            ])

            sim = simulation.Simulation(
                lhs_combined_system,
                rhs_combined_system,
                complete_dts,
                complete_cas,
                cbs,
                ccs,
                complete_lambda,
                precision
            )

            intg_seed = np.random.randint(np.iinfo(np.int32).max)
            # intg_seed = args.seed

            combined_ligand = Chem.CombineMols(mol_a, mol_b)
            combined_pdb = Chem.CombineMols(Chem.MolFromPDBFile(host_pdb_file, removeHs=False), combined_ligand)
            combined_pdb_str = StringIO(Chem.MolToPDBBlock(combined_pdb))
            out_file = os.path.join(args.out_dir, "rbfe_"+str(lamb)+".pdb")
            writer = PDBWriter(combined_pdb_str, out_file)
            # writer = None

            host_conf = []
            for x,y,z in host_pdb.positions:
                host_conf.append([to_md_units(x),to_md_units(y),to_md_units(z)])
            host_conf = np.array(host_conf)

            print("num host atoms", host_conf.shape[0])

            conformer = mol_a.GetConformer(0)
            mol_a_conf = np.array(conformer.GetPositions(), dtype=np.float64)
            mol_a_conf = mol_a_conf/10 # convert to md_units

            conformer = mol_b.GetConformer(0)
            mol_b_conf = np.array(conformer.GetPositions(), dtype=np.float64)
            mol_b_conf = mol_b_conf/10 # convert to md_units

            x0 = np.concatenate([host_conf, mol_a_conf, mol_b_conf]) # combined geometry
            v0 = np.zeros_like(x0)

            parent_conn, child_conn = Pipe()

            input_args = (x0, v0, intg_seed, writer, child_conn, lambda_idx % args.num_gpus)
            p = Process(target=sim.run_forward_and_backward, args=input_args)

            all_pcs.append(parent_conn)
            all_processes.append(p)
            # all_args.append(args)

            # du_dls = sim.run_forward_and_backward(x0, v0, intg_seed, writer, lambda_idx % args.num_gpus)

            # all_du_dls.append(du_dls)

            # plt.plot(du_dls, label=str(lamb))

            # plt.ylabel("du_dl")
            # plt.xlabel("timestep")
            # plt.legend()
            # plt.savefig(os.path.join(args.out_dir, "lambda_du_dls"))

        mean_du_dls = []
        std_du_dls = []
        sum_du_dls = []

        for b_idx in range(0, len(all_processes), args.num_gpus):
            for p in all_processes[b_idx:b_idx+args.num_gpus]:
                p.start()


            batch_du_dls = []
            for pc_idx, pc in enumerate(all_pcs[b_idx:b_idx+args.num_gpus]):

                lamb_idx = b_idx+pc_idx
                lamb = ti_lambdas[b_idx+pc_idx]
                # TDB FIX ME
                offset = equil_T
                full_du_dls = pc.recv() # F, T
                assert full_du_dls is not None
                pc.send(None)

                mean_du_dls.append(np.mean(full_du_dls))
                std_du_dls.append(np.std(full_du_dls))

                for du_dls in full_du_dls:


                    print("lamb", lamb, "mean/std", np.mean(du_dls), np.std(du_dls))

                    plt.plot(du_dls, label=str(lamb))

                    plt.ylabel("du_dl")
                    plt.xlabel("timestep")
                    plt.legend()

                fpath = os.path.join(args.out_dir, "lambda_du_dls_"+str(pc_idx))
                plt.savefig(fpath)

                sum_du_dls.append(np.sum(full_du_dls, axis=0))
                all_du_dls.append(full_du_dls)

            for p in all_processes[b_idx:b_idx+args.num_gpus]:
                p.join()

        plt.close()

        plt.violinplot(sum_du_dls, positions=ti_lambdas)
        plt.ylabel("du_dlambda")
        plt.savefig(os.path.join(args.out_dir, "violin_du_dls"))
        plt.close()


        plt.boxplot(sum_du_dls, positions=ti_lambdas)
        plt.ylabel("du_dlambda")
        plt.savefig(os.path.join(args.out_dir, "boxplot_du_dls"))
        plt.close()

        print("mean_du_dls", mean_du_dls)
        print("pred_dG", np.trapz(mean_du_dls, ti_lambdas))

        np.save("all_du_dls", all_du_dls)