# test that we can estimate free energies reliably using pair bar.
import multiprocessing
import os
from importlib import resources

import jax
import matplotlib.pyplot as plt
import numpy as np
import pymbar
from rdkit import Chem

from timemachine.constants import BOLTZ
from timemachine.fe import pdb_writer, single_topology_v3
from timemachine.fe.system import simulate_system
from timemachine.fe.utils import get_romol_conf
from timemachine.ff import Forcefield

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(multiprocessing.cpu_count())


def test_hif2a_free_energy_estimates():
    # Test that we can estimate the free energy differences for some simple transformations

    forcefield = Forcefield.load_from_file("smirnoff_1_1_0_ccc.py")

    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        suppl = Chem.SDMolSupplier(str(path_to_ligand), removeHs=False)

    all_mols = [x for x in suppl]
    mol_a = all_mols[1]
    mol_b = all_mols[4]

    core = np.array(
        [
            [0, 0],
            [2, 2],
            [1, 1],
            [6, 6],
            [5, 5],
            [4, 4],
            [3, 3],
            [15, 16],
            [16, 17],
            [17, 18],
            [18, 19],
            [19, 20],
            [20, 21],
            [32, 30],
            [26, 25],
            [27, 26],
            [7, 7],
            [8, 8],
            [9, 9],
            [10, 10],
            [29, 11],
            [11, 12],
            [12, 13],
            [14, 15],
            [31, 29],
            [13, 14],
            [23, 24],
            [30, 28],
            [28, 27],
            [21, 22],
        ]
    )

    st = single_topology_v3.SingleTopologyV3(mol_a, mol_b, core, forcefield)

    # lambda_schedule = np.linspace(0.0, 1.0, 5)
    lambda_schedule = np.linspace(0.0, 1.0, 12)
    # lambda_schedule = [0.4444444444444444]
    # lambda_schedule = [0.0]
    systems = [st.setup_intermediate_state(lamb) for lamb in lambda_schedule]
    U_fns = [sys.get_U_fn() for sys in systems]

    batch_U_fns = [jax.vmap(x) for x in U_fns]

    all_frames = []

    x0 = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b))

    kT = BOLTZ * 300.0
    beta = 1 / kT

    for lambda_idx, U_fn in enumerate(U_fns):
        print("lambda", lambda_schedule[lambda_idx], "U", U_fn(x0))
        # continue
        frames = simulate_system(U_fn, x0, num_samples=2000)
        all_frames.append(frames)
        writer = pdb_writer.PDBWriter([mol_a, mol_b], "debug_" + str(lambda_idx) + ".pdb")
        for f in frames:
            fc = pdb_writer.convert_single_topology_mols(f, st)
            fc = fc - np.mean(fc, axis=0)
            writer.write_frame(fc * 10)
        writer.close()

        if lambda_idx > 0:

            prev_frames = all_frames[lambda_idx - 1]
            cur_frames = all_frames[lambda_idx]

            prev_U_fn = batch_U_fns[lambda_idx - 1]
            cur_U_fn = batch_U_fns[lambda_idx]

            fwd_delta_u = beta * (cur_U_fn(prev_frames) - prev_U_fn(prev_frames))
            rev_delta_u = beta * (prev_U_fn(cur_frames) - cur_U_fn(cur_frames))

            plt.clf()
            plt.hist(fwd_delta_u, alpha=0.5, label="fwd")
            plt.hist(-rev_delta_u, alpha=0.5, label="-rev")
            plt.legend()
            plt.savefig(f"lambda_{lambda_idx-1}_{lambda_idx}.png")

            dG_exact, exact_bar_err = pymbar.BAR(fwd_delta_u, rev_delta_u)
            dG_exact /= beta
            exact_bar_err /= beta

            print(
                f"BAR: lambda {lambda_schedule[lambda_idx-1]:.3f} -> {lambda_schedule[lambda_idx]:.3f} dG: {dG_exact:.3f} dG_err: {exact_bar_err:.3f}"
            )
