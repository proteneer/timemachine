import jax
import matplotlib.pyplot as plt
import numpy as np
import pytest

from timemachine.constants import BOLTZ, DEFAULT_ATOM_MAPPING_KWARGS
from timemachine.fe import atom_mapping, cif_writer, single_topology, utils
from timemachine.fe.bar import bar
from timemachine.fe.system import simulate_system
from timemachine.fe.utils import get_romol_conf
from timemachine.ff import Forcefield
from timemachine.utils import path_to_internal_file


@pytest.mark.skip(reason="This is currently too slow to run on CI")
def test_hif2a_free_energy_estimates():
    # Test that we can estimate the free energy differences for some simple transformations

    forcefield = Forcefield.load_default()

    with path_to_internal_file("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        all_mols = utils.read_sdf(path_to_ligand)

    mol_a = all_mols[1]
    mol_b = all_mols[4]

    all_cores = atom_mapping.get_cores(
        mol_a,
        mol_b,
        **DEFAULT_ATOM_MAPPING_KWARGS,
    )
    core = all_cores[0]
    svg = utils.plot_atom_mapping_grid(mol_a, mol_b, core)
    with open("atom_mapping.svg", "w") as fh:
        fh.write(svg)

    st = single_topology.SingleTopology(mol_a, mol_b, core, forcefield)

    lambda_schedule = np.linspace(0.0, 1.0, 12)
    systems = [st.setup_intermediate_state(lamb) for lamb in lambda_schedule]
    U_fns = [sys.get_U_fn() for sys in systems]

    batch_U_fns = [jax.vmap(x) for x in U_fns]

    all_frames = []

    x0 = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b))

    kT = BOLTZ * 300.0
    beta = 1 / kT

    for lambda_idx, U_fn in enumerate(U_fns):
        # print("lambda", lambda_schedule[lambda_idx], "U", U_fn(x0))
        # continue
        frames = simulate_system(U_fn, x0, num_samples=2000)
        all_frames.append(frames)
        writer = cif_writer.CIFWriter([mol_a, mol_b], "debug_" + str(lambda_idx) + ".cif")
        for f in frames:
            fc = cif_writer.convert_single_topology_mols(f, st)
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
            plt.savefig(f"lambda_{lambda_idx - 1}_{lambda_idx}.png")

            dG_exact, exact_bar_err = bar(fwd_delta_u, rev_delta_u)
            dG_exact /= beta
            exact_bar_err /= beta

            print(
                f"BAR: lambda {lambda_schedule[lambda_idx - 1]:.3f} -> {lambda_schedule[lambda_idx]:.3f} dG: {dG_exact:.3f} dG_err: {exact_bar_err:.3f}"
            )
