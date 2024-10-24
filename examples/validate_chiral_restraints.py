import os
import pickle
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem

from timemachine.constants import DEFAULT_ATOM_MAPPING_KWARGS
from timemachine.fe import atom_mapping
from timemachine.fe.rbfe import DEFAULT_HREX_PARAMS, run_solvent
from timemachine.fe.utils import plot_atom_mapping_grid
from timemachine.ff import Forcefield


def plot_and_save(f, fname, *args, **kwargs):
    """
    Given a function which generates a plot, return the plot as png bytes.
    """
    plt.clf()
    f(*args, **kwargs)
    with open(fname, "wb") as fh:
        plt.savefig(fh, format="png", bbox_inches="tight")


from timemachine.fe.plots import (
    plot_hrex_replica_state_distribution,
    plot_hrex_replica_state_distribution_heatmap,
    plot_hrex_swap_acceptance_rates_convergence,
    plot_hrex_transition_matrix,
)


def get_hif2a_truncated():
    mol_a = Chem.MolFromMolBlock(
        """hif2a_1
                    3D
 Schrodinger Suite 2023-3.
 14 14  0  0  1  0            999 V2000
   27.5742    1.5579  -11.2202 O   0  0  0  0  0  0
   27.1479    0.2994  -11.6692 C   0  0  2  0  0  0
   26.5267    0.4129  -13.0689 C   0  0  0  0  0  0
   25.0255    0.5956  -12.8327 C   0  0  0  0  0  0
   24.8103   -0.0581  -11.4965 C   0  0  0  0  0  0
   26.0181   -0.2298  -10.8141 C   0  0  0  0  0  0
   26.0194   -0.6446   -9.8169 H   0  0  0  0  0  0
   23.8782   -0.3505  -11.0360 H   0  0  0  0  0  0
   28.3584    1.8102  -11.7281 H   0  0  0  0  0  0
   27.9789   -0.4080  -11.7029 H   0  0  0  0  0  0
   26.6995   -0.5291  -13.6482 H   0  0  0  0  0  0
   26.9532    1.2021  -13.6359 H   0  0  0  0  0  0
   24.7973    1.6781  -12.7806 H   0  0  0  0  0  0
   24.4358    0.1715  -13.6220 H   0  0  0  0  0  0
  1  2  1  0  0  0
  1  9  1  0  0  0
  2  3  1  0  0  0
  2  6  1  0  0  0
  2 10  1  0  0  0
  3  4  1  0  0  0
  3 11  1  0  0  0
  3 12  1  0  0  0
  4  5  1  0  0  0
  4 13  1  0  0  0
  4 14  1  0  0  0
  5  6  2  0  0  0
  5  8  1  0  0  0
  6  7  1  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    mol_b = Chem.MolFromMolBlock(
        """hif2a_7b
                    3D
 Schrodinger Suite 2023-3.
 17 17  0  0  1  0            999 V2000
   27.5013   -1.6332  -11.4637 O   0  0  0  0  0  0
   27.1568   -0.2839  -11.3999 C   0  0  1  0  0  0
   26.6886    0.3329  -12.7025 C   0  0  0  0  0  0
   25.1969    0.0379  -12.7689 C   0  0  1  0  0  0
   24.8105   -0.1505  -11.3108 C   0  0  0  0  0  0
   25.9592   -0.2656  -10.5261 C   0  0  0  0  0  0
   25.9110   -0.5491   -9.4851 H   0  0  0  0  0  0
   23.8387   -0.2969  -10.8629 H   0  0  0  0  0  0
   24.4433    1.1364  -13.4936 C   0  0  0  0  0  0
   26.7744   -2.1461  -11.8362 H   0  0  0  0  0  0
   27.9807    0.3158  -11.0121 H   0  0  0  0  0  0
   26.8561    1.4115  -12.6450 H   0  0  0  0  0  0
   27.2249   -0.0134  -13.5849 H   0  0  0  0  0  0
   25.0496   -0.9005  -13.3053 H   0  0  0  0  0  0
   23.4853    0.7668  -13.8302 H   0  0  0  0  0  0
   24.9806    1.4869  -14.3669 H   0  0  0  0  0  0
   24.2853    1.9691  -12.8189 H   0  0  0  0  0  0
  1  2  1  0  0  0
  1 10  1  0  0  0
  2  3  1  0  0  0
  2  6  1  0  0  0
  2 11  1  0  0  0
  3  4  1  0  0  0
  3 12  1  0  0  0
  3 13  1  0  0  0
  4  5  1  0  0  0
  4  9  1  0  0  0
  4 14  1  0  0  0
  5  6  2  0  0  0
  5  8  1  0  0  0
  6  7  1  0  0  0
  9 15  1  0  0  0
  9 16  1  0  0  0
  9 17  1  0  0  0
M  END""",
        removeHs=False,
    )

    core = atom_mapping.get_cores(
        mol_a,
        mol_b,
        **DEFAULT_ATOM_MAPPING_KWARGS,
    )[0]

    return mol_a, mol_b, core


def get_simple_pair():
    mol_a = Chem.MolFromMolBlock(
        """identity_ring_pair
  Mrv2311 10092413403D

  6  6  0  0  0  0            999 V2000
    0.1292    1.5540   -0.4103 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.8029    0.8102    0.1698 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.0572    0.0397    0.8217 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.1063    0.7870    0.2530 C   0  0  2  0  0  0  0  0  0  0  0  0
    2.1161    1.7939    1.5497 F   0  0  0  0  0  0  0  0  0  0  0  0
    1.8910    0.0536   -0.6026 Cl  0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  1  0  0  0  0
  3  4  1  0  0  0  0
  1  4  1  0  0  0  0
  4  5  1  0  0  0  0
  4  6  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    # keeps 2 chiral restraints
    perm = [0, 1, 2, 3, 4, 5]

    # keeps all chiral restraints
    # perm = [3, 4, 5, 0, 2, 1]

    perm_kv = {}
    for new_idx, old_idx in enumerate(perm):
        perm_kv[old_idx] = new_idx

    mol_a = Chem.RenumberAtoms(mol_a, perm)
    mol_a.SetProp("_Name", "mol")
    mol_b = Chem.Mol(mol_a)

    core = np.array([[1, 1], [2, 2], [3, 3], [4, 5]], dtype=np.int32)
    core_perm = []
    for i, j in core:
        core_perm.append([perm_kv[i], perm_kv[j]])
    core = np.array(core_perm, dtype=np.int32)

    return mol_a, mol_b, core


def get_alt_pair():
    # cyclobutane ring
    mol_a = Chem.MolFromMolBlock(
        """alt_pair mol_a
                    3D
 Structure written by MMmdl.
 10 10  0  0  1  0            999 V2000
   61.0392  -33.5497  -35.2867 F   0  0  0  0  0  0
   60.3271  -33.5255  -36.4685 C   0  0  1  0  0  0
   61.0252  -33.1997  -37.8026 S   0  0  0  0  0  0
   60.0215  -34.1193  -38.5010 C   0  0  0  0  0  0
   59.8393  -34.8162  -37.1515 C   0  0  0  0  0  0
   59.4787  -32.8508  -36.3286 H   0  0  0  0  0  0
   60.4356  -34.7522  -39.2862 H   0  0  0  0  0  0
   59.1170  -33.6045  -38.8304 H   0  0  0  0  0  0
   58.8113  -35.1061  -36.9244 H   0  0  0  0  0  0
   60.5281  -35.6523  -37.0189 H   0  0  0  0  0  0
  1  2  1  0  0  0
  2  3  1  0  0  0
  2  5  1  0  0  0
  2  6  1  0  0  0
  3  4  1  0  0  0
  4  5  1  0  0  0
  4  7  1  0  0  0
  4  8  1  0  0  0
  5  9  1  0  0  0
  5 10  1  0  0  0
M  END

$$$$""",
        removeHs=False,
    )

    mol_b = Chem.MolFromMolBlock(
        """alt_pair mol_b
                    3D
 Structure written by MMmdl.
 14 14  0  0  1  0            999 V2000
   60.9363  -33.6820  -35.2687 F   0  0  0  0  0  0
   60.3022  -33.9870  -36.4558 C   0  0  1  0  0  0
   60.2039  -32.8187  -37.4561 S   0  0  0  0  0  0
   59.6848  -33.3119  -38.8168 C   0  0  0  0  0  0
   60.5213  -34.3539  -39.3031 O   0  0  0  0  0  0
   60.4244  -35.5141  -38.4856 C   0  0  0  0  0  0
   61.0143  -35.1860  -37.1086 C   0  0  0  0  0  0
   59.2791  -34.2938  -36.2244 H   0  0  0  0  0  0
   59.7052  -32.4975  -39.5385 H   0  0  0  0  0  0
   58.6547  -33.6631  -38.7462 H   0  0  0  0  0  0
   60.9837  -36.3255  -38.9505 H   0  0  0  0  0  0
   59.3881  -35.8507  -38.4036 H   0  0  0  0  0  0
   60.9455  -36.0595  -36.4589 H   0  0  0  0  0  0
   62.0769  -34.9691  -37.2220 H   0  0  0  0  0  0
  1  2  1  0  0  0
  2  3  1  0  0  0
  2  7  1  0  0  0
  2  8  1  0  0  0
  3  4  1  0  0  0
  4  5  1  0  0  0
  4  9  1  0  0  0
  4 10  1  0  0  0
  5  6  1  0  0  0
  6  7  1  0  0  0
  6 11  1  0  0  0
  6 12  1  0  0  0
  7 13  1  0  0  0
  7 14  1  0  0  0
M  END

$$$$
""",
        removeHs=False,
    )
    import copy

    atom_mapping_kwargs = copy.deepcopy(DEFAULT_ATOM_MAPPING_KWARGS)
    atom_mapping_kwargs["enforce_core_core"] = True
    atom_mapping_kwargs["ring_matches_ring_only"] = False
    atom_mapping_kwargs["ring_cutoff"] = 0.2
    atom_mapping_kwargs["chain_cutoff"] = 0.2

    core = atom_mapping.get_cores(
        mol_a,
        mol_b,
        **atom_mapping_kwargs,
    )[0]

    return mol_a, mol_b, core


def get_hif2a_ring():
    from importlib import resources

    from timemachine.fe.utils import read_sdf

    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        mols = read_sdf(path_to_ligand)

    mol_a = mols[1]
    mol_b = mols[-7]

    print(mol_a.GetProp("_Name"), "->", mol_b.GetProp("_Name"))

    core = atom_mapping.get_cores(
        mol_a,
        mol_b,
        **DEFAULT_ATOM_MAPPING_KWARGS,
    )[0]

    return mol_a, mol_b, core


def test_ring_breaking_chiral_restraints_failure(mol_a, mol_b, core):
    res = plot_atom_mapping_grid(mol_a, mol_b, core)

    base_dir = f"result_{mol_a.GetProp('_Name')}_{mol_b.GetProp('_Name')}"
    try:
        os.makedirs(base_dir)
    except Exception as e:
        print("Pass", e)

    fpath = "atom_mapping.svg"
    print("core mapping written to", fpath)
    with open(os.path.join(base_dir, fpath), "w") as fh:
        fh.write(res)

    short_hrex_params = replace(DEFAULT_HREX_PARAMS, n_frames=2000, n_eq_steps=10000, steps_per_frame=400)
    ff = Forcefield.load_default()

    # sim_res = run_vacuum(mol_a, mol_b, core, ff, None, short_hrex_params, n_windows=48, min_overlap=0.66)
    sim_res = run_solvent(mol_a, mol_b, core, ff, None, short_hrex_params, n_windows=48, min_overlap=0.66)

    print("dGs", sim_res.final_result.dGs, "sum", sum(res.final_result.dGs))

    with open(os.path.join(base_dir, "sim_res.pkl"), "wb") as fh:
        pickle.dump((sim_res, mol_a, mol_b, core), fh)

    with open(os.path.join(base_dir, "vacuum_overlap.png"), "wb") as fh:
        fh.write(sim_res.plots.overlap_detail_png)

    plot_and_save(
        plot_hrex_swap_acceptance_rates_convergence,
        os.path.join(base_dir, "vac_plot_hrex_swap_acceptance_rates_convergence.png"),
        sim_res.hrex_diagnostics.cumulative_swap_acceptance_rates,
    )
    plot_and_save(
        plot_hrex_transition_matrix,
        os.path.join(base_dir, "vac_plot_hrex_transition_matrix.png"),
        sim_res.hrex_diagnostics.transition_matrix,
    )

    plot_and_save(
        plot_hrex_replica_state_distribution,
        os.path.join(base_dir, "vac_plot_hrex_replica_state_distribution.png"),
        sim_res.hrex_diagnostics.cumulative_replica_state_counts,
    )
    plot_and_save(
        plot_hrex_replica_state_distribution_heatmap,
        os.path.join(base_dir, "vac_plot_hrex_replica_state_distribution_heatmap.png"),
        sim_res.hrex_diagnostics.cumulative_replica_state_counts,
    )


if __name__ == "__main__":
    # mol_a, mol_b, core = get_simple_pair()
    # mol_a, mol_b, core = get_alt_pair()
    # mol_a, mol_b, core = get_hif2a_truncated()
    # mol_a, mol_b, core = get_hif2a_ring()

    fns = [get_simple_pair, get_alt_pair, get_hif2a_truncated, get_hif2a_ring]

    for fn in fns:
        mol_a, mol_b, core = fn()
        test_ring_breaking_chiral_restraints_failure(mol_a, mol_b, core)
