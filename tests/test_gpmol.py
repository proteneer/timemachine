# construct a relative transformation
from importlib import resources

import numpy as np
from rdkit.Chem import Draw

from timemachine.constants import DEFAULT_ATOM_MAPPING_KWARGS
from timemachine.fe import atom_mapping, gpmol
from timemachine.fe.gpmol import AtomState, BondState, GPMol
from timemachine.fe.utils import plot_atom_mapping_grid, read_sdf


def get_hif2a_ligand_pair_single_topology(a_idx, b_idx):
    """Return two ligands from hif2a and the manually specified atom mapping"""

    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        all_mols = read_sdf(str(path_to_ligand))

    mol_a = all_mols[a_idx]
    mol_b = all_mols[b_idx]

    cores = atom_mapping.get_cores(mol_a, mol_b, **DEFAULT_ATOM_MAPPING_KWARGS)
    core = cores[0]

    return mol_a, mol_b, core


def process(mol_idx_a, mol_idx_b):
    print(f"processing {mol_idx_a} -> {mol_idx_b}")

    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology(mol_idx_a, mol_idx_b)

    fpath = f"atom_mapping_{mol_idx_a}_{mol_idx_b}.svg"
    with open(fpath, "w") as fh:
        fh.write(plot_atom_mapping_grid(mol_a, mol_b, core))

    core = np.array(core, dtype=np.int32)

    atom_primitives_a = gpmol.initialize_atom_primitives(mol_a)

    atom_states_a = np.array([AtomState.REAL for _ in range(mol_a.GetNumAtoms())])
    bond_states_a = np.array([BondState.REAL for _ in range(mol_a.GetNumBonds())])
    gp_a = GPMol(mol_a, core[:, 0], atom_primitives_a, atom_states_a, bond_states_a)
    cur_gp = gp_a
    path_mols = []

    counter = 0
    while True:
        svg = cur_gp.draw_mol()

        fpath = f"mol_{counter}.svg"
        with open(fpath, "w") as fh:
            fh.write(svg)

        counter += 1
        path_mols.append(cur_gp.induced_mol())

        new_gp = None
        for atom_edit in cur_gp.find_allowed_atom_edits():
            new_gp = cur_gp.turn_atom_into_dummy(atom_edit)
            break

        # no atom edits were found
        if new_gp is None:
            for src, dst in cur_gp.find_allowed_bond_edits():
                new_gp = cur_gp.delete_bond(src, dst)
                break

        if new_gp:
            cur_gp = new_gp
        else:
            break

    return path_mols


def test_gmol():
    pm_fwd = process(8, 1)
    pm_rev = process(1, 8)

    pm_all = pm_fwd + pm_rev[::-1]

    # writer = Chem.SDWriter("out.sdf")
    # for idx, mol in enumerate(pm_all):
    #     mol.SetProp("_Name", str(idx))
    #     writer.write(mol)
    # writer.close()

    svg = Draw.MolsToGridImage(pm_all, useSVG=True, molsPerRow=len(pm_all))

    fpath = "mol_a_path_all.svg"
    with open(fpath, "w") as fh:
        fh.write(svg)

    # print(path_mols)
