# construct a relative transformation
from importlib import resources

import numpy as np
from rdkit.Chem import Draw
from scipy.stats import special_ortho_group

from timemachine.constants import DEFAULT_ATOM_MAPPING_KWARGS
from timemachine.fe import atom_mapping, gpmol
from timemachine.fe.gpmol import AtomState, BondState, GPMol
from timemachine.fe.utils import get_romol_conf, read_sdf, recenter_mol, rotate_mol


def process(mol_a, core_atoms):
    atom_primitives_a = gpmol.initialize_atom_primitives(mol_a)

    atom_states_a = np.array([AtomState.INTERACTING for _ in range(mol_a.GetNumAtoms())])
    bond_states_a = np.array([BondState.INTERACTING for _ in range(mol_a.GetNumBonds())])
    gp_a = GPMol(mol_a, core_atoms, atom_primitives_a, atom_states_a, bond_states_a)

    dgas, dsg = gp_a.find_dummy_groups()

    print(dgas)
    # print(dgas, dgas)

    # for k, v in dgas.items():
    #     if len(v) > 1:
    #         print(gp_a.split_multiple_anchor_dummy_groups(k, v))

    # assert 0
    # assert 0
    print("!", gp_a.find_allowed_bond_deletions())

    return [gp_a]

    cur_gp = gp_a
    path_gps = []

    counter = 0
    while True:
        svg = cur_gp.draw_mol()
        fpath = f"mol_{counter}.svg"
        with open(fpath, "w") as fh:
            fh.write(svg)

        counter += 1
        path_gps.append(cur_gp)

        new_gp = None
        for atom_group in cur_gp.find_allowed_atom_deletions():
            print("deleting atom_group", atom_group)
            new_gp = cur_gp.turn_atoms_into_dummy(atom_group)
            break

        # no atom edits were found
        if new_gp is None:
            for src, dst in cur_gp.find_allowed_bond_deletions():
                new_gp = cur_gp.delete_bond(src, dst)
                break

        if new_gp:
            cur_gp = new_gp
        else:
            break

    return path_gps


def process_core(gp_a, gp_b):
    cur_gp = gp_a
    path_gps = []

    counter = 0
    while True:
        counter += 1
        path_gps.append(cur_gp)

        new_gp = None
        for atom_idx_in_a in cur_gp.find_allowed_core_mutations(gp_b):
            new_gp = cur_gp.mutate_atom(atom_idx_in_a, gp_b)
            break

        if new_gp:
            cur_gp = new_gp
        else:
            break

    return path_gps


def get_hif2a_ligand_pair_single_topology(a_idx, b_idx):
    """Return two ligands from hif2a and the manually specified atom mapping"""

    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        all_mols = read_sdf(str(path_to_ligand))

    mol_a = all_mols[a_idx]
    mol_b = all_mols[b_idx]

    cores = atom_mapping.get_cores(mol_a, mol_b, **DEFAULT_ATOM_MAPPING_KWARGS)
    core = cores[0]

    return mol_a, mol_b, core


def score_2d(conf, norm=2):
    # get the goodness of a 2D depiction
    # low_score = good, high_score = bad

    score = 0
    for idx, (x0, y0, _) in enumerate(conf):
        for x1, y1, _ in conf[idx + 1 :]:
            score += 1 / ((x0 - x1) ** norm + (y0 - y1) ** norm)

    return score / len(conf)


def generate_good_rotations(
    mols,
    num_rotations: int = 3,
    max_rotations: int = 1000,
    seed: int = 1234,
):
    assert num_rotations < max_rotations
    # generate some good rotations so that the viewing angle is pleasant, (so clashes are minimized):
    confs = [get_romol_conf(mol) for mol in mols]

    unif_so3 = special_ortho_group(dim=3, seed=seed)

    scores = []
    rotations = []
    for _ in range(max_rotations):
        r = unif_so3.rvs()
        s = [score_2d(x @ r.T) for x in confs]
        # score_b = score_2d(conf_b @ r.T)
        # take the bigger of the two scores
        scores.append(max(s))
        rotations.append(r)

    perm = np.argsort(scores, kind="stable")
    return np.array(rotations)[perm][:num_rotations]


from rdkit import Chem


def test_gmol():
    all_mols = read_sdf("/home/yzhao/Code/timemachine/timemachine/testsystems/data/ligands_40.sdf")
    mol_a = all_mols[8]
    mol_b = all_mols[1]

    cores = atom_mapping.get_cores(mol_a, mol_b, **DEFAULT_ATOM_MAPPING_KWARGS)
    core = cores[0]

    #     mol_a = Chem.MolFromMolBlock("""
    #   Mrv2311 08182400222D

    #  12 12  0  0  0  0            999 V2000
    #    -7.2657    2.1197    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    #    -7.9801    1.7072    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    #    -7.9801    0.8821    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    #    -7.2657    0.4696    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    #    -6.5512    0.8821    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    #    -6.5512    1.7072    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    #    -7.2657    2.9447    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    #    -8.6945    2.1197    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    #    -8.6945    0.4696    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    #    -7.2657   -0.3554    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    #    -5.8367    0.4696    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    #    -5.8367    2.1197    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    #   1  2  2  0  0  0  0
    #   2  3  1  0  0  0  0
    #   3  4  2  0  0  0  0
    #   4  5  1  0  0  0  0
    #   5  6  2  0  0  0  0
    #   6  1  1  0  0  0  0
    #   1  7  1  0  0  0  0
    #   2  8  1  0  0  0  0
    #   3  9  1  0  0  0  0
    #   4 10  1  0  0  0  0
    #   5 11  1  0  0  0  0
    #   6 12  1  0  0  0  0
    # M  END
    # $$$$""", removeHs=False)

    #     mol_b = Chem.MolFromMolBlock("""
    #   Mrv2311 08182400222D

    #  12 12  0  0  0  0            999 V2000
    #    -7.2657    2.1197    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    #    -7.9801    1.7072    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    #    -7.9801    0.8821    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    #    -7.2657    0.4696    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    #    -6.5512    0.8821    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    #    -6.5512    1.7072    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    #    -7.2657    2.9447    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    #    -8.6945    2.1197    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    #    -8.6945    0.4696    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    #    -7.2657   -0.3554    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    #    -5.8367    0.4696    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    #    -5.8367    2.1197    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    #   1  2  2  0  0  0  0
    #   2  3  1  0  0  0  0
    #   3  4  2  0  0  0  0
    #   4  5  1  0  0  0  0
    #   5  6  2  0  0  0  0
    #   6  1  1  0  0  0  0
    #   1  7  1  0  0  0  0
    #   2  8  1  0  0  0  0
    #   3  9  1  0  0  0  0
    #   4 10  1  0  0  0  0
    #   5 11  1  0  0  0  0
    #   6 12  1  0  0  0  0
    # M  END
    # $$$$""", removeHs=False)

    #     # core = np.array([[x, x] for x in range(mol_a.GetNumAtoms())])
    #     core = np.array( [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10]])
    #     # print(core.tolist())
    fpath = f"atom_mapping_{mol_a.GetProp('_Name')}_{mol_b.GetProp('_Name')}.svg"
    with open(fpath, "w") as fh:
        from timemachine.fe.utils import plot_atom_mapping_grid

        fh.write(plot_atom_mapping_grid(mol_a, mol_b, core))

    print("fwd")
    pm_fwd = process(mol_a, core[:, 0])
    print("bwd")
    pm_rev = process(mol_b, core[:, 1])
    # pm_core = process_core(pm_fwd[-1], pm_rev[-1])

    # pm_all = pm_fwd + pm_core + pm_rev[::-1]
    pm_all = pm_fwd + pm_rev[::-1]
    # pm_all = pm_fwd
    pm_all = [recenter_mol(pm.induced_mol()) for pm in pm_all]

    # writer = Chem.SDWriter("out.sdf")
    # for idx, mol in enumerate(pm_all):
    #     mol.SetProp("_Name", str(idx))
    #     writer.write(mol)
    # writer.close()

    extra_rotations = generate_good_rotations(pm_all, num_rotations=3)

    extra_mols = []
    for rot in extra_rotations:
        for pm in pm_all:
            extra_mols.append(rotate_mol(pm, rot))

    svg = Draw.MolsToGridImage(pm_all + extra_mols, useSVG=True, molsPerRow=len(pm_all))

    fpath = "mol_a_path_all.svg"
    with open(fpath, "w") as fh:
        fh.write(svg)

    # print(path_mols)
