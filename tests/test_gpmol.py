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
        for atom_group in cur_gp.find_simply_factorizable_atom_deletions():
            new_gp = cur_gp.turn_atoms_into_dummy(atom_group)
            break

        # no atom edits were found
        if new_gp is None:
            for atom_group in cur_gp.find_anchor_dummy_atom_deletions():
                new_gp = cur_gp.turn_atoms_into_dummy(atom_group)
                break

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


def test_gmol():
    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        all_mols = read_sdf(str(path_to_ligand))
    mol_a = all_mols[9]
    mol_b = all_mols[8]

    cores = atom_mapping.get_cores(mol_a, mol_b, **DEFAULT_ATOM_MAPPING_KWARGS)
    core = cores[0]

    fpath = f"atom_mapping_{mol_a.GetProp('_Name')}_{mol_b.GetProp('_Name')}.svg"
    with open(fpath, "w") as fh:
        from timemachine.fe.utils import plot_atom_mapping_grid

        fh.write(plot_atom_mapping_grid(mol_a, mol_b, core))

    print("fwd")
    pm_fwd = process(mol_a, core[:, 0])
    # print("bwd")
    pm_rev = process(mol_b, core[:, 1])
    # pm_core = process_core(pm_fwd[-1], pm_rev[-1])

    # pm_all = pm_fwd + pm_core + pm_rev[::-1]
    pm_all = pm_fwd + pm_rev[::-1]
    # pm_all = pm_fwd
    pm_all = [recenter_mol(pm.induced_mol()) for pm in pm_all]

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
