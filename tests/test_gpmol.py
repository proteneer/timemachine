# construct a relative transformation
from importlib import resources

import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from scipy.stats import special_ortho_group

from timemachine.constants import DEFAULT_ATOM_MAPPING_KWARGS
from timemachine.fe import atom_mapping, gpmol
from timemachine.fe.single_topology import SingleTopology
from timemachine.fe.system import simulate_system
from timemachine.fe.utils import get_romol_conf, read_sdf, recenter_mol, rotate_mol
from timemachine.ff import Forcefield


def process_core(gp_a, gp_b):
    cur_gp = gp_a
    path_gps = []

    counter = 0
    while True:
        counter += 1
        path_gps.append(cur_gp)

        new_gp = None
        for atom_idx_in_a in cur_gp.find_allowed_core_geometry_mutations(gp_b):
            new_gp = cur_gp.mutate_core_atom(atom_idx_in_a, gp_b)
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
    mol_a = all_mols[8]
    mol_b = all_mols[1]

    cores = atom_mapping.get_cores(mol_a, mol_b, **DEFAULT_ATOM_MAPPING_KWARGS)
    core = cores[0]

    ff = Forcefield.load_default()
    # bt_a = BaseTopology(mol_a, ff)
    # bt_b = BaseTopology(mol_b, ff)
    # from timemachine.fe.single_topology import AtomMapMixin
    # amm = AtomMapMixin(mol_a, mol_b, core)

    fpath = f"atom_mapping_{mol_a.GetProp('_Name')}_{mol_b.GetProp('_Name')}.svg"
    with open(fpath, "w") as fh:
        from timemachine.fe.utils import plot_atom_mapping_grid

        fh.write(plot_atom_mapping_grid(mol_a, mol_b, core))

    st = gpmol.SingleTopologyV5(mol_a, mol_b, core, ff)
    # pm_fwd = process(mol_a, core[:, 0])
    # pm_rev = process(mol_b, core[:, 1])
    # pm_core = process_core(pm_fwd[-1], pm_rev[-1])

    # pm_all = pm_fwd + pm_core + pm_rev[::-1]
    # pm_all = pm_fwd + pm_rev[::-1]
    # pm_all = pm_fwd
    pm_all = st.mol_a_path + st.mol_b_path[::-1]
    pm_all = [recenter_mol(pm.induced_mol()) for pm in pm_all]
    extra_rotations = generate_good_rotations(pm_all, num_rotations=3)

    extra_mols = []
    for rot in extra_rotations:
        for pm in pm_all:
            extra_mols.append(rotate_mol(pm, rot))

    svg = Draw.MolsToGridImage(pm_all + extra_mols, useSVG=True, molsPerRow=len(pm_all))

    fpath = "path_all.svg"
    with open(fpath, "w") as fh:
        fh.write(svg)

    for lamb_idx, lamb in enumerate(np.linspace(0, 1.0, 20)):
        # print("Processing lambda", lamb)
        i_state = st.setup_intermediate_state(lamb)
        i_mol, i_kv = st.setup_intermediate_mol_and_kv(lamb)

        # continue

        U_fn = i_state.get_U_fn()
        x_a = get_romol_conf(mol_a)
        x_b = get_romol_conf(mol_b)
        x0 = SingleTopology(mol_a, mol_b, core, ff).combine_confs(x_a, x_b, 0)
        frames = simulate_system(U_fn, x0, num_samples=200)

        with open(f"intermediate_{lamb_idx}.sdf", "w") as fh:
            writer = Chem.SDWriter(fh)
            for frame in frames:
                frame -= np.mean(frame, axis=0)
                mol_conf = Chem.Conformer(i_mol.GetNumAtoms())
                mol_copy = Chem.Mol(i_mol)
                for a_idx, pos in enumerate(frame):
                    if i_kv[a_idx] != -1:
                        mol_conf.SetAtomPosition(i_kv[a_idx], (pos * 10).astype(np.float64))
                mol_copy.AddConformer(mol_conf)
                writer.write(mol_copy)
            writer.close()

    assert 0

    i_mols, kvs = st.generate_intermediate_mols_and_kvs()
    # 0.3 relative to states 1/2, i.e. => 1.3333
    intermediate_state = gpmol.setup_intermediate_state_standard(0.3, st.checkpoint_states[2], st.checkpoint_states[3])

    # careful: need to use the right idx if we're going from the right!
    i_mol, kv = i_mols[2], kvs[2]

    print("i_mol atoms/bonds", i_mol.GetNumAtoms(), i_mol.GetNumBonds())

    print("Processing")
    U_fn = intermediate_state.get_U_fn()
    x_a = get_romol_conf(mol_a)
    x_b = get_romol_conf(mol_b)
    x0 = SingleTopology(mol_a, mol_b, core, ff).combine_confs(x_a, x_b, 0)
    frames = simulate_system(U_fn, x0, num_samples=200)

    with open(f"intermediate.sdf", "w") as fh:
        writer = Chem.SDWriter(fh)
        for frame in frames:
            frame -= np.mean(frame, axis=0)
            mol_conf = Chem.Conformer(i_mol.GetNumAtoms())
            mol_copy = Chem.Mol(i_mol)
            for a_idx, pos in enumerate(frame):
                if kv[a_idx] != -1:
                    mol_conf.SetAtomPosition(kv[a_idx], (pos * 10).astype(np.float64))
            mol_copy.AddConformer(mol_conf)
            writer.write(mol_copy)
        writer.close()

    # use left one right one depending if we're < left_idx or not.

    for idx, (vs, i_mol, kv) in enumerate(zip(st.checkpoint_states, i_mols, kvs)):
        print("i_mol atoms/bonds", i_mol.GetNumAtoms(), i_mol.GetNumBonds())

        print("Processing checkpoint state", idx)
        U_fn = vs.get_U_fn()
        x_a = get_romol_conf(mol_a)
        x_b = get_romol_conf(mol_b)
        x0 = SingleTopology(mol_a, mol_b, core, ff).combine_confs(x_a, x_b, 0)
        frames = simulate_system(U_fn, x0, num_samples=200)

        with open(f"traj_{idx}.sdf", "w") as fh:
            writer = Chem.SDWriter(fh)
            for frame in frames:
                frame -= np.mean(frame, axis=0)
                mol_conf = Chem.Conformer(i_mol.GetNumAtoms())
                mol_copy = Chem.Mol(i_mol)
                for a_idx, pos in enumerate(frame):
                    if kv[a_idx] != -1:
                        mol_conf.SetAtomPosition(kv[a_idx], (pos * 10).astype(np.float64))
                mol_copy.AddConformer(mol_conf)
                writer.write(mol_copy)
            writer.close()

        # assert 0

    assert 0

    # print(path_mols)
