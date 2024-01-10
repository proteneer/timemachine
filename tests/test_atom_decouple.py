import matplotlib.pyplot as plt
import numpy as np
import pytest
from rdkit.Chem import rdDistGeom

from timemachine.constants import DEFAULT_ATOM_MAPPING_KWARGS
from timemachine.fe.atom_decouple import (
    estimate_avg_volumes_along_schedule,
    estimate_radii,
    estimate_volume,
    estimate_volumes_along_schedule,
)
from timemachine.fe.atom_mapping import get_cores
from timemachine.fe.single_topology import SingleTopology
from timemachine.fe.utils import get_multiple_romol_confs, get_romol_conf, read_sdf
from timemachine.ff import Forcefield


def test_atom_by_atom_decouple():
    mols = read_sdf("tests/data/benzene_subs.sdf")

    benzene_mono_sub_top = mols[0]
    benzene_mono_sub_bot = mols[1]
    benzene_di_both = mols[2]
    benzene_no_sub = mols[3]

    pairs = [
        (benzene_mono_sub_top, benzene_mono_sub_bot),
        (benzene_no_sub, benzene_mono_sub_top),
        (benzene_no_sub, benzene_di_both),
    ]

    lamb_schedule = np.linspace(0, 1.0, 24)
    lamb_idxs = np.arange(len(lamb_schedule))
    ff = Forcefield.load_default()

    for mol_a, mol_b in pairs:
        label = mol_a.GetProp("_Name") + " -> " + mol_b.GetProp("_Name")
        print("processing", label)
        core = get_cores(mol_a, mol_b, **DEFAULT_ATOM_MAPPING_KWARGS)[0]
        print(core)
        vols = estimate_volumes_along_schedule(mol_a, mol_b, core, ff, lamb_schedule)

        plt.plot(lamb_idxs, vols, label=label)

    plt.xlabel("lamb_idx")
    plt.ylabel("volume")
    plt.legend()
    plt.show()


def test_greedy_heuristic():
    mols = read_sdf("tests/data/benzene_subs.sdf")

    benzene_mono_sub_top = mols[0]
    benzene_mono_sub_bot = mols[1]
    benzene_di_both = mols[2]
    benzene_no_sub = mols[3]

    pairs = [
        (benzene_mono_sub_top, benzene_mono_sub_bot),
        (benzene_no_sub, benzene_mono_sub_top),
        (benzene_no_sub, benzene_di_both),
    ]

    # lamb_schedule = np.linspace(0, 1.0, 24)
    # lamb_idxs = np.arange(len(lamb_schedule))
    ff = Forcefield.load_default()
    cutoff = 1.2

    for mol_a, mol_b in pairs:
        conf_a = get_romol_conf(mol_a)
        conf_b = get_romol_conf(mol_b)
        label = mol_a.GetProp("_Name") + " -> " + mol_b.GetProp("_Name")
        print("processing", label)
        core = get_cores(mol_a, mol_b, **DEFAULT_ATOM_MAPPING_KWARGS)[0]

        st = SingleTopology(mol_a, mol_b, core, ff)
        conf_c = st.combine_confs(conf_a, conf_b, 0.0)

        dummy_atoms_a = st.get_dummy_atoms_in_mol_a()  # "toggle" turns these off
        dummy_atoms_b = st.get_dummy_atoms_in_mol_b()  # "toggle" turns these on

        # Q: does this generate a symmetrized sequence? (i.e. rank fwd = rank rev[::-1])
        # start from lamb = 0
        cur_qwljs = st._get_guest_params(st.ff.q_handle_solv, st.ff.lj_handle_solv, lamb=0.0, cutoff=1.2)
        cur_vol = estimate_volume(conf_c, estimate_radii(cur_qwljs), n_samples=1000000)
        cur_choices = set()
        for atom in dummy_atoms_a:
            cur_choices.add((atom, "A"))
        for atom in dummy_atoms_b:
            cur_choices.add((atom, "B"))

        print(cur_choices)

        toggle_order = []

        # picking the smallest change isn't good, and not very monotonic

        while len(cur_choices) > 0:
            test_vols = []
            test_choices = []
            test_qljws = []
            for atom_idx, state in cur_choices:
                test_choices.append((atom_idx, state))
                qljws = np.copy(cur_qwljs)

                # toggle state, can probably do this implicitly without storing state (via ~| trick)
                if state == "A":
                    qljws[atom_idx, -1] = cutoff
                elif state == "B":
                    qljws[atom_idx, -1] = 0

                test_qljws.append(qljws)

                new_radii = estimate_radii(qljws)
                new_vol = estimate_volume(conf_c, new_radii, n_samples=100000)
                # print("test toggle", cur_qwljs[atom_idx, -1], "->", qljws[atom_idx, -1], "vol", new_vol)
                test_vols.append(new_vol)

            # pick the move that changes the volume the least
            choice_idx = np.argmin(np.abs(test_vols - cur_vol))
            choice_item = test_choices[choice_idx]

            toggle_order.append(choice_item)
            cur_vol = test_vols[choice_idx]
            cur_qwljs = test_qljws[choice_idx]
            cur_choices.remove(choice_item)
            print("picked", choice_item, "cur_vol", cur_vol)


@pytest.mark.skip(reason="this currently runs way too slow to be of any practical use")
def test_avg_volume_along_lambda():
    mols = read_sdf("tests/data/benzene_subs.sdf")

    benzene_mono_sub_top = mols[0]
    benzene_mono_sub_bot = mols[1]
    benzene_di_both = mols[2]
    benzene_no_sub = mols[3]

    pairs = [
        (benzene_no_sub, benzene_mono_sub_top),
        (benzene_no_sub, benzene_di_both),
        (benzene_mono_sub_top, benzene_mono_sub_bot),
    ]

    ff = Forcefield.load_default()
    # generate atom-mapping before we blow away the conformers
    topologies = []
    for mol_a, mol_b in pairs:
        core = get_cores(mol_a, mol_b, **DEFAULT_ATOM_MAPPING_KWARGS)[0]
        topologies.append(SingleTopology(mol_a, mol_b, core, ff))

    n_frames = 10
    n_windows = 24
    lamb_schedule = np.linspace(0, 1.0, n_windows)
    lamb_idxs = np.arange(len(lamb_schedule))

    # generate new trajectories
    for mol in mols:
        mol.RemoveAllConformers()
        print("Embedding", mol.GetProp("_Name"))
        rdDistGeom.EmbedMultipleConfs(mol, n_frames * n_windows, randomSeed=2024)

    for (mol_a, mol_b), st in zip(pairs, topologies):
        label = mol_a.GetProp("_Name") + " -> " + mol_b.GetProp("_Name")
        print("processing", label)

        qwljs_by_state = []
        for lamb in lamb_schedule:
            cutoff = 1.2
            qwljs = st._get_guest_params(st.ff.q_handle_solv, st.ff.lj_handle_solv, lamb, cutoff=cutoff)
            qwljs_by_state.append(qwljs)

        # generate an artificial trajectory by aligning conformations of mol_a
        mol_a_frames_by_state = get_multiple_romol_confs(mol_a).reshape(n_windows, n_frames, mol_a.GetNumAtoms(), 3)
        mol_b_frames_by_state = get_multiple_romol_confs(mol_b).reshape(n_windows, n_frames, mol_b.GetNumAtoms(), 3)
        mol_c_frames = []
        for mol_a_frames, mol_b_frames in zip(mol_a_frames_by_state, mol_b_frames_by_state):
            for conf_a, conf_b in zip(mol_a_frames, mol_b_frames):
                mol_c_frames.append(st.combine_confs(conf_a, conf_b))

        mol_c_frames = np.array(mol_c_frames)
        mol_c_frames_by_state = mol_c_frames.reshape(n_windows, n_frames, st.get_num_atoms(), 3)

        vols = estimate_avg_volumes_along_schedule(qwljs_by_state, mol_c_frames_by_state)

        plt.plot(lamb_idxs, vols, label=label)

    plt.xlabel("lamb_idx")
    plt.ylabel("volume")
    plt.legend()
    plt.show()
