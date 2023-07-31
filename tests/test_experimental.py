import numpy as np
from jax import vmap

from timemachine.datasets import fetch_freesolv
from timemachine.fe.experimental import DecoupleByAtomRank, rank_atoms_by_path_length_to_src


def test_decouple_by_atom_rank():
    """for a few random molecules, make basic assertions about per-atom decoupling schedule"""

    mols = fetch_freesolv()

    num_test_mols = 10
    np.random.seed(1234)
    lambdas = np.linspace(0, 1, 100)
    for mol in mols[:num_test_mols]:
        n_atoms = mol.GetNumAtoms()
        n_src_idxs = np.random.randint(1, min(5, n_atoms))  # arbitrary threshold: between 1 and 5 src_idxs
        src_idxs = list(set(np.random.randint(0, n_atoms, n_src_idxs)))

        atom_idxs = np.arange(n_atoms)
        atom_ranks = rank_atoms_by_path_length_to_src(mol, src_idxs)
        assert atom_ranks.shape == atom_idxs.shape

        decoupler = DecoupleByAtomRank(atom_idxs, atom_ranks)
        atom_lams = vmap(decoupler.atom_lams_from_global_lam)(lambdas)

        assert atom_lams.shape == (len(lambdas), n_atoms), "wrong shape"
        assert (atom_lams >= 0).all() and (atom_lams <= 1).all(), "wrong range"

        diffs = np.diff(atom_lams, axis=0)
        assert (diffs >= 0).all(), "not monotonic"

        started = atom_lams > 0
        finished = atom_lams == 1
        assert not started[0].any(), "should start at 0"
        assert finished[-1].all(), "should end at 1"

        # check somewhere in the middle of the first stage -- could check other stages too
        fractional_t = 0.25 * (1.0 / len(set(atom_ranks)))
        t = int(fractional_t * len(lambdas))
        assert (started[t] == (atom_ranks == 0)).all(), "didn't start with atom_rank 0"
