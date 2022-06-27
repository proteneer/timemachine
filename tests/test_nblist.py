from typing import List

import numpy as np
import pytest
from common import hilbert_sort
from numpy.typing import NDArray

from timemachine.fe.utils import get_romol_conf
from timemachine.lib import custom_ops
from timemachine.md.builders import build_water_system
from timemachine.testsystems.relative import hif2a_ligand_pair

pytestmark = [pytest.mark.memcheck]


def test_block_bounds():
    np.random.seed(2020)
    sizes = [128, 156, 298]
    max_size = max(sizes)
    nblist = custom_ops.Neighborlist_f64(max_size)
    for N in sizes:
        nblist.resize(N)
        block_size = 32

        D = 3

        coords = np.random.randn(N, D)
        box_diag = np.random.rand(3) + 1
        box = np.eye(3) * box_diag
        num_blocks = (N + block_size - 1) // block_size

        ref_ctrs = []
        ref_exts = []

        for bidx in range(num_blocks):
            start_idx = bidx * block_size
            end_idx = min((bidx + 1) * block_size, N)
            block_coords = coords[start_idx:end_idx]
            min_coords = block_coords[0]
            max_coords = block_coords[0]
            for new_coords in block_coords[1:]:
                center = 0.5 * (max_coords + min_coords)
                new_coords -= box_diag * np.floor((new_coords - center) / box_diag + 0.5)
                min_coords = np.minimum(min_coords, new_coords)
                max_coords = np.maximum(max_coords, new_coords)

            ref_ctrs.append((max_coords + min_coords) / 2)
            ref_exts.append((max_coords - min_coords) / 2)

        ref_ctrs = np.array(ref_ctrs)
        ref_exts = np.array(ref_exts)

        test_ctrs, test_exts = nblist.compute_block_bounds(coords, box, block_size)

        np.testing.assert_almost_equal(ref_ctrs, test_ctrs)
        np.testing.assert_almost_equal(ref_exts, test_exts)


def get_water_coords(D, sort=False):
    x = np.load("tests/data/water.npy").astype(np.float32).astype(np.float64)
    x = x[:, :D]

    return x


def image_coords(x: NDArray, box_diag: NDArray) -> NDArray:
    return x - box_diag * np.floor(x / box_diag + 0.5)


def build_reference_ixn_list(coords: NDArray, box: NDArray, cutoff: float) -> List[List[float]]:
    # compute the sparsity of the tile
    ref_ixn_list = []
    N = coords.shape[0]

    block_size = 32

    num_blocks = (N + block_size - 1) // block_size
    col_coords = np.expand_dims(coords, axis=0)

    box_diag = np.diagonal(box)
    for rbidx in range(num_blocks):
        row_start = rbidx * block_size
        row_end = min((rbidx + 1) * block_size, N)
        row_coords = coords[row_start:row_end]
        row_coords = np.expand_dims(row_coords, axis=1)

        deltas = image_coords(row_coords - col_coords, box_diag)

        # block size x N, tbd make periodic
        dij = np.linalg.norm(deltas, axis=-1)
        dij[:, :row_start] = cutoff  # slight hack to discard duplicates
        idxs = np.argwhere(np.any(dij < cutoff, axis=0))
        ref_ixn_list.append(idxs.reshape(-1).tolist())
    return ref_ixn_list


def build_reference_ixn_list_with_subset(
    coords: NDArray, box: NDArray, cutoff: float, row_idxs: NDArray
) -> List[List[int]]:
    N = coords.shape[0]
    block_size = 32
    identity_idxs = np.arange(N)
    col_idxs = np.delete(identity_idxs, row_idxs)
    box_diag = np.diagonal(box)

    # Verify that the row_idxs and col_idxs are unique
    np.testing.assert_array_equal(
        np.sort(np.concatenate([col_idxs, row_idxs])),
        identity_idxs,
    )

    col_coords = coords[col_idxs]
    col_coords = np.expand_dims(col_coords, axis=0)
    # Compute the reference interactions of the ligand
    ref_ixn_list = []
    all_row_coords = coords[row_idxs]
    row_length = all_row_coords.shape[0]
    num_blocks = (row_length + block_size - 1) // block_size

    for rbidx in range(num_blocks):
        row_start = rbidx * block_size
        row_end = min((rbidx + 1) * block_size, N)
        row_coords = all_row_coords[row_start:row_end]
        row_coords = np.expand_dims(row_coords, axis=1)
        deltas = image_coords(row_coords - col_coords, box_diag)

        dij = np.linalg.norm(deltas, axis=-1)
        # Since the row and columns are unique, don't need to handle duplicates
        idxs = np.argwhere(np.any(dij < cutoff, axis=0))
        # Get back the column indices that are ixns
        idxs = col_idxs[idxs.reshape(-1)]
        ref_ixn_list.append(idxs.reshape(-1).tolist())
    return ref_ixn_list


def assert_ixn_lists_are_equal(ref_ixn, test_ixn):
    for bidx, (a, b) in enumerate(zip(ref_ixn, test_ixn)):
        if sorted(a) != sorted(b):
            print("TESTING bidx", bidx)
            print(sorted(a))
            print(sorted(b))
        np.testing.assert_equal(sorted(a), sorted(b))


def test_nblist_row_indices_are_order_independent():
    D = 3
    cutoff = 1.0
    padding = 0.1
    sizes = [35, 64, 129, 1025, 1259, 2029]
    max_size = max(sizes)
    water_coords = get_water_coords(D, sort=False)
    nblists = [custom_ops.Neighborlist_f32(max_size), custom_ops.Neighborlist_f64(max_size)]
    for size in sizes:
        print("testing size:", size)

        np.random.seed(1234)
        water_idxs = np.random.choice(np.arange(water_coords.shape[0]), size, replace=False)
        coords = water_coords[water_idxs]
        diag = np.amax(coords, axis=0) - np.amin(coords, axis=0) + padding
        box = np.diag(diag)

        atom_idxs = np.random.choice(np.arange(coords.shape[0]), size // 2, replace=False)
        atom_idxs = atom_idxs.astype(np.uint32)

        reference_ixns = build_reference_ixn_list_with_subset(coords, box, cutoff, atom_idxs)
        # Shuffle idxs, should still have the same set of interactions
        shuffled_idxs = atom_idxs.copy()
        np.random.shuffle(shuffled_idxs)

        assert not np.all(shuffled_idxs == atom_idxs)

        shuffled_ixns = build_reference_ixn_list_with_subset(coords, box, cutoff, shuffled_idxs)

        # Verify that the ixns are the same, different ordering so each block will be different
        reference_ixns_set = set(np.concatenate(reference_ixns).reshape(-1))
        shuffled_ixns_set = set(np.concatenate(shuffled_ixns).reshape(-1))

        np.testing.assert_array_equal(reference_ixns_set, shuffled_ixns_set)

        # Verify that the C++ agrees
        for nblist in nblists:
            nblist.resize(size)
            nblist.set_row_idxs(atom_idxs)
            test_ixn_list = nblist.get_nblist(coords, box, cutoff)
            test_ixns_set = set(np.concatenate(test_ixn_list).reshape(-1))
            assert reference_ixns_set == test_ixns_set
            assert_ixn_lists_are_equal(reference_ixns, test_ixn_list)

            nblist.set_row_idxs(shuffled_idxs)
            test_shuffle_ixn_list = nblist.get_nblist(coords, box, cutoff)
            test_shuffle_ixns_set = set(np.concatenate(test_shuffle_ixn_list).reshape(-1))
            assert shuffled_ixns_set == test_shuffle_ixns_set
            assert_ixn_lists_are_equal(shuffled_ixns, test_shuffle_ixn_list)


def test_neighborlist():
    water_coords = get_water_coords(3, sort=False)
    sizes = [35, 64, 129, 1025, 1259, 2029]
    max_size = max(sizes)
    nblists = [custom_ops.Neighborlist_f32(max_size), custom_ops.Neighborlist_f64(max_size)]
    for size in sizes:
        print("testing size:", size)

        np.random.seed(1234)
        atom_idxs = np.random.choice(np.arange(size), size, replace=False)
        coords = water_coords[atom_idxs]
        padding = 0.1
        diag = np.amax(coords, axis=0) - np.amin(coords, axis=0) + padding
        box = np.eye(3) * diag

        D = 3
        cutoff = 1.0

        sort = True
        if sort:
            perm = hilbert_sort(coords + np.argmin(coords), D)
            coords = coords[perm]

        ref_ixn_list = build_reference_ixn_list(coords, box, cutoff)
        for nblist in nblists:
            # Resize the nblist accordingly
            nblist.resize(size)
            # Run twice to ensure deterministic results
            for _ in range(2):
                test_ixn_list = nblist.get_nblist(coords, box, cutoff)

                assert len(ref_ixn_list) == len(test_ixn_list)

                assert_ixn_lists_are_equal(ref_ixn_list, test_ixn_list)


def test_neighborlist_resize():
    N = 3

    # Verify that the sizes of the rows and columns match how the NBlist was constructed
    for nblist in (
        custom_ops.Neighborlist_f32(N),
        custom_ops.Neighborlist_f64(N),
    ):
        with pytest.raises(RuntimeError) as e:
            nblist.resize(0)
        assert "size is must be at least 1" == str(e.value)

        with pytest.raises(RuntimeError) as e:
            nblist.resize(N + 1)
        assert "size is greater than max size" == str(e.value)


def test_neighborlist_invalid_row_idxs():
    N = 3

    # Verify that the sizes of the rows and columns match how the NBlist was constructed
    for nblist in (
        custom_ops.Neighborlist_f32(N),
        custom_ops.Neighborlist_f64(N),
    ):
        with pytest.raises(RuntimeError) as e:
            nblist.set_row_idxs(np.zeros(0, dtype=np.uint32))
        assert "idxs can't be empty" == str(e.value)

        with pytest.raises(RuntimeError) as e:
            nblist.set_row_idxs(np.zeros(2, dtype=np.uint32))
        assert "atom indices must be unique" == str(e.value)

        with pytest.raises(RuntimeError) as e:
            nblist.set_row_idxs(np.arange(N * 5, dtype=np.uint32))
        assert "number of idxs must be less than N" == str(e.value)

        with pytest.raises(RuntimeError) as e:
            nblist.set_row_idxs(np.arange(N - 1, dtype=np.uint32) * N * 5)
        assert "indices values must be less than N" == str(e.value)


def test_neighborlist_on_subset_of_system():
    ligand = hif2a_ligand_pair.mol_a
    ligand_coords = get_romol_conf(ligand)

    system, host_coords, box, top = build_water_system(4.0)
    num_host_atoms = host_coords.shape[0]
    host_coords = np.array(host_coords)

    coords = np.concatenate([host_coords, ligand_coords])
    N = coords.shape[0]

    D = 3
    cutoff = 1.0
    padding = 0.1

    np.random.seed(1234)
    diag = np.amax(coords, axis=0) - np.amin(coords, axis=0) + padding
    box = np.eye(3) * diag

    atom_idxs = np.arange(num_host_atoms, N, dtype=np.uint32)
    sort = True
    if sort:
        perm = hilbert_sort(coords + np.argmin(coords), D)
        coords = coords[perm]
        # Get the new idxs of the ligand atoms
        atom_idxs = np.isin(perm, atom_idxs).nonzero()[0]
    atom_idxs = atom_idxs.astype(np.uint32)

    reference_subset_ixns = build_reference_ixn_list_with_subset(coords, box, cutoff, atom_idxs)
    reference_complete_ixns = build_reference_ixn_list(coords, box, cutoff)

    for nblist in (
        custom_ops.Neighborlist_f32(N),
        custom_ops.Neighborlist_f64(N),
    ):
        nblist.set_row_idxs(atom_idxs)
        for _ in range(2):

            test_ixn_list = nblist.get_nblist(coords, box, cutoff)
            # compute the sparsity of the tile
            assert len(reference_subset_ixns) == len(test_ixn_list), "Number of blocks with interactions don't agree"

            assert_ixn_lists_are_equal(reference_subset_ixns, test_ixn_list)
        # Verify that you can reset the indices and go back to the regular neighborlist
        nblist.reset_row_idxs()
        test_ixn_list = nblist.get_nblist(coords, box, cutoff)
        # compute the sparsity of the tile
        assert len(reference_complete_ixns) == len(test_ixn_list), "Number of blocks with interactions don't agree"

        assert_ixn_lists_are_equal(reference_complete_ixns, test_ixn_list)
