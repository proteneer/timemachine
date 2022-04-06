import numpy as np
import pytest
from common import hilbert_sort

from timemachine.fe.utils import get_romol_conf
from timemachine.lib import custom_ops
from timemachine.md.builders import build_water_system
from timemachine.testsystems.relative import hif2a_ligand_pair

pytestmark = [pytest.mark.memcheck]


def test_block_bounds():

    np.random.seed(2020)

    for N in [128, 156, 298]:
        block_size = 32

        D = 3

        coords = np.random.randn(N, D)
        box_diag = np.random.rand(3) + 1
        box = np.diag(box_diag)
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

        nblist = custom_ops.Neighborlist_f64(N)
        test_ctrs, test_exts = nblist.compute_block_bounds(coords, box, block_size)

        np.testing.assert_almost_equal(ref_ctrs, test_ctrs)
        np.testing.assert_almost_equal(ref_exts, test_exts)


def get_water_coords(D, sort=False):
    x = np.load("tests/data/water.npy").astype(np.float32).astype(np.float64)
    x = x[:, :D]

    return x


def test_neighborlist():

    for size in [35, 64, 129, 1025, 1259, 2029]:

        for nblist in (custom_ops.Neighborlist_f32(size), custom_ops.Neighborlist_f64(size)):
            water_coords = get_water_coords(3, sort=False)

            for _ in range(2):

                print("testing size:", size)

                atom_idxs = np.random.choice(np.arange(size), size, replace=False)
                coords = water_coords[atom_idxs]
                padding = 0.1
                diag = np.amax(coords, axis=0) - np.amin(coords, axis=0) + padding
                box = np.diag(diag)

                N = coords.shape[0]
                np.random.seed(1234)
                D = 3

                sort = True
                if sort:
                    perm = hilbert_sort(coords + np.argmin(coords), D)
                    coords = coords[perm]

                num_blocks_of_32 = (N + 32 - 1) // 32
                col_coords = np.expand_dims(coords, axis=0)

                cutoff = 1.0

                test_ixn_list = nblist.get_nblist(coords, box, cutoff)

                # for each tile, print list of interacting atoms

                # compute the sparsity of the tile
                ref_ixn_list = []

                box_diag = np.diag(box)
                for rbidx in range(num_blocks_of_32):
                    row_start = rbidx * 32
                    row_end = min((rbidx + 1) * 32, N)
                    row_coords = coords[row_start:row_end]
                    row_coords = np.expand_dims(row_coords, axis=1)

                    deltas = row_coords - col_coords
                    deltas -= box_diag * np.floor(deltas / box_diag + 0.5)

                    # block size x N, tbd make periodic
                    dij = np.linalg.norm(deltas, axis=-1)
                    dij[:, :row_start] = cutoff  # slight hack to discard duplicates
                    idxs = np.argwhere(np.any(dij < cutoff, axis=0))
                    ref_ixn_list.append(idxs.reshape(-1).tolist())

                assert len(ref_ixn_list) == len(test_ixn_list)

                for bidx, (a, b) in enumerate(zip(ref_ixn_list, test_ixn_list)):
                    if sorted(a) != sorted(b):
                        print("TESTING bidx", bidx)
                        print(sorted(a))
                        print(sorted(b))
                    np.testing.assert_equal(sorted(a), sorted(b))


def test_neighborlist_ligand_host_invalid_parameters():
    cols = 10
    rows = 5
    box = np.diag(np.ones(3))
    cutoff = 1.0

    # Constructing NBlist with rows > cols is invalid
    with pytest.raises(RuntimeError) as e:
        custom_ops.Neighborlist_f32(rows, cols)
    assert "NR is greater than NC" in str(e.value)

    # Verify that the sizes of the rows and columns match how the NBlist was constructed
    for nblist in (
        custom_ops.Neighborlist_f32(cols, rows),
        custom_ops.Neighborlist_f64(cols, rows),
    ):
        with pytest.raises(RuntimeError) as e:
            # Flip the order of rows and columns
            nblist.get_nblist_host_ligand(
                np.random.rand(rows, 3),
                np.random.rand(cols, 3),
                box,
                cutoff,
            )
            assert "NC != NC_" in str(e.value)


def test_neighborlist_ligand_host():
    ligand = hif2a_ligand_pair.mol_a
    ligand_coords = get_romol_conf(ligand)

    system, host_coords, box, top = build_water_system(4.0)
    num_host_atoms = host_coords.shape[0]
    host_coords = np.array(host_coords)

    coords = np.concatenate([host_coords, ligand_coords])

    N = coords.shape[0]
    D = 3
    cutoff = 1.0
    block_size = 32
    padding = 0.1

    np.random.seed(1234)
    diag = np.amax(coords, axis=0) - np.amin(coords, axis=0) + padding
    box = np.diag(diag)

    # Can only sort the host coords, but not the row/ligand
    sort = True
    if sort:
        perm = hilbert_sort(coords[:num_host_atoms] + np.argmin(coords[:num_host_atoms]), D)
        coords[:num_host_atoms] = coords[:num_host_atoms][perm]

    col_coords = np.expand_dims(coords[:num_host_atoms], axis=0)
    # Compute the reference interactions of the ligand
    ref_ixn_list = []
    num_ligand_atoms = coords[num_host_atoms:].shape[0]
    num_blocks_of_32 = (num_ligand_atoms + block_size - 1) // block_size
    box_diag = np.diag(box)
    for rbidx in range(num_blocks_of_32):
        row_start = num_host_atoms + (rbidx * block_size)
        row_end = min(num_host_atoms + ((rbidx + 1) * block_size), N)
        row_coords = coords[row_start:row_end]
        row_coords = np.expand_dims(row_coords, axis=1)
        deltas = row_coords - col_coords
        deltas -= box_diag * np.floor(deltas / box_diag + 0.5)

        dij = np.linalg.norm(deltas, axis=-1)
        # Since the row and columns are unique, don't need to handle duplicates
        idxs = np.argwhere(np.any(dij < cutoff, axis=0))
        ref_ixn_list.append(idxs.reshape(-1).tolist())

    for nblist in (
        custom_ops.Neighborlist_f32(num_host_atoms, num_ligand_atoms),
        custom_ops.Neighborlist_f64(num_host_atoms, num_ligand_atoms),
    ):
        for _ in range(2):

            test_ixn_list = nblist.get_nblist_host_ligand(coords[:num_host_atoms], coords[num_host_atoms:], box, cutoff)
            # compute the sparsity of the tile
            assert len(ref_ixn_list) == len(test_ixn_list), "Number of blocks with interactions don't agree"

            for bidx, (a, b) in enumerate(zip(ref_ixn_list, test_ixn_list)):
                if sorted(a) != sorted(b):
                    print("TESTING bidx", bidx)
                    print(sorted(a))
                    print(sorted(b))
                np.testing.assert_equal(sorted(a), sorted(b))
