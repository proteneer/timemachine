import numpy as np
from timemachine.lib import custom_ops

from hilbertcurve.hilbertcurve import HilbertCurve

import time

def test_block_bounds():

    np.random.seed(2020)

    for N in [128, 156, 298]:
        block_size = 32

        D = 3

        coords = np.random.randn(N, D)
        box_diag = (np.random.rand(3) + 1)
        box = np.eye(3) * box_diag
        num_blocks = (N + block_size - 1)//block_size
        
        ref_ctrs = []
        ref_exts = []

        for bidx in range(num_blocks):
            start_idx = bidx*block_size
            end_idx = min((bidx+1)*block_size, N)
            block_coords = coords[start_idx:end_idx]
            min_coords = block_coords[0]
            max_coords = block_coords[0]
            for new_coords in block_coords[1:]:
                center = 0.5 * (max_coords + min_coords)
                new_coords -= box_diag * np.floor((new_coords - center) / box_diag + 0.5)
                min_coords = np.minimum(min_coords, new_coords)
                max_coords = np.maximum(max_coords, new_coords)

            ref_ctrs.append((max_coords + min_coords)/2)
            ref_exts.append((max_coords - min_coords)/2)

        ref_ctrs = np.array(ref_ctrs)
        ref_exts = np.array(ref_exts)

        nblist = custom_ops.Neighborlist_f64(N)
        test_ctrs, test_exts = nblist.compute_block_bounds(coords, box, block_size)

        np.testing.assert_almost_equal(ref_ctrs, test_ctrs)
        np.testing.assert_almost_equal(ref_exts, test_exts)


def hilbert_sort(conf, D):
    hc = HilbertCurve(64, D)
    int_confs = (conf*1000).astype(np.int64)
    dists = []
    for xyz in int_confs.tolist():
        dist = hc.distance_from_coordinates(xyz)
        dists.append(dist)
    perm = np.argsort(dists)
    return perm


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
                box = np.eye(3)
                np.fill_diagonal(box, diag)

                N = coords.shape[0]
                np.random.seed(1234)
                D = 3

                sort = True
                if sort:
                    perm = hilbert_sort(coords+np.argmin(coords), D)
                    coords = coords[perm]

                num_blocks_of_32 = (N + 32 - 1)//32
                col_coords = np.expand_dims(coords, axis=0)

                cutoff = 1.0

                test_ixn_list = nblist.get_nblist(coords, box, cutoff)

                # for each tile, print list of interacting atoms

                # compute the sparsity of the tile
                ref_ixn_list = []

                box_diag = np.diag(box)
                for rbidx in range(num_blocks_of_32):
                    row_start = rbidx*32
                    row_end = min((rbidx+1)*32, N)
                    row_coords = coords[row_start:row_end]
                    row_coords = np.expand_dims(row_coords, axis=1)

                    deltas = row_coords - col_coords
                    deltas -= box_diag*np.floor(deltas/box_diag+0.5)

                    # block size x N, tbd make periodic
                    dij = np.linalg.norm(deltas, axis=-1)
                    dij[:, :row_start] = cutoff # slight hack to discard duplicates
                    idxs = np.argwhere(np.any(dij < cutoff, axis=0))
                    ref_ixn_list.append(idxs.reshape(-1).tolist())


                assert len(ref_ixn_list) == len(test_ixn_list)

                for bidx, (a, b) in enumerate(zip(ref_ixn_list, test_ixn_list)):
                    if sorted(a) != sorted(b):
                        print("TESTING bidx", bidx)
                        print(sorted(a))
                        print(sorted(b))
                    # print(a, b)
                    np.testing.assert_equal(sorted(a), sorted(b))

                # np.testing.assert_equal(ref_ixn_list, test_ixn_list)

                # output is a group of 32
