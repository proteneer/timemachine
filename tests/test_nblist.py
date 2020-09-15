import numpy as np
from timemachine.lib import custom_ops


def test_block_bounds():

    for N in [128, 156, 298]:
        for block_size in [23, 32, 35]:

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
                block_coords -= box_diag*np.floor(block_coords//box_diag)

                c_max = np.amax(block_coords, axis=0)       
                c_min = np.amin(block_coords, axis=0)

                ref_ctrs.append((c_max + c_min)/2)
                ref_exts.append((c_max - c_min)/2)

            ref_ctrs = np.array(ref_ctrs)
            ref_exts = np.array(ref_exts)

            nblist = custom_ops.Neighborlist(N, D)
            test_ctrs, test_exts = nblist.compute_block_bounds(coords, box, block_size)

            np.testing.assert_almost_equal(ref_ctrs, test_ctrs)
            np.testing.assert_almost_equal(ref_exts, test_exts)
