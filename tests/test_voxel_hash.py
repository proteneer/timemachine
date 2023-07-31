# test voxel hash
import numpy as np

from timemachine.exchange.voxel_hash import VoxelHash, delta_r_np


class ReferenceVoxelHash:
    def __init__(self, cell_width, box):
        dims = np.diag(box)
        self.box = box
        self.nx, self.ny, self.nz = np.ceil(dims / cell_width).astype(np.int32)

        self.gx = np.arange(0, self.nx) * cell_width
        self.gy = np.arange(0, self.ny) * cell_width
        self.gz = np.arange(0, self.nz) * cell_width

        self.occupancy = np.zeros((len(self.gx), len(self.gy), len(self.gz)), dtype=np.int32)

    def delsert(self, xyz, vdw_radius, sign):
        # debug = set()
        for x_idx, xi in enumerate(self.gx):
            for y_idx, yi in enumerate(self.gy):
                for z_idx, zi in enumerate(self.gz):
                    gp = np.array([xi, yi, zi])
                    dij = np.linalg.norm(delta_r_np(gp, xyz, self.box))
                    if dij < vdw_radius:
                        # print("ref", gp, xyz)
                        # debug.add((int(x_idx), int(y_idx), int(z_idx)))
                        self.occupancy[x_idx][y_idx][z_idx] += sign

        # return debug

    def count_nonzero(self):
        return np.count_nonzero(self.occupancy)

    def count_total(self):
        return np.sum(self.occupancy)


# def test_reference_voxel_hash():
#     box = np.zeros((3, 3))
#     box[0][0] = 1.1
#     box[1][1] = 0.9
#     box[2][2] = 0.4
#     rvh = ReferenceVoxelHash(cell_width=0.3, box=box)
#     assert rvh.nx == 4
#     assert rvh.ny == 3
#     assert rvh.nz == 2


def test_insertions():
    np.random.seed(2023)
    box = np.zeros((3, 3))
    box[0][0] = 5.5
    box[1][1] = 4.6
    box[2][2] = 6.7

    rvh = ReferenceVoxelHash(cell_width=0.2, box=box)
    vh = VoxelHash(cell_width=0.2, box=box)

    point = np.array([1.8, 2.5, 3.3])
    vdw = 0.4
    sign = 1

    # debug
    # point = np.array([5.47053907, 3.46565584, 2.45099344])
    # vdw = 0.12227461627795376

    point = np.array([2.61727451, 2.79926738, 0.20419879])
    vdw = 2.264018662235129
    sign = 1

    vh.delsert(point, vdw, sign=sign)
    rvh.delsert(point, vdw, sign=sign)
    # np.testing.assert_equal(frozenset(ref_set), frozenset(test_set))
    # print(test_set.difference(ref_set))
    # print(ref_test)
    # assert 0
    # print(test - ref)

    assert vh.count_nonzero() == rvh.count_nonzero()
    assert vh.count_total() == rvh.count_total()

    # test random insertions, all particles in the box, all positive particles
    for _ in range(10000):
        point = np.random.rand(3) * np.diag(box)
        vdw = np.random.rand() * np.min(np.diag(box) / 2)  # between 0 and half MIN of box
        sign = np.random.randint(0, 2) * 2 - 1

        # sign = 1
        print(point, vdw, sign)
        vh.delsert(point, vdw, sign=sign)
        rvh.delsert(point, vdw, sign=sign)

        assert vh.count_nonzero() == rvh.count_nonzero()
        assert vh.count_total() == rvh.count_total()


# def test_insert_delete():
#     # 5Angstrom
#     box = np.zeros((3, 3))
#     box[0][0] = 5.5
#     box[1][1] = 4.6
#     box[2][2] = 6.7

#     vh = VoxelHash(cell_width=0.2, box=box)
#     vh.delsert(np.array([1.8, 2.5, 3.3]), 0.4, sign=1)
#     vh.delsert(np.array([1.8, 2.5, 3.3]), 0.4, sign=-1)
#     assert vh.count_nonzero() == 0

#     # test double insert
#     vh.delsert(np.array([1.8, 2.5, 3.3]), 0.4, sign=1)
#     old_count = vh.count_nonzero()
#     vh.delsert(np.array([1.8, 2.5, 3.3]), 0.4, sign=1)
#     new_count = vh.count_nonzero()

#     assert old_count == new_count
