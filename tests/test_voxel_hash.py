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
        for x_idx, xi in enumerate(self.gx):
            for y_idx, yi in enumerate(self.gy):
                for z_idx, zi in enumerate(self.gz):
                    gp = np.array([xi, yi, zi])
                    dij = np.linalg.norm(delta_r_np(gp, xyz, self.box))
                    if dij < vdw_radius:
                        self.occupancy[x_idx][y_idx][z_idx] += sign

    def count_nonzero(self):
        return np.count_nonzero(self.occupancy)

    def count_total(self):
        return np.sum(self.occupancy)


def test_reference_voxel_hash():
    box = np.zeros((3, 3))
    box[0][0] = 1.1
    box[1][1] = 0.9
    box[2][2] = 0.4
    rvh = ReferenceVoxelHash(cell_width=0.3, box=box)
    assert rvh.nx == 4
    assert rvh.ny == 3
    assert rvh.nz == 2


def test_insertions():
    np.random.seed(2023)
    box = np.zeros((3, 3))
    box[0][0] = 5.5
    box[1][1] = 4.6
    box[2][2] = 6.7

    for _ in range(50):
        cw = np.random.rand() * np.min(np.diag(box) / 2)  # between 0 and half MIN of box
        if cw < 0.1:
            # ignore very small cell widths since they take forever to run
            continue
        rvh = ReferenceVoxelHash(cell_width=cw, box=box)
        vh = VoxelHash(cell_width=cw, box=box)
        for _ in range(50):
            point = np.random.rand(3) * np.diag(box)
            vdw = np.random.rand() * np.min(np.diag(box) / 2)  # between 0 and half MIN of box
            sign = np.random.randint(0, 2) * 2 - 1

            # sign = 1
            print(point, vdw, sign, cw)
            vh.delsert(point, vdw, sign=sign)
            rvh.delsert(point, vdw, sign=sign)

            assert vh.count_nonzero() == rvh.count_nonzero()
            assert vh.count_total() == rvh.count_total()


def test_insert_delete():
    # 5Angstrom
    box = np.zeros((3, 3))
    box[0][0] = 5.5
    box[1][1] = 4.6
    box[2][2] = 6.7

    vh = VoxelHash(cell_width=0.2, box=box)
    vh.delsert(np.array([1.8, 2.5, 3.3]), 0.4, sign=1)
    vh.delsert(np.array([1.8, 2.5, 3.3]), 0.4, sign=-1)
    assert vh.count_nonzero() == 0

    # test double insert
    vh.delsert(np.array([1.8, 2.5, 3.3]), 0.4, sign=1)
    old_count = vh.count_nonzero()
    vh.delsert(np.array([1.8, 2.5, 3.3]), 0.4, sign=1)
    new_count = vh.count_nonzero()

    assert old_count == new_count
