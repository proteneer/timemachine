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


def test_propose_insertion():
    np.random.seed(2023)
    box = np.zeros((3, 3))
    box[0][0] = 5.5
    box[1][1] = 4.6
    box[2][2] = 6.7

    vh = VoxelHash(cell_width=0.2, box=box)
    vh.delsert(np.array([1.8, 2.5, 3.3]), 0.4, sign=1)

    for _ in range(1000):
        coords = vh.propose_insertion_site()
        print(coords)
        grid_idx = np.floor(coords / vh.cell_width).astype(np.int32)
        xi, yi, zi = grid_idx
        assert vh.occupancy[xi][yi][zi] == 0

        prob = vh.compute_insertion_probability(coords)
        assert prob == 1 / (vh.count_zero() * np.prod(vh.get_cell_widths(grid_idx)))
        assert coords[0] < box[0][0]
        assert coords[1] < box[1][1]
        assert coords[2] < box[2][2]


import tqdm
from openmm import app

from timemachine.md.builders import strip_units


def test_cell_widths():
    box = np.zeros((3, 3))
    box[0][0] = 1.1
    box[1][1] = 1.25
    box[2][2] = 0.4

    vh = VoxelHash(cell_width=0.3, box=box)

    np.testing.assert_array_almost_equal(vh.get_cell_widths(np.array([2, 4, 0])), np.array([0.3, 0.05, 0.3]))
    np.testing.assert_array_almost_equal(vh.get_cell_widths(np.array([2, 3, 0])), np.array([0.3, 0.3, 0.3]))
    np.testing.assert_array_almost_equal(vh.get_cell_widths(np.array([3, 2, 0])), np.array([0.2, 0.3, 0.3]))
    np.testing.assert_array_almost_equal(vh.get_cell_widths(np.array([3, 4, 0])), np.array([0.2, 0.05, 0.3]))
    np.testing.assert_array_almost_equal(vh.get_cell_widths(np.array([3, 4, 1])), np.array([0.2, 0.05, 0.1]))
    np.testing.assert_array_almost_equal(vh.get_cell_widths(np.array([0, 0, 0])), np.array([0.3, 0.3, 0.3]))
    np.testing.assert_array_almost_equal(vh.get_cell_widths(np.array([1, 1, 1])), np.array([0.3, 0.3, 0.1]))


def test_water_box():
    host_pdb = app.PDBFile("timemachine/datasets/water_exchange/bb_0_waters.pdb")
    host_coords = strip_units(host_pdb.positions)
    box_lengths = np.amax(host_coords, axis=0) - np.amin(host_coords, axis=0)
    box_lengths = box_lengths + 0.2
    box = np.eye(3, dtype=np.float64) * box_lengths

    cw = 0.2
    vh = VoxelHash(cell_width=cw, box=box)

    # print(box)
    # insert only water vdw radiis
    for coords in tqdm.tqdm(host_coords[::3]):
        vdw = 0.4
        # print("inserting", coords)
        vh.delsert(coords, vdw, sign=1)

    print(vh.count_total())
    print(vh.count_zero())
    print(vh.count_nonzero())
