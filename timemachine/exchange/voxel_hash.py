import numpy as np


def delta_r_np(ri, rj, box):
    diff = ri - rj  # this can be either N,N,3 or B,3
    if box is not None:
        box_diag = np.diag(box)
        diff -= box_diag * np.floor(diff / box_diag + 0.5)
    return diff


class VoxelHash:
    def __init__(self, cell_width, box):
        dims = np.diag(box)
        self.box = box
        self.cell_width = cell_width
        self.cell_counts = np.ceil(dims / cell_width).astype(np.int32)

        # do we actually need identities? or just the counts here?
        self.occupancy = np.zeros(shape=self.cell_counts, dtype=np.int32)

    def count_nonzero(self):
        return np.count_nonzero(self.occupancy)

    def count_total(self):
        return np.sum(self.occupancy)

    # gpu optimization later on... sort by vdw radius to reduce warp divergence, skip vdw=0 for hydrogens
    def delsert(self, xyz, vdw_radius, sign):
        assert vdw_radius < self.box[0][0] / 2
        assert vdw_radius < self.box[1][1] / 2
        assert vdw_radius < self.box[2][2] / 2
        # sign: +1 means insert, -1 means delete
        # activates nearby cells based on occupancy, not the most efficient method
        # inclusive range
        x_min, y_min, z_min = np.floor((xyz - vdw_radius) / self.cell_width).astype(np.int32)
        x_max, y_max, z_max = np.ceil((xyz + vdw_radius) / self.cell_width).astype(np.int32)

        # corner case, to deal with  (xyz - vdw_radii), (xyz+vdw_radii) generating grid size
        # that can be larger than
        x_max = min(x_max, x_min + self.cell_counts[0] - 1)
        y_max = min(y_max, y_min + self.cell_counts[1] - 1)
        z_max = min(z_max, z_min + self.cell_counts[2] - 1)

        # optimize into bounding sphere, not box
        debug = set()
        for xi in range(x_min, x_max + 1):
            for yi in range(y_min, y_max + 1):
                for zi in range(z_min, z_max + 1):
                    # get the wrapped grid idx
                    wrapped_xi = xi % self.cell_counts[0]
                    wrapped_yi = yi % self.cell_counts[1]
                    wrapped_zi = zi % self.cell_counts[2]

                    assert wrapped_xi < self.cell_counts[0]
                    assert wrapped_yi < self.cell_counts[1]
                    assert wrapped_zi < self.cell_counts[2]

                    assert wrapped_xi >= 0
                    assert wrapped_yi >= 0
                    assert wrapped_zi >= 0
                    # print("?", xi, wrapped_xi, self.cell_counts[0])

                    # get the coordinates of the wrapped grid idx
                    gx = wrapped_xi * self.cell_width
                    gy = wrapped_yi * self.cell_width
                    gz = wrapped_zi * self.cell_width
                    grid_point = np.array([gx, gy, gz])

                    # compute the distance: tbd, PBCs?
                    dij = np.linalg.norm(delta_r_np(grid_point, xyz, self.box))
                    if dij < vdw_radius:
                        # print("test", grid_point, xyz)
                        # key = (int(wrapped_xi), int(wrapped_yi), int(wrapped_zi))
                        # if key in debug:
                        # print(key)
                        # assert key not in debug
                        # debug.add(key)
                        self.occupancy[wrapped_xi][wrapped_yi][wrapped_zi] += sign

        return debug


def test_vh():
    # 5Angstrom
    box = np.zeros((3, 3))
    box[0] = 5.5
    box[1] = 4.6
    box[2] = 6.7

    vh = VoxelHash(cell_width=0.2, box=box)
    vh.delsert(np.array([1.8, 2.5, 3.3]), 0.4, sign=1)
    print(np.sum(vh.occupancy))
    vh.delsert(np.array([1.8, 2.5, 3.3]), 0.4, sign=-1)
    print(np.sum(vh.occupancy))


if __name__ == "__main__":
    test_vh()
