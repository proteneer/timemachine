from copy import deepcopy
import numpy as np


class LinearMixer():

    def __init__(self, n_a, map_a_to_b):
        self.n_a = n_a
        self.cmap_a_to_b = {}
        for src, dst in map_a_to_b.items():
            self.cmap_a_to_b[src] = dst + self.n_a

        self.cmap_b_to_a = {}
        for src, dst in self.cmap_a_to_b.items():
            self.cmap_b_to_a[dst] = src

    def mix_bonds(self,
        a_bond_idxs,
        a_param_idxs,
        b_bond_idxs,
        b_param_idxs):

        # left system:
        # increment indices of b system by n_a
        # param_idxs stays the same

        lhs_a_bond_idxs = deepcopy(a_bond_idxs)
        lhs_b_bond_idxs = []
        for src, dst in b_bond_idxs:
            lhs_b_bond_idxs.append((src+self.n_a, dst+self.n_a))

        lhs_bond_idxs = np.concatenate([lhs_a_bond_idxs, lhs_b_bond_idxs])
        lhs_param_idxs = np.concatenate([a_param_idxs, b_param_idxs])

        # right system:
        # turn b into a
        rhs_a_bond_idxs = []
        for b_idx, (src, dst) in enumerate(b_bond_idxs):
            src, dst = src + self.n_a, dst + self.n_a
            src = self.cmap_b_to_a.get(src, src)
            dst = self.cmap_b_to_a.get(dst, dst)
            rhs_a_bond_idxs.append((src, dst))

        # turn a into b
        rhs_b_bond_idxs = []
        for src, dst in a_bond_idxs:
            src = self.cmap_a_to_b.get(src, src)
            dst = self.cmap_a_to_b.get(dst, dst)
            rhs_b_bond_idxs.append((src, dst))

        rhs_bond_idxs = np.concatenate([rhs_a_bond_idxs, rhs_b_bond_idxs])
        rhs_param_idxs = np.concatenate([b_param_idxs, a_param_idxs])
        
        return lhs_bond_idxs, lhs_param_idxs, rhs_bond_idxs, rhs_param_idxs


    def mix_lambda_planes(self, n_a, n_b):

        assert n_a == self.n_a
        lambda_plane_idxs = np.concatenate([np.ones(n_a, dtype=np.int32), np.zeros(n_b, dtype=np.int32)])
        lambda_offset_idxs = []

        for a_idx in range(n_a):
            if a_idx in self.cmap_a_to_b:
                # core atoms stay fixed
                lambda_offset_idxs.append(0)
            else:
                # non core atoms are moved up
                lambda_offset_idxs.append(1)

        for b_idx in range(n_b):
            if b_idx + n_a in self.cmap_b_to_a:
                lambda_offset_idxs.append(0)
            else:
                lambda_offset_idxs.append(1)

        return lambda_plane_idxs, lambda_offset_idxs

    def mix_nonbonded_parameters(self, params_a, params_b):
        """
        Mix parameter indices.
        """
        lhs_params = np.concatenate([params_a, params_b])
        rhs_params = []

        # we only need to modify the core params
        rhs_params_a = deepcopy(params_a)
        rhs_params_b = deepcopy(params_b)
        for src, dst in self.cmap_a_to_b.items():
            dst = dst - self.n_a
            rhs_params_a[src] = params_b[dst]
            rhs_params_b[dst] = params_a[src]

        return lhs_params, np.concatenate([rhs_params_a, rhs_params_b])
