from copy import deepcopy
import numpy as np


def assert_unique_exclusions(exclusion_idxs):
    sorted_exclusion_idxs = set()
    for src, dst in exclusion_idxs:
        src, dst = sorted((src, dst))
        sorted_exclusion_idxs.add((src, dst))
    assert len(sorted_exclusion_idxs) == len(exclusion_idxs)


class LinearMixer():

    def __init__(self, n_a, map_a_to_b):
        self.n_a = n_a
        self.cmap_a_to_b = {}
        self.core_atoms = set()
        for src, dst in map_a_to_b.items():
            self.cmap_a_to_b[src] = dst + self.n_a

        self.cmap_b_to_a = {}
        for src, dst in self.cmap_a_to_b.items():
            self.core_atoms.add(src)
            self.core_atoms.add(dst)
            self.cmap_b_to_a[dst] = src

    def mix_arbitrary_bonds(self,
        a_bond_idxs,
        a_param_idxs,
        b_bond_idxs,
        b_param_idxs):
        lhs_a_bond_idxs = deepcopy(a_bond_idxs)
        lhs_b_bond_idxs = []
        for atoms in b_bond_idxs:
            lhs_b_bond_idxs.append(atoms+self.n_a)

        lhs_bond_idxs = np.concatenate([lhs_a_bond_idxs, lhs_b_bond_idxs])
        lhs_param_idxs = np.concatenate([a_param_idxs, b_param_idxs])

        # right system:
        # turn b into a
        rhs_a_bond_idxs = []
        for atoms in b_bond_idxs:
            atoms = atoms + self.n_a
            new_atoms = []
            for a_idx in atoms:
                new_atoms.append(self.cmap_b_to_a.get(a_idx, a_idx))
            rhs_a_bond_idxs.append(new_atoms)

        # turn a into b
        rhs_b_bond_idxs = []
        for atoms in a_bond_idxs:
            new_atoms = []
            for a_idx in atoms:
                new_atoms.append(self.cmap_a_to_b.get(a_idx, a_idx))
            rhs_b_bond_idxs.append(new_atoms)

        rhs_bond_idxs = np.concatenate([rhs_a_bond_idxs, rhs_b_bond_idxs])
        rhs_param_idxs = np.concatenate([b_param_idxs, a_param_idxs])
        
        return lhs_bond_idxs, lhs_param_idxs, rhs_bond_idxs, rhs_param_idxs

    def mix_exclusions(self,
        exclusions_a,
        exclusion_params_a,
        exclusions_b,
        exclusion_params_b):

        assert_unique_exclusions(exclusions_a)
        assert_unique_exclusions(exclusions_b)

        lhs_exclusions = {}

        for (src, dst), param in zip(exclusions_a, exclusion_params_a):
            pkey = tuple(sorted((src, dst)))
            assert pkey not in lhs_exclusions
            lhs_exclusions[pkey] = param

        for (src, dst), param in zip(exclusions_b, exclusion_params_b):
            src, dst = src + self.n_a, dst + self.n_a
            pkey = tuple(sorted((src, dst)))
            assert pkey not in lhs_exclusions
            lhs_exclusions[pkey] = param

        rhs_exclusions = {}

        for (src, dst), param in zip(exclusions_b, exclusion_params_b):
            src, dst = src + self.n_a, dst + self.n_a
            src = self.cmap_b_to_a.get(src, src)
            dst = self.cmap_b_to_a.get(dst, dst)
            pkey = tuple(sorted((src, dst)))
            assert pkey not in rhs_exclusions
            rhs_exclusions[pkey] = param

        for (src, dst), param in zip(exclusions_a, exclusion_params_a):
            src = self.cmap_a_to_b.get(src, src)
            dst = self.cmap_a_to_b.get(dst, dst)
            pkey = tuple(sorted((src, dst)))
            assert pkey not in rhs_exclusions
            rhs_exclusions[pkey] = param

        # (ytz): this is commented out because it's buggy
        # we may introduce an endpoint where two particles are excluded but
        # there lacks a bond - which can cause a numerical overflow in how we
        # compute the exclusions

        # merge exclusions
        # add non core exclusions from rhs into lhs
        # for (src, dst), param in rhs_exclusions.items():
        #     if src not in self.core_atoms or dst not in self.core_atoms:
        #         lhs_exclusions[(src, dst)] = param

        # # add non core exclusions from lhs into rhs
        # for(src, dst), param in lhs_exclusions.items():
        #     if src not in self.core_atoms or dst not in self.core_atoms:
        #         rhs_exclusions[src, dst] = param

        lhs_exclusion_idxs = []
        lhs_exclusion_params = []

        for k, v in lhs_exclusions.items():
            lhs_exclusion_idxs.append(k)
            lhs_exclusion_params.append(v)

        rhs_exclusion_idxs = []
        rhs_exclusion_params = []

        for k, v in rhs_exclusions.items():
            rhs_exclusion_idxs.append(k)
            rhs_exclusion_params.append(v)

        return (lhs_exclusion_idxs, lhs_exclusion_params), (rhs_exclusion_idxs, rhs_exclusion_params)


    def mix_lambda_planes(self, n_a, n_b):

        assert n_a == self.n_a
        lambda_plane_idxs = np.concatenate([np.ones(n_a, dtype=np.int32)*2, np.zeros(n_b, dtype=np.int32)])
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


    def mix_lambda_planes_stage_middle(self, n_a, n_b):
        """
        For stage 2 mixing we want:
        C_B to be +0
        R_B to be +1
        C_A to be +2
        R_A to be +3
        """
        assert n_a == self.n_a
        # lambda_plane_idxs = np.concatenate([np.ones(n_a, dtype=np.int32), np.zeros(n_b, dtype=np.int32)])
        lambda_plane_idxs = []

        for a_idx in range(n_a):
            if a_idx in self.cmap_a_to_b:
                # C_A
                lambda_plane_idxs.append(2)
            else:
                # R_A
                lambda_plane_idxs.append(3)

        for b_idx in range(n_b):
            if b_idx + n_a in self.cmap_b_to_a:
                # C_B
                lambda_plane_idxs.append(0)
            else:
                # R_B
                lambda_plane_idxs.append(1)

        lambda_offset_idxs = np.zeros_like(lambda_plane_idxs)

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
