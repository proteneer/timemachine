from copy import deepcopy
import numpy as np

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
        """ Mix an arbitrary bonded term. This can be harmonic bond,
        angle, torsions, exlcusions etc. """

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

        lhs_exclusions = [] # dump *all* the exclusions
        lhs_exclusion_params = []

        for (src, dst), param in zip(exclusions_a, exclusion_params_a):
            lhs_exclusions.append((src, dst))
            lhs_exclusion_params.append(param)

        for (src, dst), param in zip(exclusions_b, exclusion_params_b):
            src, dst = src + self.n_a, dst + self.n_a
            lhs_exclusions.append((src, dst))
            lhs_exclusion_params.append(param)

        rhs_exclusions = []
        rhs_exclusion_params = []

        for (src, dst), param in zip(exclusions_b, exclusion_params_b):
            src, dst = src + self.n_a, dst + self.n_a
            src = self.cmap_b_to_a.get(src, src)
            dst = self.cmap_b_to_a.get(dst, dst)
            rhs_exclusions.append((src, dst))
            rhs_exclusion_params.append(param)

        for (src, dst), param in zip(exclusions_a, exclusion_params_a):
            src = self.cmap_a_to_b.get(src, src)
            dst = self.cmap_a_to_b.get(dst, dst)
            rhs_exclusions.append((src, dst))
            rhs_exclusion_params.append(param)

        # add non core exclusions from rhs into lhs
        for (src, dst), rhs_param in zip(rhs_exclusions, rhs_exclusion_params):
            if src not in self.core_atoms or dst not in self.core_atoms:
                lhs_exclusions.append((src, dst))
                lhs_exclusion_params.append(rhs_param)

        # add non core exclusions from lhs into rhs
        for (src, dst), lhs_param in zip(lhs_exclusions, lhs_exclusion_params):
            if src not in self.core_atoms or dst not in self.core_atoms:
                rhs_exclusions.append((src, dst))
                rhs_exclusion_params.append(lhs_param)

        # uniquify
        def uniquify(keys, vals):
            new_map = {}
            for (src, dst), param in zip(keys, vals):
                src, dst = sorted((src, dst))
                new_map[(src, dst)] = param

            sorted_keys = sorted(new_map.keys())

            new_exc = []
            new_params = []
            for k in sorted_keys:
                new_exc.append(k)
                new_params.append(new_map[k])

            return new_exc, new_params

        return uniquify(lhs_exclusions, lhs_exclusion_params), uniquify(rhs_exclusions, rhs_exclusion_params)


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
