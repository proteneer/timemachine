import numpy as np

from timemachine.lib import ops
from fe import linear_mixer
from ff import system

class System():

    def __init__(self, nrg_fns, params, param_groups, masses):

        self.nrg_fns = nrg_fns
        self.params = params
        self.param_groups = param_groups
        self.masses = masses

    def mix(self, other, self_to_other_map_nonbonded, self_to_other_map_bonded):
        """
        Alchemically mix two ligand systems together.

        Parameters
        ----------
        other: system.System
            The other ligand we wish to morph into

        self_to_other_map: dict
            Mapping of atoms in self to atoms in other

        """
        a_masses = self.masses
        b_masses = other.masses
        a_to_b_map_nonbonded = self_to_other_map_nonbonded
        a_to_b_map_bonded = self_to_other_map_bonded

        combined_masses = np.concatenate([a_masses, b_masses])

        a_nrg_fns = self.nrg_fns
        b_nrg_fns = other.nrg_fns

        np.testing.assert_equal(self.params, other.params)
        np.testing.assert_equal(self.param_groups, other.param_groups)

        a_bond_idxs, a_bond_param_idxs = a_nrg_fns['HarmonicBond']
        b_bond_idxs, b_bond_param_idxs = b_nrg_fns['HarmonicBond']

        n_a = len(a_masses)
        n_b = len(b_masses)

        lm = linear_mixer.LinearMixer(n_a, a_to_b_map_bonded)

        lhs_bond_idxs, lhs_bond_param_idxs, rhs_bond_idxs, rhs_bond_param_idxs = lm.mix_arbitrary_bonds(
            a_bond_idxs, a_bond_param_idxs,
            b_bond_idxs, b_bond_param_idxs
        )

        lhs_nrg_fns = {}
        rhs_nrg_fns = {}

        # bonded mixing
        # for bonded mixing to work well we need make sure that that the bond types match *exactly*

        lhs_nrg_fns['HarmonicBond'] = (lhs_bond_idxs, lhs_bond_param_idxs)
        rhs_nrg_fns['HarmonicBond'] = (rhs_bond_idxs, rhs_bond_param_idxs)

        # print(lhs_bond_idxs)
        # print(rhs_bond_idxs)

        # set_a = set()
        # for pair in lhs_bond_idxs:
        #     set_a.add(tuple(pair))


        # set_b = set()
        # for pair in rhs_bond_idxs:
        #     set_b.add(tuple(pair))

        # print(set_a == set_b)

        # assert 0

        a_angle_idxs, a_angle_param_idxs = a_nrg_fns['HarmonicAngle']
        b_angle_idxs, b_angle_param_idxs = b_nrg_fns['HarmonicAngle']

        lhs_angle_idxs, lhs_angle_param_idxs, rhs_angle_idxs, rhs_angle_param_idxs = lm.mix_arbitrary_bonds(
            a_angle_idxs, a_angle_param_idxs,
            b_angle_idxs, b_angle_param_idxs
        )

        lhs_nrg_fns['HarmonicAngle'] = (lhs_angle_idxs, lhs_angle_param_idxs)
        rhs_nrg_fns['HarmonicAngle'] = (rhs_angle_idxs, rhs_angle_param_idxs)

        a_torsion_idxs, a_torsion_param_idxs = a_nrg_fns['PeriodicTorsion']
        b_torsion_idxs, b_torsion_param_idxs = b_nrg_fns['PeriodicTorsion']

        lhs_torsion_idxs, lhs_torsion_param_idxs, rhs_torsion_idxs, rhs_torsion_param_idxs = lm.mix_arbitrary_bonds(
            a_torsion_idxs, a_torsion_param_idxs,
            b_torsion_idxs, b_torsion_param_idxs
        )

        lhs_nrg_fns['PeriodicTorsion'] = (lhs_torsion_idxs, lhs_torsion_param_idxs)
        rhs_nrg_fns['PeriodicTorsion'] = (rhs_torsion_idxs, rhs_torsion_param_idxs)


        # nonbonded mixing
        lm = linear_mixer.LinearMixer(n_a, a_to_b_map_nonbonded)

        lambda_plane_idxs, lambda_offset_idxs = lm.mix_lambda_planes(n_a, n_b)

        a_es_param_idxs, a_lj_param_idxs, a_exc_idxs, a_es_exc_param_idxs, a_lj_exc_param_idxs, a_cutoff = a_nrg_fns['Nonbonded']
        b_es_param_idxs, b_lj_param_idxs, b_exc_idxs, b_es_exc_param_idxs, b_lj_exc_param_idxs, b_cutoff = b_nrg_fns['Nonbonded']

        assert a_cutoff == b_cutoff

        lhs_es_param_idxs, rhs_es_param_idxs = lm.mix_nonbonded_parameters(a_es_param_idxs, b_es_param_idxs)
        lhs_lj_param_idxs, rhs_lj_param_idxs = lm.mix_nonbonded_parameters(a_lj_param_idxs, b_lj_param_idxs)


        print(lhs_lj_param_idxs[14])
        print(lhs_lj_param_idxs[51])

        print(rhs_lj_param_idxs[14])
        print(rhs_lj_param_idxs[51])

        # same parameters....

        # assert 0

        (_,            lhs_lj_exc_param_idxs), (           _, rhs_lj_exc_param_idxs) = lm.mix_exclusions(a_exc_idxs, a_lj_exc_param_idxs, b_exc_idxs, b_lj_exc_param_idxs)
        (lhs_exc_idxs, lhs_es_exc_param_idxs), (rhs_exc_idxs, rhs_es_exc_param_idxs) = lm.mix_exclusions(a_exc_idxs, a_es_exc_param_idxs, b_exc_idxs, b_es_exc_param_idxs)

        lhs_exc_idxs = np.array(lhs_exc_idxs, dtype=np.int32)
        rhs_exc_idxs = np.array(rhs_exc_idxs, dtype=np.int32)

        # print(lhs_exc_idxs)
        # for pair in lhs_exc_idxs:
        #     # if pair[0] == 14 or pair[1] == 14:
        #         # print(pair)

        #     if pair[0] == 51 or pair[1] == 51:
        #         print(pair)


        # print(rhs_exc_idxs)
        for pair in rhs_exc_idxs:

            if pair[0] == 14 or pair[1] == 14:
                print(pair)

            if pair[0] == 51 or pair[1] == 51:
                print(pair)

        # assert 0

        lhs_es_exc_param_idxs = np.array(lhs_es_exc_param_idxs, dtype=np.int32) 
        rhs_es_exc_param_idxs = np.array(rhs_es_exc_param_idxs, dtype=np.int32) 
        lhs_lj_exc_param_idxs = np.array(lhs_lj_exc_param_idxs, dtype=np.int32) 
        rhs_lj_exc_param_idxs = np.array(rhs_lj_exc_param_idxs, dtype=np.int32)

        lhs_nrg_fns['Nonbonded'] = (lhs_es_param_idxs, lhs_lj_param_idxs, lhs_exc_idxs, lhs_es_exc_param_idxs, lhs_lj_exc_param_idxs, lambda_plane_idxs, lambda_offset_idxs, a_cutoff)
        rhs_nrg_fns['Nonbonded'] = (rhs_es_param_idxs, rhs_lj_param_idxs, rhs_exc_idxs, rhs_es_exc_param_idxs, rhs_lj_exc_param_idxs, lambda_plane_idxs, lambda_offset_idxs, b_cutoff)

        a_gb_args = a_nrg_fns['GBSA']
        b_gb_args = b_nrg_fns['GBSA']

        a_gb_charges, a_gb_radii, a_gb_scales = a_gb_args[:3]
        b_gb_charges, b_gb_radii, b_gb_scales = b_gb_args[:3]

        assert a_gb_args[3:] == b_gb_args[3:]

        lhs_gb_charges, rhs_gb_charges = lm.mix_nonbonded_parameters(a_gb_charges, b_gb_charges)
        lhs_gb_radii, rhs_gb_radii = lm.mix_nonbonded_parameters(a_gb_radii, b_gb_radii)
        lhs_gb_scales, rhs_gb_scales = lm.mix_nonbonded_parameters(a_gb_scales, b_gb_scales)

        lhs_nrg_fns['GBSA'] = (lhs_gb_charges, lhs_gb_radii, lhs_gb_scales, lambda_plane_idxs, lambda_offset_idxs, *a_gb_args[3:])
        rhs_nrg_fns['GBSA'] = (rhs_gb_charges, rhs_gb_radii, rhs_gb_scales, lambda_plane_idxs, lambda_offset_idxs, *a_gb_args[3:])

        lhs_system = system.System(lhs_nrg_fns, self.params, self.param_groups, combined_masses)
        rhs_system = system.System(rhs_nrg_fns, self.params, self.param_groups, combined_masses)

        return lhs_system, rhs_system


    def merge(self, other):
        """
        Duplicate merge two systems into two sets of parameters.
        """
        a_masses = self.masses
        a_params = self.params
        a_param_groups = self.param_groups
        a_nrgs = self.nrg_fns

        b_masses = other.masses
        b_params = other.params
        b_param_groups = other.param_groups
        b_nrgs = other.nrg_fns

        num_a_atoms = len(a_masses)                     # offset by number of atoms in a
        c_masses = np.concatenate([a_masses, b_masses]) # combined masses
        c_params = np.concatenate([a_params, b_params]) # combined parameters
        c_param_groups = np.concatenate([a_param_groups, b_param_groups]) # combine parameter groups

        assert a_nrgs.keys() == b_nrgs.keys()

        c_nrgs = {}

        for force_type in a_nrgs.keys():

            a_name = force_type
            b_name = force_type

            a_args = a_nrgs[force_type]
            b_args = b_nrgs[force_type]

            if a_name == "HarmonicBond":
                bond_idxs = np.concatenate([a_args[0], b_args[0] + num_a_atoms], axis=0)
                bond_param_idxs = np.concatenate([a_args[1], b_args[1] + len(a_params)], axis=0)
                c_nrgs["HarmonicBond"] = (bond_idxs.astype(np.int32), bond_param_idxs)
            elif a_name == "HarmonicAngle":
                angle_idxs = np.concatenate([a_args[0], b_args[0] + num_a_atoms], axis=0)
                angle_param_idxs = np.concatenate([a_args[1], b_args[1] + len(a_params)], axis=0)
                c_nrgs["HarmonicAngle"] = (angle_idxs.astype(np.int32), angle_param_idxs)
            elif a_name == "PeriodicTorsion":
                torsion_idxs = np.concatenate([a_args[0], b_args[0] + num_a_atoms], axis=0)
                torsion_param_idxs = np.concatenate([a_args[1], b_args[1] + len(a_params)], axis=0)
                c_nrgs["PeriodicTorsion"] = (torsion_idxs.astype(np.int32), torsion_param_idxs)
            elif a_name == "Nonbonded":
                assert a_args[7] == b_args[7] # cutoff
                es_param_idxs = np.concatenate([a_args[0], b_args[0] + len(a_params)], axis=0) # [N,]

                # print(b_params)
                # print(b_args[0])
                print("Ligand pair net charge", np.sum(np.array(b_params)[np.array(b_args[0])]))

                # assert 0

                lj_param_idxs = np.concatenate([a_args[1], b_args[1] + len(a_params)], axis=0)
                exclusion_idxs = np.concatenate([a_args[2], b_args[2] + num_a_atoms], axis=0)

                # exclusions must be unique. so we sanity check them here.
                sorted_exclusion_idxs = set()
                for src, dst in exclusion_idxs:
                    src, dst = sorted((src, dst))
                    sorted_exclusion_idxs.add((src, dst))
                assert len(sorted_exclusion_idxs) == len(exclusion_idxs)

                es_exclusion_param_idxs = np.concatenate([a_args[3], b_args[3] + len(a_params)], axis=0)  # [E, 1]
                lj_exclusion_param_idxs = np.concatenate([a_args[4], b_args[4] + len(a_params)], axis=0)  # [E, 1]

                lambda_plane_idxs = np.concatenate([a_args[5], b_args[5]])
                lambda_offset_idxs = np.concatenate([a_args[6], b_args[6]])

                c_nrgs["Nonbonded"] = (
                    es_param_idxs.astype(np.int32),
                    lj_param_idxs.astype(np.int32),
                    exclusion_idxs.astype(np.int32),
                    es_exclusion_param_idxs.astype(np.int32),
                    lj_exclusion_param_idxs.astype(np.int32),
                    lambda_plane_idxs.astype(np.int32),
                    lambda_offset_idxs.astype(np.int32),
                    a_args[7]
                )

            elif a_name == "GBSA":

                # skip GB
                # print("skipping GB")
                # continue
                charge_param_idxs = np.concatenate([a_args[0], b_args[0] + len(a_params)], axis=0)
                radius_param_idxs = np.concatenate([a_args[1], b_args[1] + len(a_params)], axis=0)
                scale_param_idxs = np.concatenate([a_args[2], b_args[2] + len(a_params)], axis=0)
                lambda_plane_idxs = np.concatenate([a_args[3], b_args[3]])
                lambda_offset_idxs = np.concatenate([a_args[4], b_args[4]])

                # +2 is due to lambda
                assert a_args[3+2] == b_args[3+2] # alpha
                assert a_args[4+2] == b_args[4+2] # beta
                assert a_args[5+2] == b_args[5+2] # gamma
                assert a_args[6+2] == b_args[6+2] # dielec_offset
                assert a_args[7+2] == b_args[7+2] # surface tension
                assert a_args[8+2] == b_args[8+2] # solute dielectric
                assert a_args[9+2] == b_args[9+2] # solvent dielectric
                assert a_args[10+2] == b_args[10+2] # probe_radius

                c_nrgs["GBSA"] = (
                    charge_param_idxs.astype(np.int32),
                    radius_param_idxs.astype(np.int32),
                    scale_param_idxs.astype(np.int32),
                    lambda_plane_idxs.astype(np.int32),
                    lambda_offset_idxs.astype(np.int32),
                    *a_args[5:]
                )

            else:
                raise Exception("Unknown potential", a_name)

        return System(c_nrgs, c_params, c_param_groups, c_masses)

    # def make_alchemical_gradients(self, other, precision):

    #     gradients = []
    #     for k, v in self.nrg_fns.items():
    #         other_v = other.nrg_fns[k]
    #         op_fn = getattr(ops, k)
    #         grad = op_fn(*v, precision=precision)
    #         grad_other = op_fn(*other_v, precision=precision)
    #         grad_alchem = ops.AlchemicalGradient(
    #             len(self.masses),
    #             len(self.params),
    #             grad,
    #             grad_other
    #         )
    #         gradients.append(grad_alchem)

    #     return gradients

    def make_gradients(self, precision):
        """
        Instantiate time-machine based functional forms.
        """
        gradients = []
        for k, v in self.nrg_fns.items():
            op_fn = getattr(ops, k)
            grad = op_fn(*v, precision=precision)
            gradients.append(grad)

        return gradients
