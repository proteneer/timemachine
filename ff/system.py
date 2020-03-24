import numpy as np

from timemachine.lib import ops

class System():

    def __init__(self, nrg_fns, params, param_groups, masses):

        self.nrg_fns = nrg_fns
        self.params = params
        self.param_groups = param_groups
        self.masses = masses

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
                c_nrgs["HarmonicBond"] = (bond_idxs, bond_param_idxs)
            elif a_name == "HarmonicAngle":
                angle_idxs = np.concatenate([a_args[0], b_args[0] + num_a_atoms], axis=0)
                angle_param_idxs = np.concatenate([a_args[1], b_args[1] + len(a_params)], axis=0)
                c_nrgs["HarmonicAngle"] = (angle_idxs, angle_param_idxs)
            elif a_name == "PeriodicTorsion":
                torsion_idxs = np.concatenate([a_args[0], b_args[0] + num_a_atoms], axis=0)
                torsion_param_idxs = np.concatenate([a_args[1], b_args[1] + len(a_params)], axis=0)
                c_nrgs["PeriodicTorsion"] = (torsion_idxs, torsion_param_idxs)
            elif a_name == "Nonbonded":
                assert a_args[5] == b_args[5] # cutoff
                es_param_idxs = np.concatenate([a_args[0], b_args[0] + len(a_params)], axis=0) # [N,]
                lj_param_idxs = np.concatenate([a_args[1], b_args[1] + len(a_params)], axis=0)

                exclusion_idxs = np.concatenate([a_args[2], b_args[2] + num_a_atoms], axis=0)

                es_exclusion_param_idxs = np.concatenate([a_args[3], b_args[3] + len(a_params)], axis=0)  # [E, 1]
                lj_exclusion_param_idxs = np.concatenate([a_args[4], b_args[4] + len(a_params)], axis=0)  # [E, 1]

                c_nrgs["Nonbonded"] = (
                    es_param_idxs,
                    lj_param_idxs,
                    exclusion_idxs,
                    es_exclusion_param_idxs,
                    lj_exclusion_param_idxs,
                    a_args[5]
                )
            elif a_name == "GBSA":

                # skip GB
                print("skipping GB")
                continue

                charge_param_idxs = np.concatenate([a_args[0], b_args[0] + len(a_params)], axis=0)
                radius_param_idxs = np.concatenate([a_args[1], b_args[1] + len(a_params)], axis=0)
                scale_param_idxs = np.concatenate([a_args[2], b_args[2] + len(a_params)], axis=0)

                assert a_args[3] == b_args[3] # alpha
                assert a_args[4] == b_args[4] # beta
                assert a_args[5] == b_args[5] # gamma
                assert a_args[6] == b_args[6] # dielec_offset
                assert a_args[7] == b_args[7] # surface tension
                assert a_args[8] == b_args[8] # solute dielectric
                assert a_args[9] == b_args[9] # solvent dielectric
                assert a_args[10] == b_args[10] # probe_radius

                c_nrgs["GBSA"] = (
                    charge_param_idxs.astype(np.int32),
                    radius_param_idxs.astype(np.int32),
                    scale_param_idxs.astype(np.int32),
                    *a_args[3:]
                )

            else:
                raise Exception("Unknown potential", a_name)

        return System(c_nrgs, c_params, c_param_groups, c_masses)

    def make_gradients(self, dimension, precision):
        """
        Instantiate time-machine based functional forms.
        """
        gradients = []
        for k, v in self.nrg_fns.items():
            op_fn = getattr(ops, k)
            grad = op_fn(*v, dimension, precision=precision)
            gradients.append(grad)

        return gradients