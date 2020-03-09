# forcefield

import ast
import numpy as np

from rdkit import Chem
from ff import system
# from timemachine.lib import ops

class Forcefield():

    def __init__(self, handle):
        """
        Initialize the forcefield class.

        Parameters
        ----------
        handle: str or dict
            If str, then the handle is interpret as a path to be opened. If dict,
            then the handle will be used directly
        """

        if isinstance(handle, str):
            handle = open(handle).read()
            ff_raw = ast.literal_eval(handle)
        elif isinstance(handle, dict):
            ff_raw = handle

        global_params = []
        global_param_groups = []

        def add_param(p, p_group):
            assert isinstance(p_group, int)
            length = len(global_params)
            global_params.append(p)
            global_param_groups.append(p_group)
            return length

        # recursively replace parameters with indices and appending them into a global list.
        def recursive_replace(val, p_group):
            if isinstance(val, list):
                arr = []
                for f_idx, f in enumerate(val):
                    if isinstance(f, list):
                        fg = p_group
                    else:
                        fg = p_group[f_idx]
                    v = recursive_replace(f, fg)
                    arr.append(v)
                return arr
            elif isinstance(val, float) or isinstance(val, int):
                return add_param(val, p_group)
            else:
                raise Exception("Unsupported type")

        # (ytz): temporary useful debug code. remove later
        # print(recursive_replace([0.5, 0.6], (2,3)))
        # print(recursive_replace([[4.0, 2.0], [5.0, 1.0], [4.0, 233.0], [645.0, 1.0]], (11,12)))
        # print(global_params)
        # print(global_param_groups)
        # assert 0

        group_map = {
            "Angle": (0,1),
            "Bond": (2,3),
            "Improper": (4,5,6),
            "Proper": (7,8,9),
            "vdW": (10,11),
            "GBSA": (12,13),
            "SimpleCharges": (14,)
        }

        self.forcefield = {}
        # convert raw to proper
        for force_type, values in ff_raw.items():
            new_params = []
            for v in values["params"]:
                smirks = v[0]
                params = v[1:]
                param_idxs = recursive_replace(params, group_map[force_type])
                new_params.append([smirks, *param_idxs])
            self.forcefield[force_type] = {}
            self.forcefield[force_type]["params"] = new_params
            if "props" in values:
                self.forcefield[force_type]["props"] = values["props"]

        assert len(global_params) == len(global_param_groups)
        
        self.params = global_params
        self.param_groups = global_param_groups

        # hacky temp code to deal with exclusions
        exclusion_param = 1.0
        exclusion_param_group = 20
        self.params.append(exclusion_param)
        self.param_groups.append(exclusion_param_group)

    def get_exclusion_idx(self):
        return len(self.params)-1

    def serialize(self):
        """
        Serialize the forcefield to an python dictionary.
        """

        def recursive_lookup(val):
            if isinstance(val, list):
                arr = []
                for f in val:
                    v = recursive_lookup(f)
                    arr.append(v)
                return arr
            elif isinstance(val, int):
                return self.params[val]
            else:
                raise Exception("Unsupported type")

        raw_ff = {}

        for force_type, values in self.forcefield.items():
            raw_ff[force_type] = {}
            new_params = []
            for v in values["params"]:
                smirks = v[0]
                param_idxs = v[1:]
                # print(smirks, param_idxs)
                param_vals = recursive_lookup(param_idxs)
                new_params.append([smirks, *param_vals])

            raw_ff[force_type]["params"] = new_params
            if "props" in values:
                raw_ff[force_type]["props"] = values["props"]

        return raw_ff

    def parameterize(self, mol):
        """
        Given a RDKit Molecule, return a parameterized system.
        """

        def match_smirks(mol, smirks):
            
            # Make a copy of the molecule
            rdmol = Chem.Mol(mol)
            # Use designated aromaticity model
            Chem.SanitizeMol(rdmol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_SETAROMATICITY)
            Chem.SetAromaticity(rdmol, Chem.AromaticityModel.AROMATICITY_MDL)
            
            # Set up query.
            qmol = Chem.MolFromSmarts(smirks)  #cannot catch the error
            if qmol is None:
                raise ValueError('RDKit could not parse the SMIRKS string "{}"'.format(smirks))

            # Create atom mapping for query molecule
            idx_map = dict()
            for atom in qmol.GetAtoms():
                smirks_index = atom.GetAtomMapNum()
                if smirks_index != 0:
                    idx_map[smirks_index - 1] = atom.GetIdx()
            map_list = [idx_map[x] for x in sorted(idx_map)]

            # Perform matching
            matches = list()
            for match in rdmol.GetSubstructMatches(qmol, uniquify=False):
                mas = [match[x] for x in map_list]
                matches.append(tuple(mas))

            return matches

        def sort_tuple(arr):

            container_type = type(arr)

            if len(arr) == 0:
                raise Exception("zero sized array")
            elif len(arr) == 1:
                return arr
            elif arr[0] > arr[-1]:
                return container_type(reversed(arr))
            else:
                return arr

        nrg_fns = {}

        torsion_idxs = []
        torsion_param_idxs = []


        exclusion_param_idx = self.get_exclusion_idx()

        exclusions = {}

        for force_type, values in self.forcefield.items():

            params = values["params"]

            def make_vd():
                vd = {}
                for p_idx, p in enumerate(params):
                    smirks = p[0]
                    matches = match_smirks(mol, smirks)
                    for m in matches:
                        sorted_m = sort_tuple(m)
                        vd[sorted_m] = p_idx
                return vd

            vd = make_vd()

            if force_type == "Bond":

                bond_idxs = []
                bond_param_idxs = []

                for atom_idxs, p_idx in vd.items():
                    bond_idxs.append(atom_idxs)
                    pp = params[p_idx]
                    k_idx, b_idx = pp[1], pp[2]
                    bond_param_idxs.append((k_idx, b_idx))

                    src, dst = atom_idxs
                    assert src < dst
                    # assert (src, dst) not in exclusions
                    # print("bonds", src, dst)
                    exclusions[(src, dst)] = exclusion_param_idx

                nrg_fns['HarmonicBond'] = (
                    np.array(bond_idxs, dtype=np.int32),
                    np.array(bond_param_idxs, dtype=np.int32)
                )

            elif force_type == "Angle":

                angle_idxs = []
                angle_param_idxs = []

                for atom_idxs, p_idx in vd.items():
                    angle_idxs.append(atom_idxs)
                    pp = params[p_idx]
                    k_idx, a_idx = pp[1], pp[2]
                    angle_param_idxs.append((k_idx, a_idx))
                    src, _, dst = atom_idxs
                    assert src < dst
                    # assert (src, dst) not in exclusions
                    # print("angle", src, dst)
                    exclusions[(src, dst)] = exclusion_param_idx

                nrg_fns['HarmonicAngle'] = (
                    np.array(angle_idxs, dtype=np.int32),
                    np.array(angle_param_idxs, dtype=np.int32)
                )

            elif force_type == "Proper":

                for atom_idxs, p_idx in vd.items():
                    pp = params[p_idx]
                    components = pp[1]
                    for proper_torsion in components:
                        torsion_idxs.append(atom_idxs)
                        torsion_param_idxs.append(proper_torsion)

                    src, _, _, dst = atom_idxs
                    assert src < dst
                    # print("torsion exclusions", atom_idxs)
                    # assert (src, dst) not in exclusions
                    exclusions[(src, dst)] = exclusion_param_idx

            # elif force_type == 'Improper':

            #     for atom_idxs, p_idx in vd.items():
            #         pp = params[p_idx]
            #         k_idx, phase_idx, period_idx = pp[1], pp[2], pp[3]
            #         m = atom_idxs

            #         # trefoil
            #         others = [atom_idxs[0], atom_idxs[2], atom_idxs[3]]
            #         for p in [(others[i], others[j], others[k]) for (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]]:
            #             improper_idx = (atom_idxs[1], p[0], p[1], p[2])
            #             torsion_idxs.append(improper_idx)
            #             torsion_param_idxs.append((k_idx, phase_idx, period_idx))

            elif force_type == 'vdW':

                lj_param_idxs = []

                for atom_idx, p_idx in vd.items():
                    pp = params[p_idx]
                    sig_idx, eps_idx = pp[1], pp[2]
                    lj_param_idxs.append((sig_idx, eps_idx))

            elif force_type == 'SimpleCharges':

                es_param_idxs = []

                for atom_idx, p_idx in vd.items():
                    pp = params[p_idx]
                    q_idx = pp[1]
                    es_param_idxs.append(q_idx)

            elif force_type == "GBSA":

                gb_radii_idxs = []
                gb_scale_idxs = []
                # gb_props = []
                for atom_idx, p_idx in vd.items():
                    pp = params[p_idx]
                    radii_idx, scale_idx = pp[1], pp[2]
                    gb_radii_idxs.append(radii_idx)
                    gb_scale_idxs.append(scale_idx)

                props = values["props"]
                gb_args = (
                    props["alpha"],
                    props["beta"],
                    props["gamma"],
                    props["dielectric_offset"],
                    props["surface_tension"],
                    props["solute_dielectric"],
                    props["solvent_dielectric"],
                    props["probe_radius"],
                    10000.0
                )

        nrg_fns['PeriodicTorsion'] = (
            np.array(torsion_idxs, dtype=np.int32),
            np.array(torsion_param_idxs, dtype=np.int32)
        )

        exclusion_idxs = []
        exclusion_param_idxs = []

        for k, v in exclusions.items():
            exclusion_idxs.append(k)
            exclusion_param_idxs.append(v)

        exclusion_idxs = np.array(exclusion_idxs)


        # assert 0

        nrg_fns['Nonbonded'] = (
            np.array(es_param_idxs, dtype=np.int32),
            np.array(lj_param_idxs, dtype=np.int32),
            np.array(exclusion_idxs, dtype=np.int32),
            np.array(exclusion_param_idxs, dtype=np.int32),
            np.array(exclusion_param_idxs, dtype=np.int32),
            10000.0
        )

        nrg_fns['GBSA'] = (
            np.array(es_param_idxs, dtype=np.int32),
            np.array(gb_radii_idxs, dtype=np.int32),
            np.array(es_param_idxs, dtype=np.int32),
            *gb_args
        )

        return nrg_fns
