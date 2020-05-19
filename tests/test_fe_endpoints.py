# test that endpoints are analytically correct.

import numpy as np
import unittest
from rdkit import Chem
from ff import forcefield
from fe import linear_mixer, atom_mapping
from timemachine.lib import ops

def swap_coords(a_conf, b_conf, a_to_b):

    b_to_a = {}
    for src, dst in a_to_b.items():
        b_to_a[dst] = src

    new_a_conf = []
    for a_idx, coords in enumerate(a_conf):
        if a_idx in a_to_b:
            new_a_conf.append(b_conf[a_to_b[a_idx]])
        else:
            new_a_conf.append(coords)

    new_b_conf = []
    for b_idx, coords in enumerate(b_conf):
        if b_idx in b_to_a:
            new_b_conf.append(a_conf[b_to_a[b_idx]])
        else:
            new_b_conf.append(coords)

    return new_a_conf, new_b_conf


class TestEndpoints(unittest.TestCase):

    def setUp(self):

        self.ff = forcefield.Forcefield("ff/smirnoff_1.1.0.py")
        suppl = Chem.SDMolSupplier("tests/hif2a_ligands.sdf", removeHs=False)

        all_guest_mols = []
        for guest_idx, guest_mol in enumerate(suppl):
            all_guest_mols.append(guest_mol)

        self.mol_a = all_guest_mols[0]
        self.mol_b = all_guest_mols[1]
        self.atom_mapping_a_to_b = atom_mapping.mcs_map(
            self.mol_a,
            self.mol_b,
            variant='Nonbonded'
        )

        n_a = self.mol_a.GetNumAtoms()

        # try am1=True later
        self.a_system = self.ff.parameterize(self.mol_a, cutoff=10000.0, am1=False)
        self.b_system = self.ff.parameterize(self.mol_b, cutoff=10000.0, am1=False)


        conformer = self.mol_a.GetConformer(0)
        self.mol_a_conf = np.array(conformer.GetPositions(), dtype=np.float64)
        self.mol_a_conf = self.mol_a_conf/10 # convert to md_units

        conformer = self.mol_b.GetConformer(0)
        self.mol_b_conf = np.array(conformer.GetPositions(), dtype=np.float64)
        self.mol_b_conf = self.mol_b_conf/10 # convert to md_units

    def test_bond_mixing(self):

        for bonded_type in ['HarmonicBond', 'HarmonicAngle', 'PeriodicTorsion']:

            print("Testing mixing of", bonded_type)

            a_bond_idxs, a_param_idxs = self.a_system.nrg_fns[bonded_type]
            b_bond_idxs, b_param_idxs = self.b_system.nrg_fns[bonded_type]

            mixer = linear_mixer.LinearMixer(self.mol_a.GetNumAtoms(), self.atom_mapping_a_to_b)
            lhs_bond_idxs, lhs_param_idxs, rhs_bond_idxs, rhs_param_idxs = mixer.mix_arbitrary_bonds(
                a_bond_idxs,
                a_param_idxs,
                b_bond_idxs,
                b_param_idxs
            )

            op_fn = getattr(ops, bonded_type)

            rhs_op = op_fn(
                rhs_bond_idxs.astype(np.int32),
                rhs_param_idxs.astype(np.int32),
                precision=np.float64
            )

            x_ab = np.concatenate([self.mol_a_conf, self.mol_b_conf])

            # (ytz): lambda doesn't actually matter
            test_grads, test_du_dl, test_energy = rhs_op.execute_lambda(x_ab, self.ff.params, 0.0)
            ref_bond_idxs = np.concatenate([b_bond_idxs, a_bond_idxs+self.mol_b.GetNumAtoms()])
            ref_param_idxs = np.concatenate([b_param_idxs, a_param_idxs])

            ref_op = op_fn(
                ref_bond_idxs.astype(np.int32),
                ref_param_idxs.astype(np.int32),
                precision=np.float64
            )

            new_a_conf, new_b_conf = swap_coords(self.mol_a_conf, self.mol_b_conf, self.atom_mapping_a_to_b)
            x_ba = np.concatenate([new_b_conf, new_a_conf])
            ref_grads, ref_du_dl, ref_energy = ref_op.execute_lambda(x_ba, self.ff.params, 0.0)

            np.testing.assert_almost_equal(test_energy, ref_energy)

