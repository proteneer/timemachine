import jax

import functools
import numpy as np

from ff.handlers.utils import match_smirks, sort_tuple

class BondedHandler():

    @staticmethod
    def generate_bonded_idxs(mol, smirks):
        """
        Generate bond_idxs and param_idxs given a molecule.
        """

        vd = {}

        for p_idx, patt in enumerate(smirks):
            matches = match_smirks(mol, patt)
            for m in matches:
                sorted_m = sort_tuple(m)
                vd[sorted_m] = p_idx

        bond_idxs = np.array(list(vd.keys()), dtype=np.int32)
        param_idxs = np.array(list(vd.values()), dtype=np.int32)

        return bond_idxs, param_idxs

    @staticmethod
    def parameterize_ligand(params, param_idxs):
        return params[param_idxs]

class HarmonicBondHandler(BondedHandler):

    def __init__(self, smirks, params):
        self.smirks = smirks
        self.params = params
        assert len(self.smirks) == len(self.params)

    def parameterize(self, mol):
        """
        Parameterize given molecule

        Parameters
        ----------
        mol: Chem.ROMol
            rdkit molecule, should have hydrogens pre-added

        Returns
        -------
        tuple of (Q,2) (np.int32), ((Q,2), fn: R^Qx2 -> R^Px2))
            System bond idxes, parameters, and the vjp_fn.

        """

        bond_idxs, param_idxs = self.generate_bonded_idxs(mol, self.smirks)

        param_fn = functools.partial(self.parameterize_ligand, param_idxs=param_idxs)

        sys_params, vjp_fn = jax.vjp(param_fn, self.params)

        return bond_idxs, (sys_params, vjp_fn)