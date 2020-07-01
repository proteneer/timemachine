
import functools
import numpy as np

import jax

from ff.handlers.utils import match_smirks, sort_tuple

def generate_vd_idxs(mol, smirks):
    """
    Generate bonded indices using a valence dict. The indices generated
    are assumed to be reversible. i.e. reversing the indices evaluates to
    an identical energy. This is not intended to be used for ImproperTorsions.
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

def parameterize_ligand(params, param_idxs):
    return params[param_idxs]

# its trivial to re-use this for everything except the ImproperTorsions
class ReversibleBondHandler:

    def __init__(self, smirks, params, props):
        self.smirks = smirks
        self.params = params
        self.props = props
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

        bond_idxs, param_idxs = generate_vd_idxs(mol, self.smirks)
        param_fn = functools.partial(parameterize_ligand, param_idxs=param_idxs)
        sys_params, vjp_fn = jax.vjp(param_fn, self.params)

        return np.array(bond_idxs, dtype=np.int32), (np.array(sys_params, dtype=np.float64), vjp_fn)

# we need to subclass to get the names backout
class HarmonicBondHandler(ReversibleBondHandler):
    pass

class HarmonicAngleHandler(ReversibleBondHandler):
    pass

class PeriodicTorsionHandler(ReversibleBondHandler):
    pass

# HarmonicBondHandler = ReversibleBondHandler
# HarmonicAngleHandler = ReversibleBondHandler
# PeriodicTorsionHandler = ReversibleBondHandler

class ImproperTorsionHandler:

    def __init__(self, smirks, params, props):
        self.smirks = smirks
        self.params = params
        self.props = props
        assert self.params.shape[1] == 3
        assert len(self.smirks) == len(self.params)

    def parameterize(self, mol):

        # improper torsions do not use a valence dict as
        # we cannot sort based on b_idxs[0] and b_idxs[-1]
        # and reverse if needed. Impropers are centered around
        # the first atom.
        impropers = dict()

        def make_key(idxs):
            assert len(idxs) == 4
            # pivot around the center
            ctr = idxs[1]
            # sort the neighbors so they're unique
            nbs = idxs[0], idxs[2], idxs[3]
            nbs = sorted(nbs)
            return nbs[0], ctr, nbs[1], nbs[2]

        for p_idx, patt in enumerate(self.smirks):
            matches = match_smirks(mol, patt)

            for m in matches:
                key = make_key(m)
                impropers[key] = p_idx

        improper_idxs = []
        param_idxs = []

        for atom_idxs, p_idx in impropers.items():
            center = atom_idxs[1]
            others = [atom_idxs[0], atom_idxs[2], atom_idxs[3]]
            for p in [(others[i], others[j], others[k]) for (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]]:
                improper_idxs.append((center, p[0], p[1], p[2]))
                param_idxs.append(p_idx)


        param_fn = functools.partial(parameterize_ligand, param_idxs=param_idxs)
        sys_params, vjp_fn = jax.vjp(param_fn, self.params)

        return np.array(improper_idxs, dtype=np.int32), (np.array(sys_params, dtype=np.float64), vjp_fn)
