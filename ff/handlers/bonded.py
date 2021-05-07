
import functools
import numpy as np

import jax

from ff.handlers.utils import match_smirks, sort_tuple
from ff.handlers.serialize import SerializableMixIn, bin_to_str
from ff.handlers.suffix import _SUFFIX

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
class ReversibleBondHandler(SerializableMixIn):

    def __init__(self, smirks, params, props):
        """ "Reversible" here means that bond energy is symmetric to index reversal
        u_bond(x[i], x[j]) = u_bond(x[j], x[i])"""
        self.smirks = smirks
        self.params = np.array(params, dtype=np.float64)
        self.props = props
        assert len(self.smirks) == len(self.params)

    def lookup_smirks(self, query):
        for s_idx, s in enumerate(self.smirks):
            if s == query:
                return self.params[s_idx]

    def partial_parameterize(self, params, mol):
        return self.static_parameterize(params, self.smirks, mol)

    def parameterize(self, mol):
        return self.static_parameterize(self.params, self.smirks, mol)

    @staticmethod
    def static_parameterize(params, smirks, mol):
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

        bond_idxs, param_idxs = generate_vd_idxs(mol, smirks)
        return params[param_idxs], bond_idxs

# we need to subclass to get the names backout
class HarmonicBondHandler(ReversibleBondHandler):
    pass

class HarmonicAngleHandler(ReversibleBondHandler):
    pass

class ProperTorsionHandler():

    def __init__(self, smirks, params, props):
        """
        Parameters
        ----------
        smirks: list str
            list of smirks patterns

        params: list of list
            each torsion may have a variadic number of terms.

        """
        # self.smirks = smirks

        # raw_params = params # internals is a 
        self.counts = []
        self.smirks = []
        self.params = []
        for smi, terms in zip(smirks, params):
            self.smirks.append(smi)
            self.counts.append(len(terms))
            for term in terms:
                self.params.append(term)
        
        self.counts = np.array(self.counts, dtype=np.int32)
 
        self.params = np.array(self.params, dtype=np.float64)
        self.props = props

    def parameterize(self, mol):
        return self.static_parameterize(self.params, self.smirks, self.counts, mol)

    def partial_parameterize(self, params, mol):
        return self.static_parameterize(params, self.smirks, self.counts, mol)


    @staticmethod
    def static_parameterize(params, smirks, counts, mol):
        torsion_idxs, param_idxs = generate_vd_idxs(mol, smirks)

        assert len(torsion_idxs) == len(param_idxs)

        scatter_idxs = []
        n_smirks = len(counts) # number of patterns
        repeats = []

        # prefix sum of size + 1
        pfxsum = np.concatenate([[0], np.cumsum(counts)]) 
        for p_idx in param_idxs:
            start = pfxsum[p_idx]
            end = pfxsum[p_idx+1]
            scatter_idxs.extend((range(start, end)))
            repeats.append(counts[p_idx])

        # for k, _, _ in params[scatter_idxs]:
            # if k == 0.0:
                # print("WARNING: zero force constant torsion generated.")

        scatter_idxs = np.array(scatter_idxs)

        return params[scatter_idxs], np.repeat(torsion_idxs, repeats, axis=0).astype(np.int32)

    def serialize(self):
        list_params = []
        counter = 0
        for smi_idx, smi in enumerate(self.smirks):
            t_params = []
            for _ in range(self.counts[smi_idx]):
                t_params.append(self.params[counter].tolist())
                counter += 1
            list_params.append(t_params)

        key = type(self).__name__[:-len(_SUFFIX)]
        patterns = []
        for smi, p in zip(self.smirks, list_params):
            patterns.append((smi, p))

        body = {'patterns': patterns}
        result = {key: body}

        return result

class ImproperTorsionHandler(SerializableMixIn):

    def __init__(self, smirks, params, props):
        self.smirks = smirks
        self.params = np.array(params, dtype=np.float64)
        self.props = props
        assert self.params.shape[1] == 3
        assert len(self.smirks) == len(self.params)

    def partial_parameterize(self, params, mol):
        return self.static_parameterize(params, self.smirks, mol)

    def parameterize(self, mol):
        return self.static_parameterize(self.params, self.smirks, mol)


    @staticmethod
    def static_parameterize(params, smirks, mol):

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

        for p_idx, patt in enumerate(smirks):
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

        param_idxs = np.array(param_idxs)

        return params[param_idxs], np.array(improper_idxs, dtype=np.int32)
