import functools

import numpy as np
import jax
import jax.numpy as jnp
import networkx as nx

from rdkit import Chem

from ff.handlers.utils import match_smirks, sort_tuple
from ff.handlers.serialize import SerializableMixIn

from timemachine import constants

from jax import ops

def convert_to_nx(mol):
    """
    Convert an ROMol into a networkx graph.
    """
    g = nx.Graph()

    for atom in mol.GetAtoms():
        g.add_node(atom.GetIdx())

    for bond in mol.GetBonds():
        g.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    return g


def generate_exclusion_idxs(mol, scale12, scale13, scale14):
    """ 
    Generate exclusions for a mol based on the all pairs shortest path.
    We always take the convention that exclusions for smaller distances
    override those of longer distances.

    Parameters
    ----------
    mol: Chem.ROMol
        romol

    scale12: float
        bond scales

    scale13: float
        angle scales

    scale14: float
        torsions scales

    """

    exclusions = {}

    g = convert_to_nx(mol)       
    for path in nx.all_pairs_shortest_path_length(g, cutoff=3):
        src = path[0]
        for dst, length in path[1].items():
            if length == 0:
                continue
            else:
                if length == 1:
                    scale = scale12
                elif length == 2:
                    scale = scale13
                elif length == 3:
                    scale = scale14
                else:
                    assert 0

                exclusions[sort_tuple((src, dst))] = scale

    idxs = list(exclusions.keys())
    scales = list(exclusions.values())

    return np.array(idxs, dtype=np.int32), np.array(scales, dtype=np.float64)


def generate_nonbonded_idxs(mol, smirks):
    """
    Parameterize Nonbonded indices given a mol.

    Parameters
    ----------
    smirks: list of str
        SMIRKS patterns for the forcefield type.

    mol: ROMol
        RDKit ROMol object.

    """
    N = mol.GetNumAtoms()
    param_idxs = np.zeros(N, dtype=np.int32)
    for p_idx, patt in enumerate(smirks):
        matches = match_smirks(mol, patt)
        for m in matches:
            param_idxs[m[0]] = p_idx

    return param_idxs

def parameterize_ligand(params, param_idxs):
    return params[param_idxs]

def apply_bcc(params, bond_idxs, bond_idx_params, am1_charges):
    deltas = params[bond_idx_params]
    incremented = ops.index_add(am1_charges, bond_idxs[:, 0], deltas)
    decremented = ops.index_add(incremented, bond_idxs[:, 1], -deltas)    
    return decremented 

class NonbondedHandler(SerializableMixIn):

    def __init__(self, smirks, params, props):
        """
        Parameters
        ----------
        smirks: list of str (P,)
            SMIRKS patterns for each pattern

        params: np.array, (P,)
            normalized charge for each

        """
        
        assert len(smirks) == len(params)

        self.smirks = smirks
        self.params = np.array(params, dtype=np.float64)
        self.props = props

    def parameterize(self, mol):
        """
        Carry out parameterization of given molecule, with an option to attach additional parameters
        via concateation. Typically aux_params are protein charges etc.

        Parameters
        ----------
        mol: Chem.ROMol
            rdkit molecule, should have hydrogens pre-added

        """
        param_idxs = generate_nonbonded_idxs(mol, self.smirks)
        param_fn = functools.partial(parameterize_ligand, param_idxs=param_idxs)
        return jax.vjp(param_fn, self.params)

class SimpleChargeHandler(NonbondedHandler):
    pass

class LennardJonesHandler(NonbondedHandler):
    pass

class GBSAHandler(NonbondedHandler):
    pass

class AM1BCCHandler():

    def parameterize(self, mol):
        """
        Parameters
        ----------

        mol: Chem.ROMol
            molecule to be parameterized.

        """
        # imported here for optional dependency
        from openeye import oechem
        from openeye import oequacpac

        mb = Chem.MolToMolBlock(mol)
        ims = oechem.oemolistream()
        ims.SetFormat(oechem.OEFormat_SDF)
        ims.openstring(mb)

        for buf_mol in ims.GetOEMols():
            oemol = oechem.OEMol(buf_mol)

        result = oequacpac.OEAssignCharges(oemol, oequacpac.OEAM1BCCELF10Charges())

        if result is False:
            raise Exception('Unable to assign charges')

        charges = [] 
        for index, atom in enumerate(oemol.GetAtoms()):
            q = atom.GetPartialCharge()*np.sqrt(constants.ONE_4PI_EPS0)
            charges.append(q)

        def vjp_fn(*args, **kwargs):
            return None 

        return np.array(charges, dtype=np.float64), vjp_fn

class AM1CCCHandler(SerializableMixIn):

    def __init__(self, smirks, params, props):
        """
        Parameters
        ----------
        smirks: list of str (P,)
            SMIRKS patterns for each pattern

        params: np.array, (P,)
            normalized charge for each

        """
        
        assert len(smirks) == len(params)

        self.smirks = smirks
        self.params = np.array(params, dtype=np.float64)
        self.props = props

    def parameterize(self, mol):
        """
        Parameters
        ----------

        mol: Chem.ROMol
            molecule to be parameterized.

        """
        # imported here for optional dependency
        from openeye import oechem
        from openeye import oequacpac

        mb = Chem.MolToMolBlock(mol)
        ims = oechem.oemolistream()
        ims.SetFormat(oechem.OEFormat_SDF)
        ims.openstring(mb)

        for buf_mol in ims.GetOEMols():
            oemol = oechem.OEMol(buf_mol)

        result = oequacpac.OEAssignCharges(oemol, oequacpac.OEAM1Charges())

        if result is False:
            raise Exception('Unable to assign charges')

        am1_charges = [] 
        for index, atom in enumerate(oemol.GetAtoms()):
            q = atom.GetPartialCharge()*np.sqrt(constants.ONE_4PI_EPS0)
            am1_charges.append(q)

        bond_idxs = []
        bond_idx_params = []

        for index in range(len(self.smirks)):
            smirk = self.smirks[index]  
            param = self.params[index]*np.sqrt(constants.ONE_4PI_EPS0)

            substructure_search = oechem.OESubSearch(smirk)
            substructure_search.SetMaxMatches(0)

            matched_bonds = []
            matches = []
            for match in substructure_search.Match(oemol):
                
                matched_indices = {
                    atom_match.pattern.GetMapIdx() - 1: atom_match.target.GetIdx()
                    for atom_match in match.GetAtoms()
                    if atom_match.pattern.GetMapIdx() != 0
                }
               
                matches.append(matched_indices)

            for matched_indices in matches:

                forward_matched_bond = [matched_indices[0], matched_indices[1]]
                reverse_matched_bond = [matched_indices[1], matched_indices[0]]

                if (
                    forward_matched_bond in matched_bonds
                    or reverse_matched_bond in matched_bonds
                    or forward_matched_bond in bond_idxs 
                    or reverse_matched_bond in bond_idxs
                ):
                    continue

                matched_bonds.append(forward_matched_bond)
                bond_idxs.append(forward_matched_bond)
                bond_idx_params.append(index)
        
        bcc_fn = functools.partial(
            apply_bcc,
            bond_idxs = np.array(bond_idxs),
            bond_idx_params = np.array(bond_idx_params, dtype=np.int32),
            am1_charges = np.array(am1_charges)
            )

        charges, vjp_fn = jax.vjp(bcc_fn, self.params)

        return np.array(charges, dtype=np.float64), vjp_fn
