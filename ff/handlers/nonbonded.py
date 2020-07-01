import functools

import numpy as np
import jax
import jax.numpy as jnp
import networkx as nx

from rdkit import Chem

from ff.handlers.utils import match_smirks, sort_tuple
from simtk.openmm.app.forcefield import _findExclusions

from timemachine import constants

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


class NonbondedHandler:

    def __init__(self, smirks, params, props):
        """
        Parameters
        ----------
        smirks: list of str (P,)
            SMIRKS patterns for each pattern

        params: np.array, (P,)
            normalized charge for each

        """
        
        assert len(smirks) == params.shape[0]

        self.smirks = smirks
        self.params = params
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

