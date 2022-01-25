import functools

import numpy as np
import jax
import jax.numpy as jnp
import networkx as nx
import pickle
from collections import Counter

from rdkit import Chem
from ff.handlers.utils import sort_tuple, match_smirks as rd_match_smirks
from ff.handlers.bcc_aromaticity import match_smirks as oe_match_smirks
from ff.handlers.serialize import SerializableMixIn
from ff.handlers.bcc_aromaticity import AromaticityModel

from timemachine import constants

from jax import ops

import base64


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


def convert_to_oe(mol):
    """Convert an ROMol into an OEMol"""

    # imported here for optional dependency
    from openeye import oechem

    mb = Chem.MolToMolBlock(mol)
    ims = oechem.oemolistream()
    ims.SetFormat(oechem.OEFormat_SDF)
    ims.openstring(mb)

    for buf_mol in ims.GetOEMols():
        oemol = oechem.OEMol(buf_mol)

    return oemol


def oe_assign_charges(mol, charge_model="AM1BCCELF10"):
    """assign partial charges, then premultiply by sqrt(ONE_4PI_EPS0)
    as an optimization"""

    # imported here for optional dependency
    from openeye import oequacpac

    charge_engines = {
        "AM1": oequacpac.OEAM1Charges(symmetrize=True),
        "AM1BCC": oequacpac.OEAM1BCCCharges(symmetrize=True),
        "AM1BCCELF10": oequacpac.OEAM1BCCELF10Charges(),
    }
    charge_engine = charge_engines[charge_model]

    oemol = convert_to_oe(mol)
    result = oequacpac.OEAssignCharges(oemol, charge_engine)
    if result is False:
        raise Exception("Unable to assign charges")

    partial_charges = np.array([atom.GetPartialCharge() for atom in oemol.GetAtoms()])

    # https://github.com/proteneer/timemachine#forcefield-gotchas
    # "The charges have been multiplied by sqrt(ONE_4PI_EPS0) as an optimization."
    inlined_constant = np.sqrt(constants.ONE_4PI_EPS0)

    return inlined_constant * partial_charges


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

    Returns
    -------
    idxs : int array
    scales: float array
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
    mol: ROMol
        RDKit ROMol object.

    smirks: list of str
        SMIRKS patterns for the forcefield type.

    Returns
    -------
    param_idxs

    """
    N = mol.GetNumAtoms()
    param_idxs = np.zeros(N, dtype=np.int32)
    for p_idx, patt in enumerate(smirks):
        matches = rd_match_smirks(mol, patt)
        for m in matches:
            param_idxs[m[0]] = p_idx

    return param_idxs


def parameterize_ligand(params, param_idxs):
    return params[param_idxs]


def compute_or_load_am1_charges(mol):
    """Unless already cached in mol's "AM1Cache" property, use OpenEye to compute AM1 partial charges."""

    # check for cache
    cache_key = "AM1Cache"
    if not mol.HasProp(cache_key):
        # The charges returned by OEQuacPac is not deterministic across OS platforms. It is known
        # to be an issue that the atom ordering modifies the return values as well. A follow up
        # with OpenEye is in order
        # https://github.com/openforcefield/openff-toolkit/issues/983
        am1_charges = list(oe_assign_charges(mol, "AM1"))

        mol.SetProp(cache_key, base64.b64encode(pickle.dumps(am1_charges)))

    else:
        am1_charges = pickle.loads(base64.b64decode(mol.GetProp(cache_key)))

    return np.array(am1_charges)


def bond_smirks_matches(mol, smirks_list):
    """Return an array of ordered bonds and an array of their assigned types

    Notes
    -----
    * Uses OpenEye for substructure searches
    * Order within smirks_list matters
        "First match wins."
        For example, if bond (a,b) can be matched by smirks_list[2], smirks_list[5], ..., assign type 2
    * Order within each smirks pattern matters
        For example, "[#6:1]~[#1:2]" and "[#1:1]~[#6:2]" will match atom pairs in the opposite order
    """
    oemol = convert_to_oe(mol)
    AromaticityModel.assign(oemol)

    bond_idxs = []  # [B, 2]
    type_idxs = []  # [B]

    for type_idx, smirks in enumerate(smirks_list):
        matches = oe_match_smirks(smirks, oemol)

        for matched_indices in matches:
            a, b = matched_indices[0], matched_indices[1]
            forward_matched_bond = [a, b]
            reverse_matched_bond = [b, a]

            already_assigned = forward_matched_bond in bond_idxs or reverse_matched_bond in bond_idxs

            if not already_assigned:
                bond_idxs.append(forward_matched_bond)
                type_idxs.append(type_idx)

    return np.array(bond_idxs), np.array(type_idxs)


def apply_bond_charge_corrections(initial_charges, bond_idxs, deltas):
    """For an arbitrary collection of ordered bonds and associated increments `(a, b, delta)`,
    update `charges` by `charges[a] += delta`, `charges[b] -= delta`

    Notes
    -----
    * preserves sum(initial_charges) for arbitrary values of bond_idxs or deltas
    * order within each row of bond_idxs is meaningful
        `(..., bond_idxs, deltas)`
        means the opposite of
        `(..., bond_idxs[:, ::-1], deltas)`
    * order within the first axis of bond_idxs, deltas is not meaningful
        `(..., bond_idxs[perm], deltas[perm])`
        means the same thing for any permutation `perm`
    """

    # apply bond charge corrections
    incremented = ops.index_add(initial_charges, bond_idxs[:, 0], +deltas)
    decremented = ops.index_add(incremented, bond_idxs[:, 1], -deltas)
    final_charges = decremented

    # make some safety assertions
    assert bond_idxs.shape[1] == 2
    assert len(deltas) == len(bond_idxs)

    net_charge = jnp.sum(initial_charges)
    net_charge_is_integral = jnp.isclose(net_charge, jnp.round(net_charge), atol=1e-5)

    final_net_charge = jnp.sum(final_charges)
    net_charge_is_unchanged = jnp.isclose(final_net_charge, net_charge, atol=1e-5)

    assert net_charge_is_integral
    assert net_charge_is_unchanged

    # print some safety warnings
    directed_bonds = Counter([tuple(b) for b in bond_idxs])
    undirected_bonds = Counter([tuple(sorted(b)) for b in bond_idxs])

    if max(directed_bonds.values()) > 1:
        duplicates = [bond for (bond, count) in directed_bonds.items() if count > 1]
        print(UserWarning(f"Duplicate directed bonds! {duplicates}"))
    elif max(undirected_bonds.values()) > 1:
        duplicates = [bond for (bond, count) in undirected_bonds.items() if count > 1]
        print(UserWarning(f"Duplicate undirected bonds! {duplicates}"))

    return final_charges


class NonbondedHandler(SerializableMixIn):
    def __init__(self, smirks, params, props):
        """
        Parameters
        ----------
        smirks: list of str (P,)
            SMIRKS patterns

        params: np.array, (P, k)
            parameters associated with each SMIRKS pattern

        props: any
        """

        assert len(smirks) == len(params)

        self.smirks = smirks
        self.params = np.array(params, dtype=np.float64)
        self.props = props

    def partial_parameterize(self, params, mol):
        return self.static_parameterize(params, self.smirks, mol)

    def parameterize(self, mol):
        return self.static_parameterize(self.params, self.smirks, mol)

    @staticmethod
    def static_parameterize(params, smirks, mol):
        """
        Carry out parameterization of given molecule, with an option to attach additional parameters
        via concatenation. Typically aux_params are protein charges etc.

        Parameters
        ----------
        params: np.array, (P, k)
            parameters associated with each SMIRKS pattern

        smirks: list of str (P,)
            SMIRKS patterns

        mol: Chem.ROMol
            rdkit molecule, should have hydrogens pre-added

        """
        assert len(smirks) == len(params)
        param_idxs = generate_nonbonded_idxs(mol, smirks)
        return params[param_idxs]


class SimpleChargeHandler(NonbondedHandler):
    pass


class LennardJonesHandler(NonbondedHandler):
    @staticmethod
    def static_parameterize(params, smirks, mol):
        """
        Parameters
        ----------
        params: np.array of shape (P, 2)
            Lennard-Jones associated with each SMIRKS pattern
            params[:, 0] = 2 * sqrt(sigmas)
            params[:, 1] = sqrt(epsilons)

        smirks: list of str (P,)
            SMIRKS patterns

        mol: Chem.ROMol
            molecule to be parameterized

        Returns
        -------
        applied_parameters: np.array of shape (N, 2)

        """
        param_idxs = generate_nonbonded_idxs(mol, smirks)
        params = params[param_idxs]
        sigmas = params[:, 0]
        epsilons = params[:, 1]
        # the raw parameters already in sqrt form.
        # sigmas need to be divided by two
        return jnp.stack([sigmas / 2, epsilons], axis=1)


class GBSAHandler(NonbondedHandler):
    pass


class AM1Handler(SerializableMixIn):
    def __init__(self, smirks, params, props):
        assert len(smirks) == 0
        assert len(params) == 0
        assert props is None

    def partial_parameterize(self, mol):
        return self.static_parameterize(self.smirks, mol)

    def parameterize(self, mol):
        return self.static_parameterize(mol)

    @staticmethod
    def static_parameterize(mol):
        """
        Parameters
        ----------

        mol: Chem.ROMol
            molecule to be parameterized.

        """
        return oe_assign_charges(mol, "AM1")


class AM1BCCHandler(SerializableMixIn):
    def __init__(self, smirks, params, props):
        assert len(smirks) == 0
        assert len(params) == 0
        assert props is None
        self.smirks = []
        self.params = []
        self.props = None

    def partial_parameterize(self, mol):
        return self.static_parameterize(mol)

    def parameterize(self, mol):
        return self.static_parameterize(mol)

    @staticmethod
    def static_parameterize(mol):
        """
        Parameters
        ----------

        mol: Chem.ROMol
            molecule to be parameterized.

        """
        return oe_assign_charges(mol, "AM1BCCELF10")


class AM1CCCHandler(SerializableMixIn):
    def __init__(self, smirks, params, props):
        """
        Parameters
        ----------
        smirks: list of str (P,)
            SMIRKS patterns

        params: np.array, (P,)
            normalized charge for each

        props: any

        """

        assert len(smirks) == len(params)

        self.smirks = smirks
        self.params = np.array(params, dtype=np.float64)
        self.props = props

    def partial_parameterize(self, params, mol):
        return self.static_parameterize(params, self.smirks, mol)

    def parameterize(self, mol):
        return self.static_parameterize(self.params, self.smirks, mol)

    @staticmethod
    def static_parameterize(params, smirks, mol):
        """
        Parameters
        ----------
        params: np.array, (P,)
            normalized charge increment for each matched bond
        smirks: list of str (P,)
            SMIRKS patterns matching bonds
        mol: Chem.ROMol
            molecule to be parameterized.

        """
        am1_charges = compute_or_load_am1_charges(mol)
        bond_idxs, type_idxs = bond_smirks_matches(mol, smirks)

        deltas = params[type_idxs]
        q_params = apply_bond_charge_corrections(am1_charges, bond_idxs, deltas)

        assert q_params.shape[0] == mol.GetNumAtoms()  # check that return shape is consistent with input mol

        return q_params
