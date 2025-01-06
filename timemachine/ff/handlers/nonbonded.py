import base64
import pickle
import warnings
from collections import Counter

import jax.numpy as jnp
import networkx as nx
import numpy as np
from numpy.typing import NDArray
from rdkit import Chem

from timemachine import constants
from timemachine.ff.handlers.bcc_aromaticity import AromaticityModel
from timemachine.ff.handlers.bcc_aromaticity import match_smirks as oe_match_smirks
from timemachine.ff.handlers.serialize import SerializableMixIn
from timemachine.ff.handlers.utils import canonicalize_bond, make_residue_mol, make_residue_mol_from_template
from timemachine.ff.handlers.utils import match_smirks as rd_match_smirks
from timemachine.ff.handlers.utils import update_mol_topology
from timemachine.graph_utils import convert_to_nx

CACHE_SUFFIX = "Cache"
AM1_CHARGE_CACHE = "AM1Cache"
AM1BCC_CHARGE_CACHE = "AM1BCCCache"
AM1ELF10_CHARGE_CACHE = "AM1ELF10Cache"
AM1BCCELF10_CHARGE_CACHE = "AM1BCCELF10Cache"
BOND_SMIRK_MATCH_CACHE = "BondSmirkMatchCache"
NN_FEATURES_PROPNAME = "NNFeatures"

AM1 = "AM1"
AM1ELF10 = "AM1ELF10"
AM1BCC = "AM1BCC"
AM1BCCELF10 = "AM1BCCELF10"
ELF10_MODELS = (AM1ELF10, AM1BCCELF10)


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
    ims.close()
    return oemol


def oe_generate_conformations(oemol, sample_hydrogens=True):
    """Generate conformations for the input molecule.
    The molecule is modified in place.

    Note: This currently does not filter out trans carboxylic acids.
    See https://github.com/openforcefield/openff-toolkit/pull/1171

    Note: This may permute the molecule in-place during canonicalization.
        (If the original atom ordering needs to be recovered, modify calling context using {Set/Get}MapIdx.)

    Parameters
    ----------
    oemol: oechem.OEMol
    sample_hydrogens: bool

    References
    ----------
    [1] https://docs.eyesopen.com/toolkits/cookbook/python/modeling/am1-bcc.html
    """

    from openeye import oeomega

    # generate conformations using omega
    omegaOpts = oeomega.OEOmegaOptions()
    omegaOpts.GetTorDriveOptions().SetUseGPU(False)
    omega = oeomega.OEOmega(omegaOpts)
    # exclude the initial input conformer
    omega.SetIncludeInput(False)
    omega.SetCanonOrder(True)  # may not preserve input atom ordering
    omega.SetSampleHydrogens(sample_hydrogens)
    omega.SetEnergyWindow(15.0)
    omega.SetMaxConfs(800)
    omega.SetRMSThreshold(1.0)

    has_confs = omega(oemol)
    if not has_confs:
        raise Exception(f"Unable to generate conformations for charge assignment for '{oemol.GetTitle()}'")


def oe_assign_charges(mol, charge_model=AM1BCCELF10):
    """assign partial charges, then premultiply by sqrt(ONE_4PI_EPS0)
    as an optimization"""

    oemol = convert_to_oe(mol)

    # this is to recover the ordering of atoms from input mol (current state of oemol),
    # since canonicalization in oe_generate_conformations(oemol) may modify oemol's atom order in-place
    for i, atom in enumerate(oemol.GetAtoms()):
        atom.SetMapIdx(i + 1)

    # imported here for optional dependency
    from openeye import oequacpac

    charge_engines = {
        AM1: oequacpac.OEAM1Charges(symmetrize=True),
        AM1ELF10: oequacpac.OEELFCharges(oequacpac.OEAM1Charges(symmetrize=True), 10),
        AM1BCC: oequacpac.OEAM1BCCCharges(symmetrize=True),
        AM1BCCELF10: oequacpac.OEAM1BCCELF10Charges(),
    }
    charge_engine = charge_engines[charge_model]

    if charge_model in ELF10_MODELS:
        oe_generate_conformations(oemol)

    result = oequacpac.OEAssignCharges(oemol, charge_engine)
    if result is False:
        # Turn off hydrogen sampling if charge generation fails
        # https://github.com/openforcefield/openff-toolkit/issues/346#issuecomment-505202862
        if charge_model in ELF10_MODELS:
            print(f"WARNING: Turning off hydrogen sampling for charge generation on molecule '{oemol.GetTitle()}'")
            oemol = convert_to_oe(mol)
            oe_generate_conformations(oemol, sample_hydrogens=False)
            result = oequacpac.OEAssignCharges(oemol, charge_engine)
        if result is False:
            raise Exception(f"Unable to assign charges for '{oemol.GetTitle()}'")

    partial_charges = np.array([atom.GetPartialCharge() for atom in oemol.GetAtoms()])

    # Verify that the charges sum up to an integer
    net_charge = np.sum(partial_charges)
    net_charge_is_integral = np.isclose(net_charge, np.round(net_charge), atol=1e-5)
    assert net_charge_is_integral, f"Charge is not an integer: {net_charge}"

    # https://github.com/proteneer/timemachine#forcefield-gotchas
    # "The charges have been multiplied by sqrt(ONE_4PI_EPS0) as an optimization."
    inlined_constant = np.sqrt(constants.ONE_4PI_EPS0)

    # recover original atom order
    inv_permutation = np.argsort([(atom.GetMapIdx() - 1) for atom in oemol.GetAtoms()])

    # returned charges are in TM units, in original atom ordering
    return inlined_constant * partial_charges[inv_permutation]


def generate_exclusion_idxs(
    mol: Chem.Mol, scale12: float, scale13: float, scale14_lj: float, scale14_q: float
) -> tuple[NDArray, NDArray]:
    """
    Generate exclusions for a mol based on the all pairs shortest path.
    We always take the convention that exclusions for smaller distances
    override those of longer distances.

    Parameters
    ----------
    mol: Chem.Mol
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
    scales: float, 2 array
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
                    scale = (scale12, scale12)
                elif length == 2:
                    scale = (scale13, scale13)
                elif length == 3:
                    scale = (scale14_q, scale14_lj)
                else:
                    assert 0

                exclusions[canonicalize_bond((src, dst))] = scale

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


def compute_or_load_oe_charges(mol, mode=AM1ELF10):
    """
    Unless already cached in mol's "{mode}{CACHE_SUFFIX}" property,
    use OpenEye to compute partial charges using the specified mode.

    Parameters
    ----------
    mode:
        One of AM1, AM1ELF10, AM1BCC or AM1BCCELF10.
    """
    assert mode in [AM1, AM1ELF10, AM1BCC, AM1BCCELF10]

    # check for cache
    cache_prop_name = f"{mode}{CACHE_SUFFIX}"
    if not mol.HasProp(cache_prop_name):
        # The charges returned by OEQuacPac is not deterministic across OS platforms. It is known
        # to be an issue that the atom ordering modifies the return values as well. A follow up
        # with OpenEye is in order
        # https://github.com/openforcefield/openff-toolkit/issues/983
        oe_charges = list(oe_assign_charges(mol, mode))

        mol.SetProp(cache_prop_name, base64.b64encode(pickle.dumps(oe_charges)))

    else:
        oe_charges = pickle.loads(base64.b64decode(mol.GetProp(cache_prop_name)))
        assert len(oe_charges) == mol.GetNumAtoms(), "Charge cache has different number of charges than mol atoms"

    return np.array(oe_charges)


def compute_or_load_bond_smirks_matches(mol, smirks_list):
    """Unless already cached in mol's "BondSmirkMatchCache" property, uses OpenEye to compute arrays of ordered bonds and their assigned types.

    Notes
    -----
    * Uses OpenEye for substructure searches
    * Order within smirks_list matters
        "First match wins."
        For example, if bond (a,b) can be matched by smirks_list[2], smirks_list[5], ..., assign type 2
    * Order within each smirks pattern matters
        For example, "[#6:1]~[#1:2]" and "[#1:1]~[#6:2]" will match atom pairs in the opposite order
    """
    if not mol.HasProp(BOND_SMIRK_MATCH_CACHE):
        oemol = convert_to_oe(mol)
        AromaticityModel.assign(oemol)

        bond_idxs = []  # [B, 2]
        type_idxs = []  # [B]

        for type_idx, smirks in enumerate(smirks_list):
            matches = oe_match_smirks(smirks, oemol)

            for matched_indices in matches:
                a, b = matched_indices[0], matched_indices[1]
                forward_matched_bond = [a, b]

                already_assigned = forward_matched_bond in bond_idxs

                if not already_assigned:
                    bond_idxs.append(forward_matched_bond)
                    type_idxs.append(type_idx)
        mol.SetProp(BOND_SMIRK_MATCH_CACHE, base64.b64encode(pickle.dumps((bond_idxs, type_idxs))))
    else:
        bond_idxs, type_idxs = pickle.loads(base64.b64decode(mol.GetProp(BOND_SMIRK_MATCH_CACHE)))
    return np.array(bond_idxs), np.array(type_idxs)


def apply_bond_charge_corrections(initial_charges, bond_idxs, deltas, runtime_validate=True):
    """For an arbitrary collection of ordered bonds and associated increments `(a, b, delta)`,
    update `charges` by `charges[a] += delta`, `charges[b] -= delta`.

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
    incremented = jnp.asarray(initial_charges).at[bond_idxs[:, 0]].add(+deltas)
    decremented = jnp.asarray(incremented).at[bond_idxs[:, 1]].add(-deltas)
    final_charges = decremented

    # make some safety assertions
    assert bond_idxs.shape[1] == 2
    assert len(deltas) == len(bond_idxs)

    net_charge = jnp.sum(initial_charges)
    final_net_charge = jnp.sum(final_charges)
    net_charge_is_unchanged = jnp.isclose(final_net_charge, net_charge, atol=1e-5)

    if runtime_validate:
        assert net_charge_is_unchanged

    # print some safety warnings
    directed_bonds = Counter([tuple(b) for b in bond_idxs])

    if max(directed_bonds.values()) > 1:
        duplicates = [bond for (bond, count) in directed_bonds.items() if count > 1]
        warnings.warn(f"Duplicate directed bonds! {duplicates}")

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
            SMIRKS patterns, to be parsed by RDKIT

        mol: Chem.ROMol
            rdkit molecule, should have hydrogens pre-added

        """
        assert len(smirks) == len(params)
        param_idxs = generate_nonbonded_idxs(mol, smirks)
        return params[param_idxs]


class PrecomputedChargeHandler(SerializableMixIn):
    def __init__(self, smirks, params, props):
        assert len(smirks) == 0
        assert len(params) == 0
        assert props is None
        # These fields as to enable serialization
        self.smirks = []
        self.params = []
        self.props = None

    def parameterize(self, mol):
        params = []
        for atom in mol.GetAtoms():
            q = float(atom.GetProp("PartialCharge"))
            params.append(q * np.sqrt(constants.ONE_4PI_EPS0))
        return np.array(params)

    def partial_parameterize(self, _, mol):
        return self.parameterize(mol)


class PrecomputedChargeIntraHandler(PrecomputedChargeHandler):
    pass


class SimpleChargeHandler(NonbondedHandler):
    pass


class SimpleChargeIntraHandler(SimpleChargeHandler):
    pass


class SimpleChargeSolventHandler(SimpleChargeHandler):
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
            SMIRKS patterns, to be parsed by RDKIT

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


class LennardJonesIntraHandler(LennardJonesHandler):
    pass


class LennardJonesSolventHandler(LennardJonesHandler):
    pass


class GBSAHandler(NonbondedHandler):
    pass


class AM1Handler(SerializableMixIn):
    """The AM1Handler generates charges for molecules using OpenEye's AM1[1] protocol.

    Charges are conformer and platform dependent as of OpenEye Toolkits 2020.2.0 [2].

    References
    ----------
    [1] AM1 Theory
        https://docs.eyesopen.com/toolkits/python/quacpactk/molchargetheory.html#am1-charges
    [2] Charging Inconsistencies
        https://github.com/openforcefield/openff-toolkit/issues/1170
    """

    def __init__(self, smirks, params, props):
        assert len(smirks) == 0
        assert len(params) == 0
        assert props is None

    def partial_parameterize(self, _, mol):
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
        return compute_or_load_oe_charges(mol, mode=AM1)


class NNHandler(SerializableMixIn):
    def __init__(self, layer_idxs, params, layer_sizes):
        self.smirks = layer_idxs
        self.params = params
        self.props = layer_sizes

    @staticmethod
    def static_parameterize(layer_idxs, params, layer_sizes, mol):
        features = pickle.loads(base64.b64decode(mol.GetProp(NN_FEATURES_PROPNAME)))
        atom_features = features["atom_features"]
        bond_idx_features = features["bond_idxs"]
        bond_src_features = features["bond_src_features"]
        bond_dst_features = features["bond_dst_features"]

        bond_features_by_idx = {}
        for i, bond_idx in enumerate(bond_idx_features):
            bond_features_by_idx[tuple(bond_idx)] = jnp.concatenate([bond_src_features[i], bond_dst_features[i]])

        am1_charges = compute_or_load_oe_charges(mol, mode=AM1BCC)
        bond_idxs = np.array(sorted(set(bond_features_by_idx.keys())))

        reshaped_params = []
        for layer_size, p in zip(layer_sizes, params):
            reshaped_params.append(jnp.array(p).reshape(-1, layer_size))

        params_by_layer = {int(layer_idx): param for layer_idx, param in zip(layer_idxs, reshaped_params)}
        layers_by_index = list(sorted(params_by_layer.keys()))

        def activation(x):
            return x / (1 + jnp.exp(-x))  # silu

        def eval_nn(features):
            x = features
            for layer in layers_by_index[:-1]:
                W = params_by_layer[layer]
                x = activation(jnp.dot(W, x))

            # last layer skips activation
            W = params_by_layer[layers_by_index[-1]]
            return jnp.squeeze(jnp.dot(W, x))  # scalar

        deltas = []
        for bond_idx in bond_idxs:
            bond_idx_tup = tuple(bond_idx)
            a0 = atom_features[bond_idx[0]]
            a1 = atom_features[bond_idx[1]]
            b0 = bond_features_by_idx[bond_idx_tup]
            # just b0 with src, dst swapped
            b1 = bond_features_by_idx[bond_idx_tup[::-1]]
            ff0 = jnp.concatenate([a0, a1, b0])
            ff1 = jnp.concatenate([a1, a0, b1])
            f0 = eval_nn(ff0)
            f1 = eval_nn(ff1)
            f = f1 - f0  # antisymmetric to swapping bond direction
            deltas.append(f * np.sqrt(constants.ONE_4PI_EPS0))

        final_charges = apply_bond_charge_corrections(am1_charges, bond_idxs, jnp.array(deltas), runtime_validate=False)
        return final_charges


class AM1BCCHandler(SerializableMixIn):
    """The AM1BCCHandler generates charges for molecules using OpenEye's AM1BCCELF10[1] protocol. Note that
    if a single conformer molecular is passed to this handler, the charges appear equivalent with AM1BCC.

    Charges are conformer and platform dependent as of OpenEye Toolkits 2020.2.0 [2].

    References
    ----------
    [1] AM1BCCELF10 Theory
        https://docs.eyesopen.com/toolkits/python/quacpactk/molchargetheory.html#elf-conformer-selection
    [2] Charging Inconsistencies
        https://github.com/openforcefield/openff-toolkit/issues/1170
    """

    def __init__(self, smirks, params, props):
        assert len(smirks) == 0
        assert len(params) == 0
        assert props is None
        self.smirks = []
        self.params = []
        self.props = None

    def partial_parameterize(self, _, mol):
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
        return compute_or_load_oe_charges(mol, mode=AM1BCCELF10)


class AM1BCCIntraHandler(AM1BCCHandler):
    pass


class AM1BCCSolventHandler(AM1BCCHandler):
    pass


class EnvironmentBCCHandler(SerializableMixIn):
    """
    Applies BCCs to residues in a protein. Needs a concrete openmm topology to use.
    NOTE: Currently, this only supports the amber99sbildn protein forcefield.
    """

    def __init__(self, patterns, params, protein_ff_name, water_ff_name, topology):
        # Import here to avoid triggering failures with imports in cases without openmm
        import openmm
        from openmm import app
        from openmm.app.forcefield import ForceField

        from timemachine.ff.handlers import openmm_deserializer

        assert protein_ff_name == "amber99sbildn", f"{protein_ff_name} is not currently supported"
        self.patterns = patterns
        self.params = np.array(params)
        self.env_ff = ForceField(f"{protein_ff_name}.xml", f"{water_ff_name}.xml")

        # reverse engineered from openmm's Forcefield class
        self.topology = topology
        residueTemplates = dict()
        ignoreExternalBonds = False
        self.data = ForceField._SystemData(topology)

        # template_for_residue is list a of _templateData objects,
        # where the index is the residue index and the value is a template type, which
        # may be different from the standard residue type in the PDB file itself, eg:
        # a standard HIS tag in the input PDB is processed into the specific template type:
        # {HID,HIE,HIP}
        self.template_for_residue = self.env_ff._matchAllResiduesToTemplates(
            self.data, topology, residueTemplates, ignoreExternalBonds
        )

        env_system = self.env_ff.createSystem(
            topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False
        )

        self.initial_charges = None
        for force in env_system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                nb_params, _, _, _ = openmm_deserializer.deserialize_nonbonded_force(
                    force, env_system.getNumParticles()
                )
                self.initial_charges = nb_params[:, 0]  # already scaled by sqrt(ONE_4PI_EPS0)
        assert self.initial_charges is not None

        # initial_charges is in the topology ordering
        # so map to the topology_res_mol ordering
        self.topology_idx_to_template_idx = {atm.index: j for atm, j in self.data.atomTemplateIndexes.items()}

        self.all_res_mols_by_name = {}
        for tfr in self.template_for_residue:
            if tfr.name in self.all_res_mols_by_name:
                # already processed this residue
                continue

            symbol_list = [atm.element.symbol for atm in tfr.atoms]
            bond_list = tfr.bonds
            topology_res_mol = make_residue_mol(tfr.name, symbol_list, bond_list)
            template_res_mol = make_residue_mol_from_template(tfr.name)
            if template_res_mol is None:
                # i.e. skip water/ions
                continue

            # copy the charges and bond types from proper_res to omm_res
            update_mol_topology(topology_res_mol, template_res_mol)

            # cache smirks patterns to speed up parameterize
            compute_or_load_bond_smirks_matches(topology_res_mol, self.patterns)
            self.all_res_mols_by_name[tfr.name] = topology_res_mol

    def _map_to_topology_order(self, template_charges, cur_atom, n_atoms):
        # map back from the template ordering to the topology order
        q_params = {}
        for i in range(n_atoms):
            tmpl_atom_idx = self.topology_idx_to_template_idx[cur_atom + i]
            q_params[i] = template_charges[tmpl_atom_idx]
        return jnp.array([q_params[i] for i in range(n_atoms)])

    def parameterize(self, params):
        cur_atom = 0
        final_charges = []
        template_cached_charges = {}
        for tfr in self.template_for_residue:
            n_atoms = len(tfr.atoms)
            initial_res_charges = self.initial_charges[cur_atom : cur_atom + n_atoms]

            # not a template residue, so skip
            if tfr.name not in self.all_res_mols_by_name:
                final_charges.append(initial_res_charges)
                cur_atom += n_atoms
                continue

            # only compute the charges once per residue type
            # and reuse from cache if possible
            if tfr.name in template_cached_charges:
                tmpl_q_params = template_cached_charges[tfr.name]
                final_charges.append(self._map_to_topology_order(tmpl_q_params, cur_atom, n_atoms))
                cur_atom += n_atoms
                continue

            # extract the charges in the order of the template residue
            topology_res_mol = self.all_res_mols_by_name[tfr.name]
            topology_res_mol_charges = {}
            for i in range(n_atoms):
                tmpl_atom_idx = self.topology_idx_to_template_idx[cur_atom + i]
                topology_res_mol_charges[tmpl_atom_idx] = initial_res_charges[i]
            topology_res_mol_charges_ordered = jnp.array([topology_res_mol_charges[i] for i in range(n_atoms)])

            # compute smirks on the template residue
            bond_idxs, type_idxs = compute_or_load_bond_smirks_matches(topology_res_mol, self.patterns)
            deltas = params[type_idxs]
            tmpl_q_params = apply_bond_charge_corrections(
                topology_res_mol_charges_ordered,
                bond_idxs,
                deltas,
                runtime_validate=False,  # required for jit
            )

            # map the template charges back to topology order
            template_cached_charges[tfr.name] = tmpl_q_params
            q_params_ordered = self._map_to_topology_order(tmpl_q_params, cur_atom, n_atoms)
            final_charges.append(q_params_ordered)
            cur_atom += n_atoms

        return jnp.concatenate(final_charges, axis=0)


class EnvironmentBCCPartialHandler(SerializableMixIn):
    """
    This class is used to represent the environment BCC terms
    that modify the charges used for the environment-ligand
    interaction potential.

    The current implementation skips water molecules, so only the
    protein charges are modified.

    This class is a serializable version of `EnvironmentBCCHandler`,
    so it can be stored in the forcefield python file in the normal way.

    NOTE: Currently, this only supports the amber99sbildn protein forcefield.
    """

    def __init__(self, smirks, params, props):
        self.smirks = smirks
        self.params = np.array(params)
        self.props = props

    def get_env_handle(self, omm_topology, ff) -> EnvironmentBCCHandler:
        """
        Return an initialized `EnvironmentBCCHandler` which can be used to
        get the (possibly updated) environment charges.

        Parameters
        ----------
        omm_topology:
            Openmm topology object for the environment.

        ff: Forcefield
        """
        return EnvironmentBCCHandler(self.smirks, self.params, ff.protein_ff, ff.water_ff, omm_topology)


class AM1CCCHandler(SerializableMixIn):
    """The AM1CCCHandler stands for AM1 Correctable Charge Correction (CCC) which uses OpenEye's AM1 charges[1]
    along with corrections provided by the Forcefield definition in the form of SMIRKS and charge deltas. The SMIRKS
    are currently parsed using the OpenEye Toolkits standard[2].

    This handler supports jax.grad with respect to the forcefield parameters, which is what the "Correctable" refers
    to in CCC.

    Charges are conformer and platform dependent as of OpenEye Toolkits 2020.2.0 [3].

    References
    ----------
    [1] AM1 Theory
        https://docs.eyesopen.com/toolkits/python/quacpactk/molchargetheory.html#am1-charges
    [2] OpenEye SMARTS standard
        https://docs.eyesopen.com/toolkits/cpp/oechemtk/SMARTS.html
    [3] Charging Inconsistencies
        https://github.com/openforcefield/openff-toolkit/issues/1170
    """

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
        self.supported_elements = {1, 6, 7, 8, 9, 14, 16, 17, 35, 53}  # note: omits phosphorus (15) for now

    def validate_input(self, mol):
        # TODO: read off supported elements from self.smirks, rather than hard-coding list of supported elements?
        elements = set([a.GetAtomicNum() for a in mol.GetAtoms()])
        if not elements.issubset(self.supported_elements):
            raise RuntimeError("mol contains unsupported elements: ", elements - self.supported_elements)

    def partial_parameterize(self, params, mol):
        self.validate_input(mol)
        return self.static_parameterize(params, self.smirks, mol)

    def parameterize(self, mol):
        return self.partial_parameterize(self.params, mol)

    @staticmethod
    def static_parameterize(params, smirks, mol):
        """
        Parameters
        ----------
        params: np.array, (P,)
            normalized charge increment for each matched bond
        smirks: list of str (P,)
            SMIRKS patterns matching bonds, to be parsed using OpenEye Toolkits
        mol: Chem.ROMol
            molecule to be parameterized.

        """
        return AM1CCCHandler._static_parameterize(params, smirks, mol)

    @staticmethod
    def _static_parameterize(params, smirks, mol, mode=AM1ELF10):
        """
        Parameters
        ----------
        params: np.array, (P,)
            normalized charge increment for each matched bond
        smirks: list of str (P,)
            SMIRKS patterns matching bonds, to be parsed using OpenEye Toolkits
        mol: Chem.ROMol
            molecule to be parameterized.
        mode: str
            Mode used to compute the charges, one of AM1, AM1ELF10, AM1BCC, AM1BCCELF10.
            Defaults to AM1ELF10. Note, if using AM1BCC or AM1BCCELF10, the parameters
            need to be initialized correctly.

        """
        # (ytz): leave this comment here, useful for quickly disable AM1 calculations for large mols
        # return np.zeros(mol.GetNumAtoms())
        am1_charges = compute_or_load_oe_charges(mol, mode=mode)
        bond_idxs, type_idxs = compute_or_load_bond_smirks_matches(mol, smirks)

        deltas = params[type_idxs]
        q_params = apply_bond_charge_corrections(
            am1_charges,
            bond_idxs,
            deltas,
            runtime_validate=False,  # required for jit
        )

        assert q_params.shape[0] == mol.GetNumAtoms()  # check that return shape is consistent with input mol

        return q_params


class AM1CCCIntraHandler(AM1CCCHandler):
    pass


class AM1CCCSolventHandler(AM1CCCHandler):
    pass


class AM1BCCCCCHandler(AM1CCCHandler):
    def __init__(self, smirks, params, props):
        """
        The AM1BCCCCCHandler stands for AM1BCC Correctable Charge Correction (CCC),
        that is AM1BCC is applied using OpenEye's AM1BCCELF10 method and
        then additional corrections are applied to the resulting charges.

        This handler also supports phosphorus, unlike the `AM1CCCHandler`.
        See `AM1CCCHandler` for more details.

        Parameters
        ----------
        smirks: list of str (P,)
            SMIRKS patterns
        params: np.array, (P,)
            normalized charge increment for each matched bond
        props: any
        """
        super().__init__(smirks, params, props)
        # Also supports phosphorus
        self.supported_elements.add(15)

    @staticmethod
    def static_parameterize(params, smirks, mol):
        """
        Parameters
        ----------
        params: np.array, (P,)
            normalized charge increment for each matched bond
        smirks: list of str (P,)
            SMIRKS patterns matching bonds, to be parsed using OpenEye Toolkits
        mol: Chem.ROMol
            molecule to be parameterized.

        """
        return AM1CCCHandler._static_parameterize(params, smirks, mol, mode=AM1BCCELF10)


class AM1BCCCCCIntraHandler(AM1BCCCCCHandler):
    pass


class AM1BCCCCCSolventHandler(AM1BCCCCCHandler):
    pass
