import base64
import os
import pickle
import subprocess
import tempfile
import warnings
from collections import Counter, defaultdict
from functools import partial
from shutil import which

import jax.numpy as jnp
import networkx as nx
import numpy as np
from jax import jit, vmap
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import NumRadicalElectrons
from rdkit.Chem.rdMolTransforms import GetBondLength

from timemachine import constants
from timemachine.ff.handlers.bcc_aromaticity import AromaticityModel
from timemachine.ff.handlers.bcc_aromaticity import match_smirks as oe_match_smirks
from timemachine.ff.handlers.serialize import SerializableMixIn
from timemachine.ff.handlers.utils import (
    canonicalize_bond,
    get_query_mol,
    make_residue_mol,
    make_residue_mol_from_template,
    update_mol_topology,
    match_smirks as rd_match_smirks,
)
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
RESP = "RESP"


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
        print(f"Generated {oemol.NumConfs()} OpenEye conformers")

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


def rdkit_generate_conformations(mol):
    ri = mol.GetRingInfo()
    largest_ring_size = max((len(r) for r in ri.AtomRings()), default=0)
    if largest_ring_size > 10:
        print("Detected macrocycle")
        params = Chem.rdDistGeom.ETKDGv3()
    else:
        params = Chem.rdDistGeom.srETKDGv3()
        params.useSmallRingTorsions = True

    params.pruneRmsThresh = 1.0
    params.clearConfs = True

    AllChem.EmbedMultipleConfs(mol, 800, params)
    AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)


def make_xyz(rdmol: Chem.Mol) -> str:
    xyz = []
    for atom in rdmol.GetAtoms():
        atom_index = atom.GetIdx()
        pos = rdmol.GetConformer().GetAtomPosition(atom_index)
        xyz.append(f"{atom.GetSymbol()} {pos.x} {pos.y} {pos.z}")
    return "\n".join(xyz)


def resp_assign_partial_charges(_rdmol: Chem.Mol, use_conformers: list) -> tuple[np.ndarray, float]:
    """
    Calculate RESP (Restrained ElectroStatic Potential) partial charges for a molecule.

    This function performs quantum mechanical calculations to derive atomic partial charges
    that best reproduce the molecular electrostatic potential while applying restraints
    to prevent overfitting.

    Parameters
    ----------
    _rdmol : Chem.Mol
        Input RDKit molecule object
    use_conformers : list[Quantity]
        List of conformer coordinates to use (only the first conformer is used)

    Returns
    -------
    tuple[np.ndarray, float]
        - Array of RESP partial charges for each atom
        - Total DFT energy of the molecule
    """
    from Auto3D.ASE.geometry import opt_geometry
    from gpu4pyscf.pop import esp
    from pyscf import gto, scf
    from pyscf.data import radii

    # Check that antechamber is available for symmetry checking
    ANTECHAMBER_PATH = which("antechamber")
    if ANTECHAMBER_PATH is None:
        raise ValueError("Antechamber not found, cannot run assign_partial_charges()")

    # Create a copy of the molecule and set conformer positions
    rdmol = Chem.Mol(_rdmol)
    rdmol.GetConformer().SetPositions(np.array(use_conformers[0], dtype=np.float64))

    # Compute charges
    with tempfile.TemporaryDirectory() as tmpdir:
        # Get the formal charge of the molecule for QM calculations
        net_charge = Chem.GetFormalCharge(rdmol)

        # Write molecule to SDF file for antechamber processing
        print(Chem.MolToMolBlock(rdmol), file=open(os.path.join(tmpdir, "molecule.sdf"), "w+"))

        # Run antechamber to generate atom types and connectivity information
        # This creates a .ac file with atom type assignments needed for RESP
        subprocess.check_output(
            [
                "antechamber",
                "-i",
                "molecule.sdf",
                "-o",
                "charged.ac",
                "-fo",
                "ac",
                "-fi",
                "sdf",
                "-nc",
                str(net_charge),
            ],
            cwd=tmpdir,
        )

        # Attempt to optimize geometry using AIMNET neural network potential
        # This can provide better initial coordinates for QM calculations
        try:
            out_path = opt_geometry(os.path.join(tmpdir, "molecule.sdf"), model_name="AIMNET")
            m = Chem.MolFromMolFile(out_path, removeHs=False)
            bad_bond = False
            # Check for unreasonable bond lengths that indicate optimization failure
            for bnd in m.GetBonds():
                bond_length = GetBondLength(m.GetConformer(), bnd.GetBeginAtomIdx(), bnd.GetEndAtomIdx())
                if bond_length > 3.0:
                    print(
                        f"Bond length {bond_length} is too high between {bnd.GetBeginAtom().GetSymbol()} and {bnd.GetEndAtom().GetSymbol()}"
                    )
                    bad_bond = True

            if not bad_bond:
                # Use optimized coordinates if geometry is reasonable
                xyz = make_xyz(m)
            else:
                # Fall back to original coordinates if optimization failed
                xyz = make_xyz(rdmol)
        except Exception as e:
            # If optimization fails entirely, use original coordinates
            print(e)
            xyz = make_xyz(rdmol)

        # Generate RESP input file with symmetry constraints
        # respgen identifies chemically equivalent atoms that should have equal charges
        subprocess.check_output(["respgen", "-i", "charged.ac", "-o", "tmp.respin", "-f", "resp"], cwd=tmpdir)

        # Parse symmetry constraints from respgen output
        # These ensure chemically equivalent atoms (e.g., methyl hydrogens) get identical charges
        symmetry_groups = defaultdict(set)
        with open(os.path.join(tmpdir, "tmp.respin")) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "&end" in line:
                    # Parse atom symmetry groups starting 4 lines after "&end"
                    for atm_idx, l in enumerate(lines[i + 4 :], 1):
                        parts = l.split()
                        if parts:
                            group = int(parts[1])
                            if group:
                                # Add atoms to their symmetry group
                                symmetry_groups[group].add(atm_idx - 1)
                                symmetry_groups[group].add(group - 1)

        # Set up PySCF molecule object for quantum calculations
        mol = gto.Mole()
        mol.cart = True  # Use Cartesian basis functions
        mol.charge = Chem.GetFormalCharge(rdmol)
        mol.spin = NumRadicalElectrons(rdmol)  # Number of unpaired electrons
        mol.atom = xyz  # Atomic coordinates
        mol.basis = "6-31gs"  # Standard basis set for RESP calculations
        mol.build()

        # Perform Restricted Hartree-Fock calculation on GPU
        mf = scf.RHF(mol).to_gpu()
        e_dft = mf.kernel()  # Total electronic energy
        dm = mf.make_rdm1()  # Density matrix

        # Define van der Waals radii for ESP grid point generation
        # These determine where electrostatic potential points are sampled
        # Taken from https://github.com/pyscf/gpu4pyscf/blob/57cf1d437adb820ce7f69f8872f2500c751bdd97/gpu4pyscf/pop/esp.py#L32
        # with an added parameter for Br
        rad = (
            1.0
            / radii.BOHR
            * np.asarray(
                [
                    -1,
                    1.20,  # H
                    1.20,  # He
                    1.37,  # Li
                    1.45,  # Be
                    1.45,  # B
                    1.50,  # C
                    1.50,  # N,
                    1.40,  # O
                    1.35,  # F,
                    1.30,  # Ne,
                    1.57,  # Na,
                    1.36,  # Mg
                    1.24,  # Al,
                    1.17,  # Si,
                    1.80,  # P,
                    1.75,  # S,
                    1.70,  # Cl
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    1.85,  # Br
                ]
            )
        )

        # First fit ESP charges without restraints
        # This provides initial charges that exactly reproduce the electrostatic potential
        esp.esp_solve(mol, dm, rad=rad)
        print("Fitted ESP charge")

        # First stage RESP fitting with default restraints
        # This applies weak restraints to prevent overfitting
        esp.resp_solve(mol, dm, rad=rad)
        sum_constraints = []

        # Second stage RESP fitting with symmetry constraints
        # This ensures chemically equivalent atoms have identical charges
        rows = esp.resp_solve(
            mol,
            dm,
            rad=rad,
            resp_a=1e-3,  # Restraint strength parameter
            sum_constraints=sum_constraints,
            equal_constraints=[list(s) for s in symmetry_groups.values()],  # Symmetry constraints
        ).tolist()

    return rows, e_dft


def resp_assign_elf_charges(_rdmol):
    """
    Calculate RESP charges using ELF (Electrostatically Least-interacting Functional) conformer selection.

    This function generates multiple conformers, selects the most diverse ones using ELF,
    computes RESP charges for each conformer, and returns the averaged charges.

    Parameters
    ----------
    _rdmol : Chem.Mol
        Input RDKit molecule object

    Returns
    -------
    np.array
        Array of averaged RESP partial charges scaled by sqrt(ONE_4PI_EPS0)
    """
    # Create a copy of the molecule for conformer generation

    from openff.recharge.utilities.molecule import extract_conformers
    from openff.toolkit import RDKitToolkitWrapper, unit
    from openff.toolkit.topology import Molecule

    rdmol = Chem.Mol(_rdmol)
    rdkit_generate_conformations(rdmol)

    print(f"Generated {rdmol.GetNumConformers()} RDKit conformers")

    # Convert to OpenFF Molecule for ELF conformer selection
    molecule: Molecule = Molecule.from_rdkit(rdmol)
    # Apply ELF conformer selection to get diverse, representative conformers
    # This helps ensure charges are computed from a representative ensemble
    molecule.apply_elf_conformer_selection(
        limit=10, toolkit_registry=RDKitToolkitWrapper(), rms_tolerance=1.0 * unit.angstrom
    )

    print(f"Selected {len(molecule.conformers)} RDKit conformers")

    # Extract conformer coordinates for RESP calculations
    conformers = extract_conformers(molecule)

    # Calculate RESP charges for each selected conformer
    charges = []
    energies = []
    for chs, energy in map(partial(resp_assign_partial_charges, _rdmol), [[c] for c in conformers]):
        if chs is not None:
            charges.append(chs)
            energies.append(energy)

    # Convert charges list to numpy array for averaging
    am1_partial_charges = np.array(charges)

    # Average charges across all conformers
    # This provides more robust charges that account for conformational flexibility
    partial_charges = np.mean(am1_partial_charges, axis=0)

    # Ensure total charge equals the formal charge of the molecule
    # Small numerical errors can accumulate, so we redistribute any discrepancy
    expected_charge = Chem.GetFormalCharge(rdmol)
    current_charge = 0.0
    for pc in partial_charges:
        current_charge += pc
    charge_offset = (expected_charge - current_charge) / rdmol.GetNumAtoms()
    partial_charges += charge_offset

    # Verify that the charges sum up to an integer
    net_charge = np.sum(partial_charges)
    net_charge_is_integral = np.isclose(net_charge, np.round(net_charge), atol=1e-5)
    assert net_charge_is_integral, f"Charge is not an integer: {net_charge}"

    # Apply scaling factor for TimeMachine's optimized charge representation
    # https://github.com/proteneer/timemachine#forcefield-gotchas
    # "The charges have been multiplied by sqrt(ONE_4PI_EPS0) as an optimization."
    inlined_constant = np.sqrt(constants.ONE_4PI_EPS0)

    # returned charges are in TM units, in original atom ordering
    return inlined_constant * partial_charges


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


def compute_or_load_resp_charges(mol):
    """
    Unless already cached in mol's "{mode}{CACHE_SUFFIX}" property,
    use RESP to compute partial charges.
    """

    mode = "resp"
    # check for cache
    cache_prop_name = f"{mode}{CACHE_SUFFIX}"
    if not mol.HasProp(cache_prop_name):
        resp_charges = list(resp_assign_elf_charges(mol))
        mol.SetProp(cache_prop_name, base64.b64encode(pickle.dumps(resp_charges)))
    else:
        resp_charges = pickle.loads(base64.b64decode(mol.GetProp(cache_prop_name)))
        assert len(resp_charges) == mol.GetNumAtoms(), "Charge cache has different number of charges than mol atoms"

    return np.array(resp_charges)


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


@jit
def eval_nn(features, params_by_layer):
    def activation(x):
        return x / (1 + jnp.exp(-x))  # silu

    layers_by_index = list(sorted(params_by_layer.keys()))

    x = features
    for layer in layers_by_index[:-1]:
        W = params_by_layer[layer]
        x = activation(jnp.dot(W, x))

    # last layer skips activation
    W = params_by_layer[layers_by_index[-1]]
    return jnp.squeeze(jnp.dot(W, x))  # scalar


class NNHandler(SerializableMixIn):
    def __init__(self, layer_sizes, params, props):
        assert len(layer_sizes) == 1
        assert len(params) == 1
        # TODO: Make SerializableMixIn generic w.r.t. attribute names
        self.smirks = layer_sizes
        self.params = np.array(params, dtype=np.float64)
        self.props = props

    @staticmethod
    def get_bond_idxs_and_charge_deltas(flat_params, encoded_unflatten_str, mol):
        expand_params = pickle.loads(base64.b64decode(encoded_unflatten_str[0]))
        features = pickle.loads(base64.b64decode(mol.GetProp(NN_FEATURES_PROPNAME)))
        atom_features = features["atom_features"]
        bond_idx_features = features["bond_idxs"]
        bond_src_features = features["bond_src_features"]
        bond_dst_features = features["bond_dst_features"]

        # extract bond features
        bond_features_by_idx = {}
        for i, bond_idx in enumerate(bond_idx_features):
            bond_feature = np.concatenate([bond_src_features[i], bond_dst_features[i]])
            bond_features_by_idx[tuple(bond_idx)] = bond_feature
        bond_idxs = np.array(sorted(set(bond_features_by_idx.keys())))

        # expand params
        reshaped_params = expand_params(flat_params[0])
        layer_idxs = list(range(len(reshaped_params)))
        params_by_layer = {int(layer_idx): param for layer_idx, param in zip(layer_idxs, reshaped_params)}

        # evalute on all bonds
        features_ = []
        for bond_idx in bond_idxs:
            bond_idx_tup = tuple(bond_idx)
            a0 = atom_features[bond_idx[0]]
            a1 = atom_features[bond_idx[1]]
            b0 = bond_features_by_idx[bond_idx_tup]
            features_.append(np.array(np.concatenate([a0, a1, b0])))
        batched_features = jnp.array(features_)
        c = np.sqrt(constants.ONE_4PI_EPS0)
        vmap_fxn = vmap(eval_nn, in_axes=(0, None))
        deltas = c * vmap_fxn(batched_features, params_by_layer)
        return bond_idxs, jnp.array(deltas)

    @staticmethod
    def static_parameterize(flat_params, encoded_unflatten_str, mol):
        am1_charges = compute_or_load_oe_charges(mol, mode=AM1BCCELF10)
        bond_idxs, deltas = NNHandler.get_bond_idxs_and_charge_deltas(flat_params, encoded_unflatten_str, mol)
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


class RESPHandler(SerializableMixIn):
    """The RESPHandler generates charges for molecules using RESP."""

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
        return compute_or_load_resp_charges(mol)


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
        cur_atom = 0
        for tfr in self.template_for_residue:
            n_atoms = len(tfr.atoms)
            if tfr.name in self.all_res_mols_by_name:
                # already processed this residue
                cur_atom += n_atoms
                continue

            symbol_list = [atm.element.symbol for atm in tfr.atoms]
            bond_list = tfr.bonds
            topology_res_mol = make_residue_mol(tfr.name, symbol_list, bond_list)
            template_res_mol = make_residue_mol_from_template(tfr.name)
            if template_res_mol is None:
                # i.e. skip water/ions
                cur_atom += n_atoms
                continue

            # copy the charges and bond types from template_res to topology_res_mol
            update_mol_topology(topology_res_mol, template_res_mol)

            # cache smirks patterns to speed up parameterize,
            initial_res_charges = self.initial_charges[cur_atom : cur_atom + n_atoms]
            self._compute_res_charges(tfr.name, topology_res_mol, initial_res_charges, params)

            self.all_res_mols_by_name[tfr.name] = topology_res_mol
            cur_atom += n_atoms

    def _map_to_topology_order(self, template_charges, cur_atom, n_atoms):
        # map back from the template ordering to the topology order
        q_params = {}
        for i in range(n_atoms):
            tmpl_atom_idx = self.topology_idx_to_template_idx[cur_atom + i]
            q_params[i] = template_charges[tmpl_atom_idx]
        return jnp.array([q_params[i] for i in range(n_atoms)])

    def _compute_res_charges(self, res_name, topology_res_mol, initial_res_charges, params):
        bond_idxs, type_idxs = compute_or_load_bond_smirks_matches(topology_res_mol, self.patterns)
        deltas = params[type_idxs]
        return apply_bond_charge_corrections(
            initial_res_charges,
            bond_idxs,
            deltas,
            runtime_validate=False,  # required for jit
        )

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
            topology_res_mol_charges_ordered = np.array([topology_res_mol_charges[i] for i in range(n_atoms)])

            # compute the charges on the topology res
            tmpl_q_params = self._compute_res_charges(
                tfr.name, topology_res_mol, topology_res_mol_charges_ordered, params
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


class EnvironmentNNHandler(EnvironmentBCCHandler):
    """
    Applies `NNHandler` to residues in a protein. Needs a concrete openmm topology to use.
    NOTE: Currently, this only supports the amber99sbildn protein forcefield.
    """

    def __init__(self, patterns, params, props, protein_ff_name, water_ff_name, topology):
        self.props = props
        self.nn_h = NNHandler(patterns, params, None)
        super().__init__(patterns, params, protein_ff_name, water_ff_name, topology)

    def _compute_res_charges(self, res_name, topology_res_mol, initial_res_charges, params):
        features_by_res = pickle.loads(base64.b64decode(self.props[0]))

        # NOTE: This is NOT the same template as the OMM template in `EnvironmentBCCHandler`
        template_res_mol = make_residue_mol_from_template(res_name)

        # features are already encoded
        template_res_mol.SetProp(NN_FEATURES_PROPNAME, features_by_res[res_name])

        # map bond_idxs back to topology_res_mol
        match = template_res_mol.GetSubstructMatch(get_query_mol(topology_res_mol))

        # Match maps the topology_res_mol to template_res_mol
        fwd_map = {i: v for i, v in enumerate(match)}

        # map from template_res_mol to topology_res_mol
        rev_map = {v: i for i, v in fwd_map.items()}

        # get the bond_idxs, deltas for the template residue
        bond_idxs, deltas = self.nn_h.get_bond_idxs_and_charge_deltas(params, self.patterns, template_res_mol)

        # map back to the topology residue
        top_bond_idxs = []
        top_deltas = []
        for bond_idx, delta in zip(bond_idxs, deltas):
            src_idx, dst_idx = bond_idx
            # May have a bond that is not present in the original topology
            # so skip these in the BCC
            if src_idx in rev_map and dst_idx in rev_map:
                top_src_idx = rev_map[src_idx]
                top_dst_idx = rev_map[dst_idx]
                top_bond_idxs.append((top_src_idx, top_dst_idx))
                top_deltas.append(delta)

        final_charges = apply_bond_charge_corrections(
            initial_res_charges, np.array(top_bond_idxs), jnp.array(top_deltas), runtime_validate=False
        )
        return final_charges


class EnvironmentNNPartialHandler(EnvironmentBCCPartialHandler):
    """
    Similar to `EnvironmentBCCPartialHandler` but using the NNHandler
    in place of BCC terms.
    """

    def get_env_handle(self, omm_topology, ff) -> EnvironmentNNHandler:
        """
        Return an initialized `EnvironmentNNHandler` which can be used to
        get the (possibly updated) environment charges.

        Parameters
        ----------
        omm_topology:
            Openmm topology object for the environment.

        ff: Forcefield
        """
        return EnvironmentNNHandler(self.smirks, self.params, self.props, ff.protein_ff, ff.water_ff, omm_topology)


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
