import re

import jax
import numpy as np
from jax import grad
from jax import numpy as jnp
from numpy.typing import NDArray
from openeye import oechem
from rdkit import Chem

from timemachine.ff.handlers.nonbonded import convert_to_oe


def canonicalize_bond(arr):
    """
    Canonicalize a bonded interaction. If arr[0] < arr[-1] then arr is
    returned, else if arr[0] > arr[-1], then arr[::-1] is returned. If
    arr[0] == arr[-1] then an exception is thrown.

    Parameters
    ----------
    arr: list of int
        Bond indices.

    Returns
    -------
    arr: list of int
        Canonicalized bond indices.

    """
    container_type = type(arr)

    if len(arr) == 0:
        raise ValueError("zero sized array")
    elif len(arr) == 1:
        return arr
    elif arr[0] > arr[-1]:
        return container_type(reversed(arr))
    elif arr[0] == arr[-1]:
        raise ValueError("Invalid bond with first and last indices equal")
    else:
        return arr


def match_smirks(mol, smirks):
    """
    Notes
    -----
    * See also implementations of match_smirks in
        * bootstrap_am1.py, which is identical
        * bcc_aromaticity.py, which uses OpenEye instead of RDKit
    """

    # Make a copy of the molecule
    rdmol = Chem.Mol(mol)
    # Use designated aromaticity model
    Chem.SanitizeMol(rdmol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_SETAROMATICITY)
    Chem.SetAromaticity(rdmol, Chem.AromaticityModel.AROMATICITY_MDL)

    # Set up query.
    qmol = Chem.MolFromSmarts(smirks)  # cannot catch the error
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


def check_bond_smarts_symmetric(bond_smarts: str) -> bool:
    """Match [<atom1>:1]*[<atom2>:2]
    and return whether atom1 and atom2 are identical strings

    Notes
    -----
    * The AM1CCC model contains symmetric patterns that must be assigned 0 parameters
        (Otherwise, undefined behavior when symmetric bond matches in an arbitrary direction)
    * Only checks string equivalence!
        for example
        check_bond_smarts_symmetric("[#6,#7:1]~[#7,#6:2]")
        will be a false negative
    * Does not match all possible bond smarts
        for example
        "[#6,#7:1]~[#7,#6:2]~[#1]"
        or
        "[#6:1](~[#8])(~[#16:2])"
        would throw an error
    """

    pattern = re.compile(r"\[(?P<atom1>.*)\:1\].\[(?P<atom2>.*)\:2\]")
    match = pattern.match(bond_smarts)

    assert type(match) is re.Match, "unrecognized bond smarts"
    assert match.span() == (0, len(bond_smarts)), "leftovers"
    return match.group("atom1") == match.group("atom2")


def get_symmetry_classes(rdmol: Chem.Mol) -> NDArray:
    """[atom.GetSymmetryClass() for atom in mol],
    just renumbered for convenience"""

    oemol = convert_to_oe(rdmol)
    oechem.OEPerceiveSymmetry(oemol)
    symmetry_classes = np.array([atom.GetSymmetryClass() for atom in oemol.GetAtoms()])
    n_classes = len(set(symmetry_classes))

    # make indexy / contiguous from 0 to n_classes
    idx_map = {old_idx: new_idx for (new_idx, old_idx) in enumerate(set(symmetry_classes))}
    symmetry_classes = np.array([idx_map[old_idx] for old_idx in symmetry_classes])
    assert set(symmetry_classes) == set(range(n_classes))

    return symmetry_classes


def get_spurious_param_idxs(mol, handle) -> NDArray:
    """Find all indices i such that adjusting handle.params[i] can
    result in distinct parameters being assigned to indistinguishable atoms in mol.

    Optimizing the parameters associated with these indices should be avoided.
    """

    symmetry_classes = get_symmetry_classes(mol)

    def assign_params(ff_params):
        return handle.partial_parameterize(ff_params, mol)

    def compute_spuriosity(ff_params):
        # apply parameters
        sys_params = assign_params(ff_params)

        # compute the mean per symmetry class
        class_sums = jax.ops.segment_sum(sys_params, symmetry_classes)
        class_means = class_sums / np.bincount(symmetry_classes)

        # expect no atom can be adjusted independently of others in its symmetry class
        expected_constant_within_class = class_means[symmetry_classes]
        assert expected_constant_within_class.shape == sys_params.shape
        deviation_from_class_means = sys_params - expected_constant_within_class
        spuriosity = jnp.sum(deviation_from_class_means ** 2)

        return spuriosity

    # TODO: may also want to try several points in the parameter space,
    #   randomly or systematically flipping signs...
    trial_params = np.ones(len(handle.params))  # TODO: generalize
    assert trial_params.shape == handle.params.shape

    # get idxs where component of gradient w.r.t. trial_params is != 0
    thresh = 1e-6
    g = grad(compute_spuriosity)(trial_params)
    spurious_idxs = np.where(np.abs(g) > thresh)[0]

    return spurious_idxs
