# taken from Simon Boothroyd's amazing recharge model
# https://raw.githubusercontent.com/openforcefield/openff-recharge/5384eddb00e594c14c925b2d72956cfa3e51a874/openff/recharge/charges/bcc.py
# (ytz): we should just vendor this later.

import logging
import re
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")
OEMol = Any


class RechargeException(BaseException):
    pass


class InvalidSmirksError(RechargeException):
    """An exception raised when an invalid smirks pattern is provided."""

    def __init__(self, *args, smirks: str, **kwargs):
        """
        Parameters
        ----------
        smirks
            The SMIRKS pattern which could not be parsed.
        """

        super(InvalidSmirksError, self).__init__(*args, **kwargs)
        self.smirks = smirks


def call_openeye(
    oe_callable: Callable[[T], bool],
    *args: T,
    exception_type: type[BaseException] = RuntimeError,
    exception_kwargs: Optional[dict[str, Any]] = None,
):
    """Wraps a call to an OpenEye function, either capturing the output in an
    exception if the function does not complete successfully, or redirecting it
    to the logger.
    Parameters
    ----------
    oe_callable
        The OpenEye function to call.
    args
        The arguments to pass to the OpenEye function.
    exception_type:
        The type of exception to raise when the function does not
        successfully complete.
    exception_kwargs
        The keyword arguments to pass to the exception.
    """
    from openeye import oechem

    if exception_kwargs is None:
        exception_kwargs = {}

    output_stream = oechem.oeosstream()

    oechem.OEThrow.SetOutputStream(output_stream)
    oechem.OEThrow.Clear()

    status = oe_callable(*args)

    oechem.OEThrow.SetOutputStream(oechem.oeerr)

    output_string = output_stream.str().decode("UTF-8")

    output_string = output_string.replace("Warning: ", "")
    output_string = re.sub("^: +", "", output_string, flags=re.MULTILINE)
    output_string = re.sub("\n$", "", output_string)

    if not status:
        # noinspection PyArgumentList
        raise exception_type("\n" + output_string, **exception_kwargs)

    elif len(output_string) > 0:
        logging.debug(output_string)


def match_smirks(smirks: str, oe_molecule: OEMol, unique: bool = False) -> list[dict[int, int]]:
    """Attempt to find the indices (optionally unique) of all atoms which
    match a particular SMIRKS pattern.
    Parameters
    ----------
    smirks
        The SMIRKS pattern to match.
    oe_molecule
        The molecule to match against.
    unique
        Whether to return back only unique matches.
    Returns
    -------
        A list of all the matches where each match is stored as a dictionary of
        the smirks indices and their corresponding matched atom indices.
    """
    from openeye import oechem

    query = oechem.OEQMol()
    call_openeye(
        oechem.OEParseSmarts,
        query,
        smirks,
        exception_type=InvalidSmirksError,
        exception_kwargs={"smirks": smirks},
    )

    substructure_search = oechem.OESubSearch(query)
    substructure_search.SetMaxMatches(0)

    matches = []

    for match in substructure_search.Match(oe_molecule, unique):
        matched_indices = {
            atom_match.pattern.GetMapIdx() - 1: atom_match.target.GetIdx()
            for atom_match in match.GetAtoms()
            if atom_match.pattern.GetMapIdx() != 0
        }

        matches.append(matched_indices)

    return matches


class AromaticityModel:
    """A class which will assign aromatic flags to a molecule based upon
    a specified aromatic model."""

    @classmethod
    def _set_aromatic(cls, ring_matches: list[dict[int, int]], oe_molecule: OEMol):
        """Flag all specified ring atoms and all ring bonds between those atoms
        as being aromatic.

        Parameters
        ----------
        ring_matches
            The indices of the atoms in each of the rings to flag as aromatic.
        oe_molecule
            The molecule to assign the aromatic flags to.
        """

        atoms = {atom.GetIdx(): atom for atom in oe_molecule.GetAtoms()}
        bonds = {tuple(sorted((bond.GetBgnIdx(), bond.GetEndIdx()))): bond for bond in oe_molecule.GetBonds()}

        for ring_match in ring_matches:
            ring_atom_indices = {match for match in ring_match.values()}

            for matched_atom_index in ring_atom_indices:
                atoms[matched_atom_index].SetAromatic(True)

            for (index_a, index_b), bond in bonds.items():
                if index_a not in ring_atom_indices or index_b not in ring_atom_indices:
                    continue

                if not bond.IsInRing():
                    continue

                bond.SetAromatic(True)

    @classmethod
    def _assign_am1bcc(cls, oe_molecule: OEMol):
        """Applies aromaticity flags based upon the aromaticity model
        outlined in the original AM1BCC publications _[1].

        Parameters
        ----------
        oe_molecule
            The molecule to assign aromatic flags to.

        References
        ----------
        [1] Jakalian, A., Jack, D. B., & Bayly, C. I. (2002). Fast, efficient generation
            of high-quality atomic charges. AM1-BCC model: II. Parameterization and
            validation. Journal of computational chemistry, 23(16), 1623–1641.
        """
        from openeye import oechem

        oechem.OEClearAromaticFlags(oe_molecule)

        x_type = "[#6X3,#7X2,#15X2,#7X3+1,#15X3+1,#8X2+1,#16X2+1:N]"
        y_type = "[#6X2-1,#7X2-1,#8X2,#16X2,#7X3,#15X3:N]"
        z_type = x_type

        # Case 1)
        case_1_smirks = (
            f"{x_type.replace('N', '1')}1"
            f"=@{x_type.replace('N', '2')}"
            f"-@{x_type.replace('N', '3')}"
            f"=@{x_type.replace('N', '4')}"
            f"-@{x_type.replace('N', '5')}"
            f"=@{x_type.replace('N', '6')}-@1"
        )

        case_1_matches = match_smirks(case_1_smirks, oe_molecule, unique=True)
        case_1_atoms = {match for matches in case_1_matches for match in matches.values()}

        cls._set_aromatic(case_1_matches, oe_molecule)

        # Track the ar6 assignments as there is no atom attribute to
        # safely determine if an atom is in a six member ring when
        # that same atom is also in a five member ring.
        ar6_assignments = {*case_1_atoms}

        # Case 2)
        case_2_smirks = (
            f"{x_type.replace('N', '1')}1"
            f"=@{x_type.replace('N', '2')}"
            f"-@{x_type.replace('N', '3')}"
            f"=@{x_type.replace('N', '4')}"
            f"-@{x_type.replace('N', '5')}"
            f":@{x_type.replace('N', '6')}-@1"
        )

        previous_case_2_atoms = None
        case_2_atoms: set[int] = set()

        while previous_case_2_atoms != case_2_atoms:
            case_2_matches = match_smirks(case_2_smirks, oe_molecule, unique=True)
            # Enforce the ar6 condition
            case_2_matches = [
                case_2_match
                for case_2_match in case_2_matches
                if case_2_match[4] in ar6_assignments and case_2_match[5] in ar6_assignments
            ]

            previous_case_2_atoms = case_2_atoms
            case_2_atoms = {match for matches in case_2_matches for match in matches.values()}

            ar6_assignments.update(case_2_atoms)
            cls._set_aromatic(case_2_matches, oe_molecule)

        # Case 3)
        case_3_smirks = (
            f"{x_type.replace('N', '1')}1"
            f"=@{x_type.replace('N', '2')}"
            f"-@{x_type.replace('N', '3')}"
            f":@{x_type.replace('N', '4')}"
            f"~@{x_type.replace('N', '5')}"
            f":@{x_type.replace('N', '6')}-@1"
        )

        previous_case_3_atoms = None
        case_3_atoms: set[int] = set()

        while previous_case_3_atoms != case_3_atoms:
            case_3_matches = match_smirks(case_3_smirks, oe_molecule, unique=True)

            # Enforce the ar6 condition
            case_3_matches = [
                case_3_match
                for case_3_match in case_3_matches
                if case_3_match[2] in ar6_assignments
                and case_3_match[3] in ar6_assignments
                and case_3_match[4] in ar6_assignments
                and case_3_match[5] in ar6_assignments
            ]

            previous_case_3_atoms = case_3_atoms
            case_3_atoms = {match for matches in case_3_matches for match in matches.values()}

            ar6_assignments.update(case_3_atoms)

            cls._set_aromatic(case_3_matches, oe_molecule)

        # Case 4)
        case_4_smirks = (
            "[#6+1:1]1"
            f"-@{x_type.replace('N', '2')}"
            f"=@{x_type.replace('N', '3')}"
            f"-@{x_type.replace('N', '4')}"
            f"=@{x_type.replace('N', '5')}"
            f"-@{x_type.replace('N', '6')}"
            f"=@{x_type.replace('N', '7')}-@1"
        )

        case_4_matches = match_smirks(case_4_smirks, oe_molecule, unique=True)
        case_4_atoms = {match for matches in case_4_matches for match in matches.values()}

        cls._set_aromatic(case_4_matches, oe_molecule)

        # Case 5)
        case_5_smirks = (
            f"{y_type.replace('N', '1')}1"
            f"-@{z_type.replace('N', '2')}"
            f"=@{z_type.replace('N', '3')}"
            f"-@{x_type.replace('N', '4')}"
            f"=@{x_type.replace('N', '5')}-@1"
        )

        ar_6_ar_7_matches = {
            *case_1_atoms,
            *case_2_atoms,
            *case_3_atoms,
            *case_4_atoms,
        }

        case_5_matches = match_smirks(case_5_smirks, oe_molecule, unique=True)
        case_5_matches = [
            matches
            for matches in case_5_matches
            if matches[1] not in ar_6_ar_7_matches and matches[2] not in ar_6_ar_7_matches
        ]

        cls._set_aromatic(case_5_matches, oe_molecule)

    @classmethod
    def assign(cls, oe_molecule: OEMol):
        """Clears the current aromaticity flags on a molecule and assigns
        new ones based on the specified aromaticity model.

        Parameters
        ----------
        oe_molecule
            The molecule to assign aromatic flags to.
        """
        cls._assign_am1bcc(oe_molecule)
