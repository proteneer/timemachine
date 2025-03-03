import pytest

from timemachine.md.enhanced import identify_rotatable_bonds
from timemachine.testsystems import fetch_freesolv

pytestmark = [pytest.mark.nocuda]


def test_identify_rotatable_bonds_runs_on_freesolv():
    """pass if no runtime errors are encountered"""
    mols = fetch_freesolv()

    for mol in mols:
        _ = identify_rotatable_bonds(mol)
