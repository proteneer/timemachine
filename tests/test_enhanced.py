from timemachine.datasets import fetch_freesolv
from timemachine.md.enhanced import identify_rotatable_bonds


def test_identify_rotatable_bonds_runs_on_freesolv():
    """pass if no runtime errors are encountered"""
    mols = fetch_freesolv()

    for mol in mols:
        _ = identify_rotatable_bonds(mol)
