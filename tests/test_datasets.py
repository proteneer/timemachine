from timemachine.datasets import fetch_freesolv


def test_fetch_freesolv():
    """assert expected number of molecules loaded -- with unique names and expected property annotations"""
    mols = fetch_freesolv()

    # expected number of mols loaded
    assert len(mols) == 642

    # expected mol properties present, interpretable as floats
    for mol in mols:
        props = mol.GetPropsAsDict()
        _, _ = float(props["dG"]), float(props["dG_err"])

    # unique names
    names = [mol.GetProp("_Name") for mol in mols]
    assert len(set(names)) == len(names)
