from rdkit import Chem
from rdkit.Chem import rdFMCS


def mcs_map(a, b):
    """
    Find the MCS map of going from A to B
    """
    core_pattern = rdFMCS.FindMCS([a, b], ringMatchesRingOnly=True).smartsString
    core = Chem.MolFromSmarts(core_pattern)

    # TBD take the cross product and pick the atom mapping that has the smallest RMSD

    a_to_core = {}
    a_matches = a.GetSubstructMatches(core)
    for match in a_matches:
        for core_idx, a_idx in enumerate(match):
            a_to_core[a_idx] = core_idx
        break

    if len(a_matches) > 1:
        print("WARNING: multiple core matches found for mol a")

    core_to_b = {}
    b_matches = b.GetSubstructMatches(core)
    for match in b_matches:
        for core_idx, atom in enumerate(match):
            core_to_b[core_idx] = atom
        break

    if len(b_matches) > 1:
        print("WARNING: multiple core matches found for mol b")

    a_to_b = {}
    for a, c in a_to_core.items():
        a_to_b[a] = core_to_b[a_to_core[a]]

    return a_to_b

if __name__ == "__main__":

    a = Chem.MolFromSmiles("c1ccccc1F")
    b = Chem.MolFromSmiles("c1cc(F)ccc1F")
    a = Chem.AddHs(a)
    b = Chem.AddHs(b)
    a_to_b = mcs_map(a, b)
    print(a_to_b)