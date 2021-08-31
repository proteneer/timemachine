import io
from contextlib import redirect_stdout
from fe.free_energy_rabfe import (
    RABFEResult,
    setup_relative_restraints_by_distance,
    setup_relative_restraints_using_smarts
)
from fe.atom_mapping import CompareDistNonterminal

from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS


def capture_print(closure):
    """capture anything printed when we call closure()"""

    f = io.StringIO()
    with redirect_stdout(f):
        closure()
    printed = f.getvalue()
    return printed


def test_rabfe_result_to_from_log():
    """assert equality after round-trip to/from preferred terminal log format"""

    result = RABFEResult('my mol', 1.0, 2.0, 3.0, 4.0)

    printed = capture_print(result.log)
    first_line = printed.splitlines()[0]

    reconstructed = RABFEResult.from_log(first_line)
    assert result == reconstructed

def test_setting_up_restraints_using_smarts():
    seed = 814

    mcs_params = rdFMCS.MCSParameters()
    mcs_params.AtomTyper = CompareDistNonterminal()
    mcs_params.BondTyper = rdFMCS.BondCompare.CompareAny

    smi_a = "CCCONNN"
    smi_b = "CCCNNN"
    mol_a = Chem.MolFromSmiles(smi_a)
    mol_a = Chem.AddHs(mol_a)
    mol_b = Chem.MolFromSmiles(smi_b)
    mol_b = Chem.AddHs(mol_b)
    for mol in [mol_a, mol_b]:
        AllChem.EmbedMolecule(mol, randomSeed=seed)

    result = rdFMCS.FindMCS(
        [mol_a, mol_b],
        mcs_params
    )

    core = setup_relative_restraints_using_smarts(mol_a, mol_b, result.smartsString)
    assert core.shape == (2, 2)
