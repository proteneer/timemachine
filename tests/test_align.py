from copy import deepcopy

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from timemachine.constants import DEFAULT_FF
from timemachine.fe.atom_mapping import get_core_by_mcs, get_core_with_alignment, mcs
from timemachine.fe.topology import AtomMappingError
from timemachine.fe.utils import set_romol_conf
from timemachine.ff import Forcefield
from timemachine.md import align


def get_cyclohexanes_different_confs():
    """Two cyclohexane structures that differ enough in conformations to map poorly by MCS with threshold of 2.0"""
    mol_a = Chem.MolFromMolBlock(
        """
 cyclo_1

 18 18  0  0  1  0  0  0  0  0999 V2000
    0.7780    1.1695    0.1292 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.3871   -0.1008    0.2959 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6896    1.3214   -0.2192 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.5088    0.0613    0.0503 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7859   -1.2096   -0.4242 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.6085   -1.3920    0.2133 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.1105    2.1590    0.3356 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7549    1.5841   -1.2762 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.6874   -0.0047    1.1175 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.4858    0.1560   -0.4244 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6117   -1.0357   -1.4891 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.1610   -2.0015   -0.5036 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.5422   -1.8809    1.1852 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.4054   -2.0928   -0.2686 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.3677    1.7499   -0.5802 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.9940    1.7789    1.0067 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.9567   -0.0955    1.2253 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.2449   -0.1670   -0.3734 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1  3  1  0
  1 15  1  0
  1 16  1  0
  2  6  1  0
  2 17  1  0
  2 18  1  0
  3  4  1  0
  3  7  1  0
  3  8  1  0
  4  5  1  0
  4  9  1  0
  4 10  1  0
  5  6  1  0
  5 11  1  0
  5 14  1  0
  6 12  1  0
  6 13  1  0
M  END
$$$$""",
        removeHs=False,
    )

    mol_b = Chem.MolFromMolBlock(
        """
 cyclo_2

 18 18  0  0  1  0  0  0  0  0999 V2000
    0.7953    1.1614    0.0469 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.3031   -0.0613    0.5362 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6118    1.1962   -0.5144 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.9934   -0.1785   -1.1042 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6439   -1.3144   -0.1494 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.4262   -1.2251    0.6719 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2949    1.4641    0.2937 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6968    1.9715   -1.2775 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.0662   -0.1837   -1.3042 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.4970   -0.3613   -2.0575 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.6428   -1.9811    1.4121 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2637   -2.1987   -0.1345 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.4850    1.5611   -0.6965 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.8877    1.9212    0.8230 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.8010    0.1189    1.4889 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.1753   -0.3430   -0.0537 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2711   -0.8618    0.6186 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.1781   -0.6848    1.4006 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1  3  1  0
  1 13  1  0
  1 14  1  0
  2  6  1  0
  2 15  1  0
  2 16  1  0
  3  4  1  0
  3  7  1  0
  3  8  1  0
  4  5  1  0
  4  9  1  0
  4 10  1  0
  5  6  1  0
  5 12  1  0
  5 17  1  0
  6 11  1  0
  6 18  1  0
M  END
$$$$""",
        removeHs=False,
    )
    return mol_a, mol_b


def get_core(mol_a, mol_b, threshold=2.0):
    """Simple utility that finds a core by using the conformers with a threshold"""
    mcs_result = mcs(mol_a, mol_b, threshold=threshold)
    query = Chem.MolFromSmarts(mcs_result.smartsString)
    return get_core_by_mcs(mol_a, mol_b, query, threshold=threshold)


@pytest.mark.nogpu
def test_align_mols_by_core():
    """Uses a core to align two compounds. Uses two cyclohexanes with different conformers as the test case in the case of
    finding the best MCS"""

    mol_a, mol_b = get_cyclohexanes_different_confs()

    assert mol_a.GetNumAtoms() == mol_b.GetNumAtoms()

    core = get_core(mol_a, mol_b)

    # The difference in conformer prevents a complete mapping
    assert len(core) != mol_a.GetNumAtoms()

    hydrogenless_mol_a = Chem.RemoveHs(deepcopy(mol_a))
    hydrogenless_mol_b = Chem.RemoveHs(deepcopy(mol_b))

    restraint_core = get_core(hydrogenless_mol_a, hydrogenless_mol_b)

    assert len(restraint_core) == 6  # Number of carbons in mol

    ff = Forcefield.load_from_file(DEFAULT_FF)

    conf_a, conf_b = align.align_mols_by_core(mol_a, mol_b, restraint_core, ff)

    set_romol_conf(mol_a, conf_a)
    set_romol_conf(mol_b, conf_b)

    core = get_core(mol_a, mol_b)

    # The mapping should cover the entirety of the two compounds
    assert len(core) == mol_a.GetNumAtoms()


def test_align_core_different_size_mols():
    """Verifies that this works as expected if the size of the molecules aren't the same."""
    mol_a = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))  # benzene
    mol_b = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1-c2ccccc2"))  # bi-phyenl
    seed = 2022
    AllChem.EmbedMolecule(mol_a, randomSeed=seed)
    AllChem.EmbedMolecule(mol_b, randomSeed=seed)

    # Compounds don't overlap enough to build a core
    with pytest.raises(AtomMappingError):
        get_core(mol_a, mol_b)

    # Align based on the first ring in each
    core = np.zeros((6, 2), dtype=int)
    core[:, 0] = np.arange(6, dtype=int)
    core[:, 1] = np.arange(6, dtype=int)

    ff = Forcefield.load_from_file(DEFAULT_FF)

    conf_a, conf_b = align.align_mols_by_core(mol_a, mol_b, core, ff)

    set_romol_conf(mol_a, conf_a)
    set_romol_conf(mol_b, conf_b)

    core = get_core(mol_a, mol_b)

    # The mapping should cover the entirety of the benzene minus the hydrogen covering the carbon linker
    assert len(core) == mol_a.GetNumAtoms() - 1


def test_get_core_with_alignment():

    mol_a, mol_b = get_cyclohexanes_different_confs()

    core, _ = get_core_with_alignment(mol_a, mol_b)
    assert len(core) == mol_a.GetNumAtoms()
    assert len(core) == mol_b.GetNumAtoms()
