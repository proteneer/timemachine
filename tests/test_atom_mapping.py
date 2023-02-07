import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from timemachine.fe import atom_mapping
from timemachine.fe.mcgregor import MaxVisitsError, NoMappingError
from timemachine.fe.utils import plot_atom_mapping_grid

hif2a_set = "timemachine/datasets/fep_benchmark/hif2a/ligands.sdf"
eg5_set = "timemachine/datasets/fep_benchmark/eg5/ligands.sdf"

datasets = [
    hif2a_set,
    eg5_set,
]


def get_mol_name(mol) -> str:
    """Return the title for the given mol"""
    return mol.GetProp("_Name")


# hif2a is easy
# eg5 is challenging
# notable outliers for eg5:
# CHEMBL1077227 -> CHEMBL1086410 has 20736 cores of size 56
# CHEMBL1077227 -> CHEMBL1083836 has 14976 cores of size 48
# CHEMBL1086410 -> CHEMBL1083836 has 10752 cores of size 52
# CHEMBL1086410 -> CHEMBL1084935 has 6912 cores of size 60
@pytest.mark.parametrize("filepath", datasets)
@pytest.mark.nightly(reason="Slow")
def test_all_pairs(filepath):
    mols = Chem.SDMolSupplier(filepath, removeHs=False)
    mols = [m for m in mols]
    for idx, mol_a in enumerate(mols):
        for mol_b in mols[idx + 1 :]:

            all_cores = atom_mapping.get_cores(
                mol_a,
                mol_b,
                ring_cutoff=0.1,
                chain_cutoff=0.2,
                max_visits=1e7,  # 10 million max nodes to visit
                connected_core=False,
                max_cores=1000,
                enforce_core_core=True,
                complete_rings=False,
                enforce_chiral=True,
                min_threshold=0,
            )

            # # useful for visualization
            # for core_idx, core in enumerate(all_cores[:1]):
            #     res = plot_atom_mapping_grid(mol_a, mol_b, core, num_rotations=5)
            #     with open(
            #         f"atom_mapping_{get_mol_name(mol_a)}_to_{get_mol_name(mol_b)}_core_{core_idx}.svg", "w"
            #     ) as fh:
            #         fh.write(res)

            # note that this is probably the bottleneck for hif2a
            for core in all_cores:
                # ensure more than half the atoms are mapped
                assert len(core) > mol_a.GetNumAtoms() // 2

            print(
                f"{mol_a.GetProp('_Name')} -> {mol_b.GetProp('_Name')} has {len(all_cores)} cores of size {len(all_cores[0])}"
            )


def get_mol_by_name(mols, name):
    for m in mols:
        if get_mol_name(m) == name:
            return m

    assert 0, "Mol not found"


@pytest.mark.nogpu
def test_complete_rings_only():
    # this transformation has two ring changes:
    # a 6->5 member ring size change and a unicycle to bicycle change
    # we expect the MCS algorithm with complete rings only to map the single cycle core
    mols = Chem.SDMolSupplier(datasets[0], removeHs=False)
    mols = [m for m in mols]

    mol_a = get_mol_by_name(mols, "43")
    mol_b = get_mol_by_name(mols, "224")

    all_cores = atom_mapping.get_cores(
        mol_a,
        mol_b,
        ring_cutoff=0.1,
        chain_cutoff=0.2,
        max_visits=1e7,  # 10 million max nodes to visit
        connected_core=True,
        max_cores=1000,
        enforce_core_core=True,
        complete_rings=True,
        enforce_chiral=True,
        min_threshold=0,
    )

    assert len(all_cores) == 1
    core = all_cores[0]
    np.testing.assert_array_equal(
        np.array(
            [
                [1, 4],
                [2, 5],
                [3, 6],
                [4, 7],
                [5, 8],
                [6, 9],
                [7, 10],
                [15, 12],
                [16, 13],
                [17, 14],
                [18, 15],
                [32, 18],
                [19, 19],
                [20, 20],
                [26, 30],
                [27, 31],
            ]
        ),
        core,
    )


def tuples_to_set(arr):
    res = set()
    for a, b in arr:
        key = (a, b)
        assert key not in res
        res.add(key)
    return res


def assert_cores_are_equal(core_a, core_b):
    core_set_a = tuples_to_set(core_a)
    core_set_b = tuples_to_set(core_b)
    assert core_set_a == core_set_b


def get_all_cores_fzset(all_cores):
    all_cores_fzset = set()
    for core in all_cores:
        all_cores_fzset.add(frozenset(tuples_to_set(core)))
    return all_cores_fzset


def assert_core_sets_are_equal(core_set_a, core_set_b):
    fza = get_all_cores_fzset(core_set_a)
    fzb = get_all_cores_fzset(core_set_b)
    assert fza == fzb


# spot check
@pytest.mark.nogpu
def test_linker_map():
    # test that we can map a linker size change when connected_core=False, and enforce_core_core=False
    mol_a = Chem.MolFromMolBlock(
        """
  Mrv2219 11232201352D

 10 11  0  0  0  0            999 V2000
  -12.2008    0.8688    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.4558    0.0842    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.2808    0.0842    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.5357    0.8688    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.8683    1.3538    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.1785    3.3333    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.8460    2.8484    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.5134    3.3333    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.2585    4.1179    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.4335    4.1179    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  5  1  0  0  0  0
  2  3  1  0  0  0  0
  3  4  1  0  0  0  0
  4  5  1  0  0  0  0
  6  7  1  0  0  0  0
  6 10  1  0  0  0  0
  7  8  1  0  0  0  0
  8  9  1  0  0  0  0
  9 10  1  0  0  0  0
  5  7  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )
    mol_b = Chem.MolFromMolBlock(
        """
  Mrv2219 11232201352D

 11 12  0  0  0  0            999 V2000
  -12.2008    0.8688    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.4558    0.0842    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.2808    0.0842    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.5357    0.8688    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.8683    1.3538    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.1785    3.3333    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.8460    2.8484    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.5134    3.3333    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.2585    4.1179    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.4335    4.1179    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.2416    2.1140    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  5  1  0  0  0  0
  2  3  1  0  0  0  0
  3  4  1  0  0  0  0
  4  5  1  0  0  0  0
  6  7  1  0  0  0  0
  6 10  1  0  0  0  0
  7  8  1  0  0  0  0
  8  9  1  0  0  0  0
  9 10  1  0  0  0  0
  7 11  1  0  0  0  0
 11  5  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    all_cores = atom_mapping.get_cores(
        mol_a,
        mol_b,
        ring_cutoff=0.05,
        chain_cutoff=0.05,
        max_visits=1e7,  # 10 million max nodes to visit
        connected_core=False,
        max_cores=1000000,
        enforce_core_core=False,
        complete_rings=False,
        enforce_chiral=True,
        min_threshold=0,
    )

    assert len(all_cores) == 1

    assert_cores_are_equal(
        [[6, 6], [4, 4], [9, 9], [8, 8], [7, 7], [5, 5], [3, 3], [2, 2], [1, 1], [0, 0]], all_cores[0]
    )

    # now set connected_core and enforce_core_core to True
    all_cores = atom_mapping.get_cores(
        mol_a,
        mol_b,
        ring_cutoff=0.05,
        chain_cutoff=0.05,
        max_visits=1e7,  # 10 million max nodes to visit
        connected_core=True,
        max_cores=1000000,
        enforce_core_core=True,
        complete_rings=False,
        enforce_chiral=True,
        min_threshold=0,
    )

    # 2 possible matches, returned core ordering is fully determined
    # note that we return the larger of the two disconnected components here
    # (the 5 membered ring)
    assert len(all_cores) == 2

    expected_sets = ([[6, 6], [9, 9], [8, 8], [7, 7], [5, 5]], [[4, 4], [3, 3], [2, 2], [1, 1], [0, 0]])

    assert_core_sets_are_equal(expected_sets, all_cores)

    all_cores = atom_mapping.get_cores(
        mol_a,
        mol_b,
        ring_cutoff=0.05,
        chain_cutoff=0.05,
        max_visits=1e7,  # 10 million max nodes to visit
        connected_core=False,
        max_cores=1000000,
        enforce_core_core=True,
        complete_rings=False,
        enforce_chiral=True,
        min_threshold=0,
    )

    # 2 possible matches, if we do not allow for connected_core but do
    # require core_core, we have a 9-atom disconnected map, one is a 5-membered ring
    # the other is 4-membered chain. There's 2 allowed maps due to the 2 fold symmetry.
    assert len(all_cores) == 2

    expected_sets = (
        [[6, 6], [9, 9], [8, 8], [7, 7], [5, 5], [3, 3], [2, 2], [1, 1], [0, 0]],
        [[4, 4], [9, 9], [8, 8], [7, 7], [5, 5], [3, 3], [2, 2], [1, 1], [0, 0]],
    )

    assert_core_sets_are_equal(expected_sets, all_cores)


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


@pytest.mark.nogpu
def test_hif2a_failure():
    # special failure with error message:
    # pred_sgg_a = a_cycles[a] == sg_a_cycles[a], KeyError: 18
    mols = Chem.SDMolSupplier(hif2a_set, removeHs=False)
    mols = [m for m in mols]
    mol_a = get_mol_by_name(mols, "7a")
    mol_b = get_mol_by_name(mols, "224")
    all_cores = atom_mapping.get_cores(
        mol_a,
        mol_b,
        ring_cutoff=0.12,
        chain_cutoff=0.2,
        max_visits=1e7,
        connected_core=True,
        max_cores=1e6,
        enforce_core_core=True,
        complete_rings=True,
        enforce_chiral=True,
        min_threshold=0,
    )

    expected_core = np.array(
        [
            [24, 20],
            [9, 9],
            [26, 26],
            [32, 31],
            [6, 6],
            [19, 12],
            [4, 4],
            [23, 19],
            [2, 2],
            [21, 14],
            [27, 27],
            [10, 10],
            [0, 0],
            [3, 3],
            [29, 16],
            [25, 28],
            [20, 13],
            [1, 1],
            [7, 7],
            [22, 15],
            [28, 17],
            [36, 18],
            [30, 29],
            [5, 5],
            [31, 30],
            [8, 8],
        ]
    )

    assert_cores_are_equal(all_cores[0], expected_core)
    # for core_idx, core in enumerate(all_cores[:1]):
    #     res = plot_atom_mapping_grid(mol_a, mol_b, core, num_rotations=5)
    #     with open(f"atom_mapping_0_to_1_core_{core_idx}.svg", "w") as fh:
    #         fh.write(res)


@pytest.mark.nogpu
def test_cyclohexane_stereo():
    # test that cyclohexane in two different conformations has a core alignment that is stereo correct. Note that this needs a
    # larger than typical cutoff.
    mol_a, mol_b = get_cyclohexanes_different_confs()
    all_cores = atom_mapping.get_cores(
        mol_a,
        mol_b,
        ring_cutoff=0.15,
        chain_cutoff=0.30,
        max_visits=1e6,
        connected_core=True,
        max_cores=100000,
        enforce_core_core=True,
        complete_rings=True,
        enforce_chiral=True,
        min_threshold=0,
    )

    for core_idx, core in enumerate(all_cores[:1]):
        res = plot_atom_mapping_grid(mol_a, mol_b, core, num_rotations=5)
        with open(f"atom_mapping_0_to_1_core_{core_idx}.svg", "w") as fh:
            fh.write(res)

    # 1-indexed
    expected_core = np.array(
        [
            [1, 1],  # C
            [2, 2],  # C
            [3, 3],  # C
            [4, 4],  # C
            [5, 5],  # C
            [6, 6],  # C
            [16, 14],  # C1H
            [15, 13],  # C1H
            [17, 15],  # C2H
            [18, 16],  # C2H
            [7, 7],  # C3H
            [8, 8],  # C3H
            [9, 9],  # C4H
            [10, 10],  # C4H
            [14, 17],  # C5H
            [11, 12],  # C5H
            [13, 18],  # C6H
            [12, 11],  # C6H
        ]
    )

    # 0-indexed
    expected_core -= 1

    all_cores_fzset = get_all_cores_fzset(all_cores)
    assert tuples_to_set(expected_core) in all_cores_fzset

    assert len(all_cores) == 1


@pytest.mark.nogpu
def test_chiral_atom_map():
    mol_a = Chem.AddHs(Chem.MolFromSmiles("C"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("C"))

    AllChem.EmbedMolecule(mol_a, randomSeed=0)
    AllChem.EmbedMolecule(mol_b, randomSeed=0)

    core_kwargs = dict(
        ring_cutoff=np.inf,
        chain_cutoff=np.inf,
        max_visits=1e7,
        connected_core=True,
        max_cores=1e6,
        enforce_core_core=True,
        complete_rings=True,
        min_threshold=0,
    )

    chiral_aware_cores = atom_mapping.get_cores(mol_a, mol_b, enforce_chiral=True, **core_kwargs)
    chiral_oblivious_cores = atom_mapping.get_cores(mol_a, mol_b, enforce_chiral=False, **core_kwargs)

    assert len(chiral_oblivious_cores) == 4 * 3 * 2 * 1, "expected all hydrogen permutations to be valid"
    assert len(chiral_aware_cores) == (len(chiral_oblivious_cores) // 2), "expected only rotations to be valid"

    for (key, val) in chiral_aware_cores[0]:
        assert key == val, "expected first core to be identity map"
    assert len(chiral_aware_cores[0]) == 5


@pytest.mark.nogpu
def test_max_visits_exception():
    mol_a, mol_b = get_cyclohexanes_different_confs()
    core_kwargs = dict(
        ring_cutoff=0.1,
        chain_cutoff=0.2,
        connected_core=False,
        max_cores=1000,
        enforce_core_core=True,
        complete_rings=False,
        enforce_chiral=True,
        min_threshold=0,
    )
    cores = atom_mapping.get_cores(mol_a, mol_b, **core_kwargs, max_visits=10000)
    assert len(cores) > 0

    with pytest.raises(MaxVisitsError, match="Reached max number of visits: 1"):
        atom_mapping.get_cores(mol_a, mol_b, **core_kwargs, max_visits=1)


@pytest.mark.nogpu
def test_min_threshold():
    mol_a, mol_b = get_cyclohexanes_different_confs()
    core_kwargs = dict(
        ring_cutoff=0.1,
        chain_cutoff=0.2,
        connected_core=False,
        max_cores=1000,
        enforce_core_core=True,
        complete_rings=False,
        enforce_chiral=True,
        min_threshold=mol_a.GetNumAtoms(),
    )

    with pytest.raises(NoMappingError, match="Unable to find mapping with at least 18 atoms"):
        atom_mapping.get_cores(mol_a, mol_b, **core_kwargs, max_visits=10000)
