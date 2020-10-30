# test protocols for setting up relative binding free energy calculations.

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from timemachine.lib import potentials

from ff.handlers.deserialize import deserialize_handlers

from fe import rbfe
from md import Recipe

def test_stage_0():

    benzene = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    phenol = Chem.AddHs(Chem.MolFromSmiles("Oc1ccccc1"))

    AllChem.EmbedMolecule(benzene)
    AllChem.EmbedMolecule(phenol)

    ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_ccc.py').read())
    r_benzene = Recipe.from_rdkit(benzene, ff_handlers)
    r_phenol = Recipe.from_rdkit(phenol, ff_handlers)

    combined_recipe = r_benzene.combine(r_phenol)

    core_pairs = np.array([
        [0,1],
        [1,2],
        [2,3],
        [3,4],
        [4,5]
    ], dtype=np.int32)
    core_pairs[:, 1] += benzene.GetNumAtoms()

    b_idxs = np.arange(phenol.GetNumAtoms()) + benzene.GetNumAtoms()

    com_k = 10.0
    core_k = 200.0
    rbfe.stage_0(combined_recipe, b_idxs, core_pairs, com_k, core_k)

    centroid_count = 0
    core_count = 0
    nb_count = 0

    for bp in combined_recipe.bound_potentials:
        if isinstance(bp, potentials.LambdaPotential):
            u_fn = bp.get_u_fn()
            if isinstance(u_fn, potentials.CentroidRestraint):
                centroid_count += 1

                # (1-lambda)*u_fn
                np.testing.assert_equal(bp.get_multiplier(), -1.0)
                np.testing.assert_equal(bp.get_offset(), 1.0)

                np.testing.assert_equal(u_fn.get_a_idxs(), core_pairs[:, 0])
                np.testing.assert_equal(u_fn.get_b_idxs(), core_pairs[:, 1])

            elif isinstance(u_fn, potentials.CoreRestraint):
                core_count += 1

                # lambda*u_fn
                np.testing.assert_equal(bp.get_multiplier(), 1.0)
                np.testing.assert_equal(bp.get_offset(), 0.0)

                np.testing.assert_equal(u_fn.get_bond_idxs(), core_pairs)

        elif isinstance(bp, potentials.Nonbonded):
            nb_count += 1
            test_plane_idxs = bp.get_lambda_plane_idxs()
            ref_plane_idxs = np.zeros_like(test_plane_idxs)
            ref_plane_idxs[benzene.GetNumAtoms():] = 1
            np.testing.assert_array_equal(ref_plane_idxs, test_plane_idxs)

            test_offset_idxs = bp.get_lambda_offset_idxs()
            np.testing.assert_array_equal(test_offset_idxs, np.zeros_like(test_offset_idxs))

        # test C++ side of things
        print(bp)
        bp.bound_impl(precision=np.float32)

    assert nb_count == 1
    assert centroid_count == 1
    assert core_count == 1

def test_stage_1():

    benzene = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    phenol = Chem.AddHs(Chem.MolFromSmiles("Oc1ccccc1"))

    AllChem.EmbedMolecule(benzene)
    AllChem.EmbedMolecule(phenol)

    ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_ccc.py').read())
    r_benzene = Recipe.from_rdkit(benzene, ff_handlers)
    r_phenol = Recipe.from_rdkit(phenol, ff_handlers)

    combined_recipe = r_benzene.combine(r_phenol)

    core_pairs = np.array([
        [0,1],
        [1,2],
        [2,3],
        [3,4],
        [4,5]
    ], dtype=np.int32)
    core_pairs[:, 1] += benzene.GetNumAtoms()

    a_idxs = np.arange(benzene.GetNumAtoms())
    b_idxs = np.arange(phenol.GetNumAtoms()) + benzene.GetNumAtoms()

    com_k = 10.0
    core_k = 200.0
    rbfe.stage_1(combined_recipe, a_idxs, b_idxs, core_pairs, core_k)

    core_count = 0
    nb_count = 0
    for bp in combined_recipe.bound_potentials:
        if isinstance(bp, potentials.LambdaPotential):
            assert 0
        elif isinstance(bp, potentials.CentroidRestraint):
            assert 0
        elif isinstance(bp, potentials.CoreRestraint):
            core_count += 1
            np.testing.assert_equal(bp.get_bond_idxs(), core_pairs)
        elif isinstance(bp, potentials.Nonbonded):

            nb_count += 1

            test_plane_idxs = bp.get_lambda_plane_idxs()
            np.testing.assert_array_equal(test_plane_idxs, np.zeros_like(test_plane_idxs))

            test_offset_idxs = bp.get_lambda_offset_idxs()
            ref_offset_idxs = np.zeros_like(test_offset_idxs)
            ref_offset_idxs[benzene.GetNumAtoms():] = 1
            np.testing.assert_array_equal(ref_offset_idxs, test_offset_idxs)

            # ensure exclusions are added correctly
            combined_idxs = bp.get_exclusion_idxs()

            left_idxs = r_benzene.bound_potentials[-1].get_exclusion_idxs()
            right_idxs = r_phenol.bound_potentials[-1].get_exclusion_idxs()

            n_left = benzene.GetNumAtoms()
            n_right = phenol.GetNumAtoms()

            assert (len(left_idxs) + len(right_idxs) + n_left * n_right) == len(combined_idxs)
            test_set = np.sort(combined_idxs[len(left_idxs) + len(right_idxs):])

            ref_set = []
            for i in range(n_left):
                for j in range(n_right):
                    ref_set.append((i, j+n_left))
            ref_set = np.sort(np.array(ref_set, dtype=np.int32))

            np.testing.assert_array_equal(test_set, ref_set)

        bp.bound_impl(precision=np.float32)

    assert nb_count == 1
    assert core_count == 1
