import numpy as np
import md

from rdkit import Chem
from rdkit.Chem import AllChem

from ff.handlers.deserialize import deserialize_handlers

# from training import builders

from md import Recipe
from md import builders

def test_recipe_from_rdkit():
    ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_ccc.py').read())
    suppl = Chem.SDMolSupplier('tests/data/ligands_40.sdf', removeHs=False)
    for mol_idx, mol in enumerate(suppl):
        print(mol_idx, Chem.MolToSmiles(mol))
        system = md.Recipe.from_rdkit(mol, ff_handlers)
        if mol_idx > 2:
            break

def test_recipe_from_openmm():
    fname = 'tests/data/hif2a_nowater_min.pdb'
    openmm_system, _, _, _, _, _ = builders.build_protein_system(fname)
    md.Recipe.from_openmm(openmm_system)


def test_combine_recipe():
    ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_ccc.py').read())
    aspirin = Chem.AddHs(Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O"))
    AllChem.EmbedMolecule(aspirin)
    ligand_recipe = md.Recipe.from_rdkit(aspirin, ff_handlers)
    fname = 'tests/data/hif2a_nowater_min.pdb'
    pdb = open(fname, 'r').read()
    openmm_system, openmm_conf, _, _, _, _ = builders.build_protein_system('tests/data/hif2a_nowater_min.pdb')
    protein_recipe = md.Recipe.from_openmm(openmm_system)

    for left_recipe, right_recipe in [[protein_recipe, ligand_recipe], [ligand_recipe, protein_recipe]]:

        combined_recipe = left_recipe.combine(right_recipe)
        qlj = np.ones((aspirin.GetNumAtoms() + openmm_conf.shape[0], 3))

        left_nonbonded_potential = left_recipe.bound_potentials[-1]
        right_nonbonded_potential = right_recipe.bound_potentials[-1]
        combined_nonbonded_potential = combined_recipe.bound_potentials[-1]

        left_idxs = left_nonbonded_potential.get_exclusion_idxs()
        right_idxs = right_nonbonded_potential.get_exclusion_idxs()
        combined_idxs = combined_nonbonded_potential.get_exclusion_idxs()

        n_left = len(left_recipe.masses)
        n_right = len(right_recipe.masses)

        np.testing.assert_array_equal(np.concatenate([left_idxs, right_idxs + n_left]), combined_idxs)

        for bp in combined_recipe.bound_potentials:
            bp.bound_impl(precision=np.float32)
