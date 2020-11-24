# relative hydration free energy

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from fe import pdb_writer
from md import Recipe
from md import builders

from timemachine.lib import potentials, custom_ops
from timemachine.lib import LangevinIntegrator

from ff.handlers.deserialize import deserialize_handlers


# 1. build water box
# 2. build pair of ligands
# 3. convert to recipe

def get_romol_conf(mol):
    conformer = mol.GetConformer(0)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    return guest_conf / 10  # from angstroms to nm


# convert an oxygen to a fluorine
romol_a = Chem.AddHs(Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O"))
romol_b = Chem.AddHs(Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)F"))

# generate conformers
AllChem.EmbedMolecule(romol_a)
AllChem.EmbedMolecule(romol_b)

# write in order: waterbox, ligand_a, ligand_b
ligand_a_coords = get_romol_conf(romol_a)
ligand_b_coords = get_romol_conf(romol_b)
system, host_coords, box, topology = builders.build_water_system(4.0)

writer = pdb_writer.PDBWriter([topology, romol_a, romol_b], "debug.pdb")

# concatenate in order
combined_conf = np.concatenate([host_coords, ligand_a_coords, ligand_b_coords])

# "host" in this case = "waterbox"
num_host_atoms = host_coords.shape[0]

ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_ccc.py').read())

ligand_a_recipe = Recipe.from_rdkit(romol_a, ff_handlers)
ligand_b_recipe = Recipe.from_rdkit(romol_b, ff_handlers)

for bp in ligand_a_recipe.bound_potentials:
    if isinstance(bp, potentials.Nonbonded):
        array = bp.get_lambda_offset_idxs()
        array[:] = 1

for bp in ligand_b_recipe.bound_potentials:
    if isinstance(bp, potentials.Nonbonded):
        array = bp.get_lambda_offset_idxs()
        array[:] = 1

water_recipe = Recipe.from_openmm(system)

ligand_ab_recipe = ligand_a_recipe.combine(ligand_b_recipe)

num_a_atoms = ligand_a_coords.shape[0]
num_b_atoms = ligand_b_coords.shape[0]

#
for bp in ligand_ab_recipe.bound_potentials:
    if isinstance(bp, potentials.Nonbonded):
        exclusion_idxs = bp.get_exclusion_idxs()
        scale_factors = bp.get_scale_factors()
        new_exclusions = []
        new_scale_factors = []
        for i in range(num_a_atoms):
            for j in range(num_b_atoms):
                new_exclusions.append((i, j + num_a_atoms))

                # an exclusion scale factor of 1.0 means "subtract 1.0 times
                #   the contribution of this pair" -- and this is done in a NaN-
                #   safe way using some wizardry that lets you add Nan and
                #   subtract NaN from the accumulator, using some wizardry
                new_scale_factors.append((1.0, 1.0))
        new_exclusions = np.concatenate([exclusion_idxs, new_exclusions])
        new_scale_factors = np.concatenate([scale_factors, new_scale_factors])

        # note: np.int32 is important here!
        bp.set_exclusion_idxs(new_exclusions.astype(np.int32))
        bp.set_scale_factors(new_scale_factors.astype(np.float64))

# note: np.int32 is important here! may encounter type error with np.int64
restr = potentials.CentroidRestraint(
    np.arange(num_a_atoms, dtype=np.int32),
    np.arange(num_b_atoms, dtype=np.int32),
    ligand_ab_recipe.masses,
    200,
    0.0
).bind([])

ligand_ab_recipe.bound_potentials.append(restr)
ligand_ab_recipe.vjp_fns.append([])  # TODO: is this append command important?

combined_recipe = water_recipe.combine(ligand_ab_recipe)

for final_lamb in np.linspace(0, 1.2, 8):

    seed = 2020

    intg = LangevinIntegrator(
        300.0,
        1.5e-3,
        1.0,
        combined_recipe.masses,
        seed
    ).impl()

    x0 = combined_conf
    v0 = np.zeros_like(x0)

    u_impls = []

    for bp in combined_recipe.bound_potentials:
        fn = bp.bound_impl(precision=np.float32)
        du_dx, du_dl, u = fn.execute(x0, box, 100.0)
        max_norm_water = np.amax(np.linalg.norm(du_dx[:num_host_atoms], axis=0))
        max_norm_ligand = np.amax(np.linalg.norm(du_dx[num_host_atoms:], axis=0))
        u_impls.append(fn)

    ctxt = custom_ops.Context(
        x0,
        v0,
        box,
        intg,
        u_impls
    )

    for lamb_idx, lamb in enumerate(np.linspace(1.0, final_lamb, 1000)):
        if lamb_idx % 1000 == 0:
            writer.write_frame(ctxt.get_x_t() * 10)  # note angstroms/nm conversion
            print(lamb_idx, ctxt.get_u_t())
        ctxt.step(lamb)

    for step in range(5000):
        if step % 1000 == 0:
            writer.write_frame(ctxt.get_x_t() * 10)  # note angstroms/nm conversion
        ctxt.step(final_lamb)

    du_dl_obs = custom_ops.AvgPartialUPartialLambda(u_impls, 5)

    ctxt.add_observable(du_dl_obs)

    for step in range(20000):

        if step % 1000 == 0:
            writer.write_frame(ctxt.get_x_t() * 10)

        ctxt.step(final_lamb)

    writer.close()

    du_dl = du_dl_obs.avg_du_dl()

    print(final_lamb, du_dl)
