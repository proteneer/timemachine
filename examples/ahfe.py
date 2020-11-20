# hydration free energy

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from fe import pdb_writer
from md import Recipe
from md import builders

import asciiplotlib

from timemachine.lib import potentials
from timemachine.lib import potentials, custom_ops
from timemachine.lib import LangevinIntegrator

# from ff.handlers import openmm_deserializer
from ff.handlers.deserialize import deserialize_handlers

# 1. build water box
# 2. build ligand
# 3. convert to recipe

def get_romol_conf(mol):
    conformer = mol.GetConformer(0)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    guest_conf = guest_conf/10 # from angstroms to nm
    return np.array(guest_conf, dtype=np.float64)

romol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O"))

AllChem.EmbedMolecule(romol)

ligand_coords = get_romol_conf(romol)
system, host_coords, box, topology = builders.build_water_system(4.0)


writer = pdb_writer.PDBWriter([topology, romol], "debug.pdb")
# box[np.diag_indices(3)] += 10

combined_conf = np.concatenate([host_coords, ligand_coords])

num_host_atoms = host_coords.shape[0]

ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_ccc.py').read())

ligand_recipe = Recipe.from_rdkit(romol, ff_handlers)

for bp in ligand_recipe.bound_potentials:
    if isinstance(bp, potentials.Nonbonded):
        array = bp.get_lambda_offset_idxs()
        array[:] = 1 

water_recipe = Recipe.from_openmm(system)
combined_recipe = water_recipe.combine(ligand_recipe)

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
        # print(bp, du_dx)
        max_norm_water = np.amax(np.linalg.norm(du_dx[:num_host_atoms], axis=0))
        max_norm_ligand = np.amax(np.linalg.norm(du_dx[num_host_atoms:], axis=0))
        # print(bp, max_norm_water, max_norm_ligand)
        u_impls.append(fn)

    ctxt = custom_ops.Context(
        x0,
        v0,
        box,
        intg,
        u_impls
    )

    for lamb_idx, lamb in enumerate(np.linspace(1.0, final_lamb, 1000)):
        ctxt.step(lamb)

    for _ in range(5000):
        ctxt.step(final_lamb)

    du_dl_obs = custom_ops.AvgPartialUPartialLambda(u_impls, 5)

    ctxt.add_observable(du_dl_obs)

    for _ in range(100000):
        ctxt.step(final_lamb)

    du_dl = du_dl_obs.avg_du_dl()

    print(final_lamb, du_dl)


# for _ in range(10000):
    # ctxt.step(1.0)
