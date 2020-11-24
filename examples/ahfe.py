# absolute hydration free energy

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
# 2. build ligand
# 3. convert to recipe

def get_romol_conf(mol):
    """Coordinates of mol's 0th conformer, in nanometers"""
    conformer = mol.GetConformer(0)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    return guest_conf/10 # from angstroms to nm

# construct an RDKit molecule of aspirin
# note: not using OpenFF Molecule because want to avoid the dependency (YTZ?)
romol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O"))

# generate conformers
AllChem.EmbedMolecule(romol)

# extract the 0th conformer
ligand_coords = get_romol_conf(romol)

# construct a 4-nanometer water box (from openmmtools approach: selecting out
#   of a large pre-equilibrated water box snapshot)
system, host_coords, box, topology = builders.build_water_system(4.0)

# write some conformations into this PDB file
writer = pdb_writer.PDBWriter([topology, romol], "debug.pdb")

# note the order in which the coordinates are concatenated in this step --
#   in a later step we will need to combine recipes in the same order
combined_conf = np.concatenate([host_coords, ligand_coords])

num_host_atoms = host_coords.shape[0]

# note: .py file rather than .offxml file
# note: _ccc suffix means "correctable charge corrections"
ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_ccc.py').read())

ligand_recipe = Recipe.from_rdkit(romol, ff_handlers)

# note: "bound" potentials are energy functions where the parameters have been
#   set to a constant using .bind()
#   (contrast with "unbound" potentials, for cases where the parameters are
#   variable and where we may be interested in dU/dparams)
for bp in ligand_recipe.bound_potentials:
    if isinstance(bp, potentials.Nonbonded):
        # note: this is something like "bp.set_lambda_offset_idxs(ones)"
        #   (at initialization, offset_indices are all zeros)
        array = bp.get_lambda_offset_idxs()
        array[:] = 1 

water_recipe = Recipe.from_openmm(system)

# note: this step is dependent on order -- ensure consistency with the order
#   in which coordinates were concatenated above
# note: calling `recipe.combine(other)` will offset all of the bond indices etc.
#   within the sub-recipe described by `other`
combined_recipe = water_recipe.combine(ligand_recipe)

# note: lambda goes from 0 to +infinity, where we approximate +infinity with
#   any number of nanometers sufficiently large that the ligand has almost no
#   interaction with the rest of the system (in this case, 1.2 nm will do)
for final_lamb in np.linspace(0, 1.2, 8):

    seed = 2020

    # note: the .impl() call at the end returns a pickle-able version of the
    #   wrapper function -- since contexts are not pickle-able -- which will
    #   be useful later in timemachine's multi-device parallelization strategy)
    # note: OpenMM unit system used throughout
    #   (temperature: kelvin, timestep: picosecond, collision_rate: picosecond^-1)
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
        # note: YTZ recommends np.float32 here
        # note: difference between .bound_impl() and .impl()
        fn = bp.bound_impl(precision=np.float32)

        # note: fn.execute(...) rather than fn(...)
        # TODO: what was the 3rd argument of fn.execute(...)?
        du_dx, du_dl, u = fn.execute(x0, box, 100.0)

        # checking energy gradient norm can be more diagnostic than checking
        #   energy directly
        max_norm_water = np.amax(np.linalg.norm(du_dx[:num_host_atoms], axis=0))
        max_norm_ligand = np.amax(np.linalg.norm(du_dx[num_host_atoms:], axis=0))

        u_impls.append(fn)

    # context components: positions, velocities, box, integrator, energy fxns
    ctxt = custom_ops.Context(
        x0,
        v0,
        box,
        intg,
        u_impls
    )

    for lamb_idx, lamb in enumerate(np.linspace(1.0, final_lamb, 1000)):
        # note: unlike in OpenMM, .step() accepts a float (for lambda) and does
        #   not accept an integer (for n_steps) -- .step() takes 1 step at a time
        ctxt.step(lamb)

    # note: these 5000 steps are "equilibration", before we attach a reporter /
    #   "observable" to the context and start running "production"
    for _ in range(5000):
        ctxt.step(final_lamb)

    # TODO: what was the second argument -- reporting interval in steps?
    du_dl_obs = custom_ops.AvgPartialUPartialLambda(u_impls, 5)

    ctxt.add_observable(du_dl_obs)

    for _ in range(100000):
        ctxt.step(final_lamb)

    du_dl = du_dl_obs.avg_du_dl()

    print(final_lamb, du_dl)
