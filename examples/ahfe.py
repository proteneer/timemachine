# absolute hydration free energy

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from fe import pdb_writer
from fe import topology
from fe.utils import get_romol_conf
from md import builders

from timemachine.lib import potentials, custom_ops
from timemachine.lib import LangevinIntegrator

import functools
import jax

from timemachine.ff import Forcefield

from ff.handlers import nonbonded, bonded, openmm_deserializer

from ff.handlers.deserialize import deserialize_handlers

# 1. build water box
# 2. build ligand
# 3. convert to recipe

# construct an RDKit molecule of aspirin
# note: not using OpenFF Molecule because want to avoid the dependency (YTZ?)
romol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O"))

ligand_masses = [a.GetMass() for a in romol.GetAtoms()]

# generate conformers
AllChem.EmbedMolecule(romol)

# extract the 0th conformer
ligand_coords = get_romol_conf(romol)

# construct a 4-nanometer water box (from openmmtools approach: selecting out
#   of a large pre-equilibrated water box snapshot)
system, host_coords, box, omm_topology = builders.build_water_system(4.0)

host_bps, host_masses = openmm_deserializer.deserialize_system(system, cutoff=1.2)

combined_masses = np.concatenate([host_masses, ligand_masses])

# write some conformations into this PDB file
writer = pdb_writer.PDBWriter([omm_topology, romol], "debug.pdb")

# note the order in which the coordinates are concatenated in this step --
#   in a later step we will need to combine recipes in the same order
combined_coords = np.concatenate([host_coords, ligand_coords])

num_host_atoms = host_coords.shape[0]

# note: .py file rather than .offxml file
# note: _ccc suffix means "correctable charge corrections"
ff_handlers = deserialize_handlers(open("ff/params/smirnoff_1_1_0_ccc.py").read())

final_potentials = []
final_vjp_and_handles = []

# keep the bonded terms in the host the same.
# but we keep the nonbonded term for a subsequent modification
for bp in host_bps:
    if isinstance(bp, potentials.Nonbonded):
        host_p = bp
    else:
        final_potentials.append(bp)
        final_vjp_and_handles.append(None)

ff = Forcefield(ff_handlers)
gbt = topology.BaseTopology(romol, ff)
hgt = topology.HostGuestTopology(host_p, gbt)

# setup the parameter handlers for the ligand
tuples = [
    [hgt.parameterize_harmonic_bond, [ff.hb_handle]],
    [hgt.parameterize_harmonic_angle, [ff.ha_handle]],
    [hgt.parameterize_proper_torsion, [ff.pt_handle]],
    [hgt.parameterize_improper_torsion, [ff.it_handle]],
    [hgt.parameterize_nonbonded, [ff.q_handle, ff.lj_handle]],
]

# instantiate the vjps while parameterizing (forward pass)
for fn, handles in tuples:
    params, vjp_fn, potential = jax.vjp(fn, *[h.params for h in handles], has_aux=True)
    final_potentials.append(potential.bind(params))
    final_vjp_and_handles.append((vjp_fn, handles))

# note: lambda goes from 0 to 1, 0 being fully-interacting and 1.0 being fully interacting.
for final_lamb in np.linspace(0, 1, 8):

    seed = 2020

    # note: the .impl() call at the end returns a pickle-able version of the
    #   wrapper function -- since contexts are not pickle-able -- which will
    #   be useful later in timemachine's multi-device parallelization strategy)
    # note: OpenMM unit system used throughout
    #   (temperature: kelvin, timestep: picosecond, collision_rate: picosecond^-1)
    intg = LangevinIntegrator(300.0, 1.5e-3, 1.0, combined_masses, seed).impl()

    x0 = combined_coords
    v0 = np.zeros_like(x0)

    u_impls = []

    for bp in final_potentials:
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
    ctxt = custom_ops.Context(x0, v0, box, intg, u_impls)

    # initial insertion step to remove clashes, note that for production we probably
    # want to adjust the box size slighty to accomodate for a uniform water density
    # since we're doing an NVT simulation.
    for lamb_idx, lamb in enumerate(np.linspace(1.0, final_lamb, 1000)):
        # note: unlike in OpenMM, .step() accepts a float (for lambda) and does
        #   not accept an integer (for n_steps) -- .step() takes 1 step at a time
        ctxt.step(lamb)

    # print("insertion energy", ctxt._get_u_t_minus_1())

    # note: these 5000 steps are "equilibration", before we attach a reporter /
    #   "observable" to the context and start running "production"
    for _ in range(5000):
        ctxt.step(final_lamb)

    # print("equilibrium energy", ctxt._get_u_t_minus_1())

    # TODO: what was the second argument -- reporting interval in steps?
    du_dl_obs = custom_ops.AvgPartialUPartialLambda(u_impls, 5)

    du_dps = []
    for ui in u_impls:
        du_dp_obs = custom_ops.AvgPartialUPartialParam(ui, 5)
        ctxt.add_observable(du_dp_obs)
        du_dps.append(du_dp_obs)

    ctxt.add_observable(du_dl_obs)

    for _ in range(10000):
        ctxt.step(final_lamb)

    # print("final energy", ctxt._get_u_t_minus_1())

    # print vector jacobian products back into the forcefield derivative
    for du_dp_obs, vjp_and_handles in zip(du_dps, final_vjp_and_handles):
        du_dp = du_dp_obs.avg_du_dp()

        if vjp_and_handles:
            vjp_fn, handles = vjp_and_handles
            du_df = vjp_fn(du_dp)  # vjp into forcefield derivatives
            for f_grad, h in zip(du_df, handles):
                print("handle:", type(h).__name__)
                for s, vv in zip(h.smirks, f_grad):
                    if np.any(vv) != 0:
                        print(s, vv)
                print("\n")

    du_dl = du_dl_obs.avg_du_dl()

    print("lambda", final_lamb, "du_dl", du_dl)
