# relative hydration free energy

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from fe import pdb_writer
from fe import topology
from fe.utils import get_romol_conf
from md import builders
from md import minimizer

from timemachine.lib import potentials, custom_ops
from timemachine.lib import LangevinIntegrator

import functools
import jax

from ff import Forcefield

from ff.handlers import nonbonded, bonded, openmm_deserializer

from ff.handlers.deserialize import deserialize_handlers


# construct an RDKit molecule of aspirin
# note: not using OpenFF Molecule because want to avoid the dependency (YTZ?)
romol_a = Chem.AddHs(Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O"))
romol_b = Chem.AddHs(Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)OC"))

ligand_masses_a = [a.GetMass() for a in romol_a.GetAtoms()]
ligand_masses_b = [a.GetMass() for a in romol_b.GetAtoms()]

# generate conformers
AllChem.EmbedMolecule(romol_a)
AllChem.EmbedMolecule(romol_b)

# extract the 0th conformer
ligand_coords_a = get_romol_conf(romol_a)
ligand_coords_b = get_romol_conf(romol_b)

# construct a 4-nanometer water box (from openmmtools approach: selecting out
#   of a large pre-equilibrated water box snapshot)
system, host_coords, box, omm_topology = builders.build_water_system(4.0)

# padding to avoid jank
box = box + np.eye(3)*0.1

host_bps, host_masses = openmm_deserializer.deserialize_system(system, cutoff=1.2)

combined_masses = np.concatenate([host_masses, ligand_masses_a, ligand_masses_b])


# minimize coordinates

# note: .py file rather than .offxml file
# note: _ccc suffix means "correctable charge corrections"
ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_ccc.py').read())
ff = Forcefield(ff_handlers)

# for RHFE we need to insert the reference ligand first, before inserting the
# decoupling ligand
minimized_coords = minimizer.minimize_host_4d([romol_a], system, host_coords, ff, box)

# note the order in which the coordinates are concatenated in this step --
#   in a later step we will need to combine recipes in the same order
# combined_coords = np.concatenate([host_coords, ligand_coords_a, ligand_coords_b])
combined_coords = np.concatenate([minimized_coords, ligand_coords_b])

num_host_atoms = host_coords.shape[0]

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

gdt = topology.DualTopology(romol_a, romol_b, ff)
hgt = topology.HostGuestTopology(host_p, gdt)

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

# add centroid restraint

restraint_a_idxs = np.arange(romol_a.GetNumAtoms()) + num_host_atoms
restraint_b_idxs = np.arange(romol_b.GetNumAtoms()) + num_host_atoms + romol_a.GetNumAtoms()

restraint = potentials.CentroidRestraint(
    restraint_a_idxs.astype(np.int32),
    restraint_b_idxs.astype(np.int32),
    combined_masses,
    100.0,
    0.0
).bind([])

final_potentials.append(restraint)
final_vjp_and_handles.append(None)

# note: lambda goes from 0 to 1, 0 being fully-interacting and 1.0 being fully interacting.
for lamb_idx, final_lamb in enumerate(np.linspace(1, 0, 8)):


    # write some conformations into this PDB file
    writer = pdb_writer.PDBWriter([omm_topology, romol_a, romol_b], "debug_"+str(lamb_idx)+".pdb")

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
        combined_masses,
        seed
    ).impl()

    x0 = combined_coords
    v0 = np.zeros_like(x0)

    u_impls = []

    for bp in final_potentials:
        u_impls.append(bp.bound_impl(np.float32))

    # context components: positions, velocities, box, integrator, energy fxns
    ctxt = custom_ops.Context(
        x0,
        v0,
        box,
        intg,
        u_impls
    )

    for step, lamb in enumerate(np.linspace(1.0, final_lamb, 1000)):
        if step % 500 == 0:
            writer.write_frame(ctxt.get_x_t()*10)
        ctxt.step(lamb)

    # print("insertion energy", ctxt._get_u_t_minus_1())

    # note: these 5000 steps are "equilibration", before we attach a reporter /
    #   "observable" to the context and start running "production"
    for step in range(5000):
        if step % 500 == 0:
            writer.write_frame(ctxt.get_x_t()*10)
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

    for _ in range(20000):
        if step % 500 == 0:
            writer.write_frame(ctxt.get_x_t()*10)
        ctxt.step(final_lamb)

    writer.close()

    # print("final energy", ctxt._get_u_t_minus_1())

    # print vector jacobian products back into the forcefield derivative
    for du_dp_obs, vjp_and_handles in zip(du_dps, final_vjp_and_handles):
        du_dp = du_dp_obs.avg_du_dp()

        if vjp_and_handles:
            vjp_fn, handles = vjp_and_handles
            du_df = vjp_fn(du_dp) # vjp into forcefield derivatives
            for f_grad, h in zip(du_df, handles):
                print("handle:", type(h).__name__)
                for s, vv in zip(h.smirks, f_grad):
                    if np.any(vv) != 0:
                        print(s, vv)
                print("\n")

    du_dl = du_dl_obs.avg_du_dl()

    print("lambda", final_lamb, "du_dl", du_dl)
