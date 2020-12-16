# relative vacuum free energy

import numpy as np

from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import AllChem
from fe import pdb_writer
from fe import topology
from md import builders
from md import minimizer

from timemachine.lib import potentials, custom_ops
from timemachine.lib import LangevinIntegrator

import functools
import jax

from ff import Forcefield

from ff.handlers import nonbonded, bonded, openmm_deserializer

from ff.handlers.deserialize import deserialize_handlers

from fe import relative


# res = rdFMCS.FindMCS(
#     [romol_a, romol_b],
#     completeRingsOnly=True,
#     matchValences=True,
#     ringMatchesRingOnly=True
# )

# core = np.stack([
#     np.arange(romol_a.GetNumAtoms()),
#     np.arange(romol_b.GetNumAtoms())
# ], axis=1)

# # core = core[:-1] # remove the H-F mapping, become 4D
# # core = np.concatenate([core[:7], core[8:]])

# gst = topology.SingleTopology(romol_a, romol_b, core, ff)

# # setup the parameter handlers for the ligand
# bonded_tuples = [
#     [gst.parameterize_harmonic_bond, ff.hb_handle],
#     [gst.parameterize_harmonic_angle, ff.ha_handle],
#     [gst.parameterize_proper_torsion, ff.pt_handle],
#     [gst.parameterize_improper_torsion, ff.it_handle]
# ]

# # instantiate the vjps while parameterizing (forward pass)
# for fn, handle in bonded_tuples:
#     (src_params, dst_params, uni_params), vjp_fn, (src_potential, dst_potential, uni_potential) = jax.vjp(fn, handle.params, has_aux=True)
#     final_potentials.append([src_potential.bind(src_params), dst_potential.bind(dst_params), uni_potential.bind(uni_params)])
#     final_vjp_and_handles.append((vjp_fn, handle))


# nb_params, vjp_fn, nb_potential = jax.vjp(gst.parameterize_nonbonded, ff.q_handle.params, ff.lj_handle.params, has_aux=True)
# final_potentials.append([nb_potential.bind(nb_params)])
# final_vjp_and_handles.append([vjp_fn])


# # note: lambda goes from 0 to 1, 0 being fully-interacting and 1.0 being fully interacting.
# for lamb_idx, final_lamb in enumerate(np.linspace(0, 1, 64)):

#     # write some conformations into this PDB file
#     writer = pdb_writer.PDBWriter([romol_a, romol_b], "debug_"+str(lamb_idx)+".pdb")

#     seed = 2020

#     combined_masses = np.mean(gst.interpolate_params(ligand_masses_a, ligand_masses_b), axis=0)

#     intg = LangevinIntegrator(
#         300.0,
#         1.5e-3,
#         1.0,
#         combined_masses,
#         seed
#     ).impl()

#     # note the order in which the coordinates are concatenated in this step --
#     #   in a later step we will need to combine recipes in the same order
#     src_conf, dst_conf = gst.interpolate_params(ligand_coords_a, ligand_coords_b)
#     combined_coords = np.mean(gst.interpolate_params(ligand_coords_a, ligand_coords_b), axis=0)

#     x0 = combined_coords
#     v0 = np.zeros_like(x0)

#     u_impls = []
#     bonded_impls = []
#     nonbonded_impls = []

#     for bps in final_potentials:
#         for bp in bps:
#             impl = bp.bound_impl(np.float32)
#             if isinstance(bp, potentials.InterpolatedPotential):
#                 nonbonded_impls.append(impl)
#             elif isinstance(bp, potentials.LambdaPotential):
#                 bonded_impls.append(impl)
#             u_impls.append(impl)

#     box = np.eye(3) * 100.0

#     # context components: positions, velocities, box, integrator, energy fxns
#     ctxt = custom_ops.Context(
#         x0,
#         v0,
#         box,
#         intg,
#         u_impls
#     )

#     # (ytz): split the coordinates back out
#     def unjank_x(x):
#         c_x = x
#         a_x = [None]*romol_a.GetNumAtoms()
#         b_x = [None]*romol_b.GetNumAtoms()
#         for a, c in enumerate(gst.a_to_c):
#             a_x[a] = c_x[c]
#         for b, c in enumerate(gst.b_to_c):
#             b_x[b] = c_x[c]

#         return np.concatenate([
#             a_x,
#             b_x
#         ])

#     for step, lamb in enumerate(np.linspace(1.0, final_lamb, 1000)):
#         if step % 500 == 0:
#             writer.write_frame(unjank_x(ctxt.get_x_t())*10)
#         ctxt.step(lamb)

#     print("insertion energy", ctxt.get_u_t())

#     for step in range(5000):
#         if step % 500 == 0:
#             writer.write_frame(unjank_x(ctxt.get_x_t())*10)
#         ctxt.step(final_lamb)

#     print("equilibrium energy", ctxt.get_u_t())

#     bonded_du_dl_obs = custom_ops.AvgPartialUPartialLambda(bonded_impls, 5)
#     nonbonded_du_dl_obs = custom_ops.AvgPartialUPartialLambda(nonbonded_impls, 5)

#     ctxt.add_observable(bonded_du_dl_obs)
#     ctxt.add_observable(nonbonded_du_dl_obs)

#     du_dps = []
#     for ui in u_impls:
#         du_dp_obs = custom_ops.AvgPartialUPartialParam(ui, 5)
#         ctxt.add_observable(du_dp_obs)
#         du_dps.append(du_dp_obs)

#     for _ in range(20000):
#         if step % 500 == 0:
#             writer.write_frame(unjank_x(ctxt.get_x_t())*10)
#         ctxt.step(final_lamb)

#     writer.close()

#     # du_dl = du_dl_obs.avg_du_dl()

#     print("lambda", final_lamb, "bonded du_dl", bonded_du_dl_obs.avg_du_dl(), "nonbonded du_dl", nonbonded_du_dl_obs.avg_du_dl())


if __name__ == "__main__":

    suppl = Chem.SDMolSupplier('tests/data/benzene_flourinated.sdf', removeHs=False)
    all_mols = [x for x in suppl]

    mol_a = all_mols[0]
    mol_b = all_mols[1]

    ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_ccc.py').read())
    ff = Forcefield(ff_handlers)

    core = np.stack([
        np.arange(mol_a.GetNumAtoms()),
        np.arange(mol_b.GetNumAtoms())
    ], axis=1)

    rfe = relative.RelativeFreeEnergy(mol_a, mol_b, core, ff)

    # lamb = 0.1
    for lamb in [0.0, 0.4, 0.6, 1.0]:
        print(rfe.vacuum_edge(lamb, 10000, 20000))