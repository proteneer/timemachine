# # test protocols for setting up relative binding free energy calculations.

# import numpy as np
# from rdkit import Chem
# from rdkit.Chem import AllChem

# from timemachine.lib import potentials, custom_ops
# from timemachine.lib import LangevinIntegrator

# from ff.handlers import openmm_deserializer
# from ff.handlers.deserialize import deserialize_handlers


# from fe import pdb_writer # tbd: migrate to md/ folder
# from fe import rbfe
# from md import Recipe
# from md import builders

# def minimize(r_host, r_ligand, x0, box):
#     """
#     Minimize a system via 4d insertion.

#     Parameters
#     ----------
#     r_host: Recipe
#         host recipe

#     r_ligand: Recipe
#         ligand recipe

#     x0: np.ndarray float64 [N,3]
#         initial geometry

#     box: np.ndarray float64 [3,3]
#         periodic box

#     """
#     r_combined = r_host.combine(r_ligand)

#     host_atom_idxs = np.arange(len(r_host.masses))
#     ligand_atom_idxs = np.arange(len(r_ligand.masses)) + len(r_host.masses)

#     rbfe.set_nonbonded_lambda_idxs(r_combined, ligand_atom_idxs, 0, 1)

#     u_impls = []
#     for bp in r_combined.bound_potentials:
#         u_impls.append(bp.bound_impl(precision=np.float32))

#     seed = np.random.randint(np.iinfo(np.int32).max)

#     masses = np.concatenate([r_host.masses, r_ligand.masses])

#     intg = LangevinIntegrator(
#         300.0,
#         1.5e-3,
#         1.0,
#         masses,
#         seed
#     ).impl()

#     v0 = np.zeros_like(x0)

#     ctxt = custom_ops.Context(
#         x0,
#         v0,
#         box,
#         intg,
#         u_impls
#     )

#     steps = 500

#     lambda_schedule = np.linspace(0.35, 0.0, 500)
#     for lamb in lambda_schedule:
#         ctxt.step(lamb)

#     return ctxt.get_x_t()


# def get_romol_conf(mol):
#     conformer = mol.GetConformer(0)
#     guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
#     guest_conf = guest_conf/10 # from angstroms to nm
#     return np.array(guest_conf, dtype=np.float64)

# def test_stage_0():

#     benzene = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
#     phenol = Chem.AddHs(Chem.MolFromSmiles("Oc1ccccc1"))

#     AllChem.EmbedMolecule(benzene)
#     AllChem.EmbedMolecule(phenol)

#     ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_ccc.py').read())
#     r_benzene = Recipe.from_rdkit(benzene, ff_handlers)
#     r_phenol = Recipe.from_rdkit(phenol, ff_handlers)

#     r_combined = r_benzene.combine(r_phenol)

#     core_pairs = np.array([
#         [0,1],
#         [1,2],
#         [2,3],
#         [3,4],
#         [4,5]
#     ], dtype=np.int32)
#     core_pairs[:, 1] += benzene.GetNumAtoms()

#     b_idxs = np.arange(phenol.GetNumAtoms()) + benzene.GetNumAtoms()

#     com_k = 10.0
#     core_k = 200.0
#     rbfe.stage_0(r_combined, b_idxs, core_pairs, com_k, core_k)

#     centroid_count = 0
#     core_count = 0
#     nb_count = 0

#     for bp in r_combined.bound_potentials:
#         if isinstance(bp, potentials.LambdaPotential):
#             u_fn = bp.get_u_fn()
#             if isinstance(u_fn, potentials.CentroidRestraint):
#                 centroid_count += 1

#                 # (1-lambda)*u_fn
#                 np.testing.assert_equal(bp.get_multiplier(), -1.0)
#                 np.testing.assert_equal(bp.get_offset(), 1.0)

#                 np.testing.assert_equal(u_fn.get_a_idxs(), core_pairs[:, 0])
#                 np.testing.assert_equal(u_fn.get_b_idxs(), core_pairs[:, 1])

#             elif isinstance(u_fn, potentials.CoreRestraint):
#                 core_count += 1

#                 # lambda*u_fn
#                 np.testing.assert_equal(bp.get_multiplier(), 1.0)
#                 np.testing.assert_equal(bp.get_offset(), 0.0)

#                 np.testing.assert_equal(u_fn.get_idxs(), core_pairs)

#         elif isinstance(bp, potentials.Nonbonded):
#             nb_count += 1
#             test_plane_idxs = bp.get_lambda_plane_idxs()
#             ref_plane_idxs = np.zeros_like(test_plane_idxs)
#             ref_plane_idxs[benzene.GetNumAtoms():] = 1
#             np.testing.assert_array_equal(ref_plane_idxs, test_plane_idxs)

#             test_offset_idxs = bp.get_lambda_offset_idxs()
#             np.testing.assert_array_equal(test_offset_idxs, np.zeros_like(test_offset_idxs))

#         # test C++ side of things
#         bp.bound_impl(precision=np.float32)

#     assert nb_count == 1
#     assert centroid_count == 1
#     assert core_count == 1

# def test_stage_1():

#     benzene = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
#     phenol = Chem.AddHs(Chem.MolFromSmiles("Oc1ccccc1"))

#     AllChem.EmbedMolecule(benzene)
#     AllChem.EmbedMolecule(phenol)

#     ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_ccc.py').read())
#     r_benzene = Recipe.from_rdkit(benzene, ff_handlers)
#     r_phenol = Recipe.from_rdkit(phenol, ff_handlers)

#     r_combined = r_benzene.combine(r_phenol)

#     core_pairs = np.array([
#         [0,1],
#         [1,2],
#         [2,3],
#         [3,4],
#         [4,5]
#     ], dtype=np.int32)
#     core_pairs[:, 1] += benzene.GetNumAtoms()

#     a_idxs = np.arange(benzene.GetNumAtoms())
#     b_idxs = np.arange(phenol.GetNumAtoms()) + benzene.GetNumAtoms()

#     com_k = 10.0
#     core_k = 200.0
#     rbfe.stage_1(r_combined, a_idxs, b_idxs, core_pairs, core_k)

#     core_count = 0
#     nb_count = 0
#     for bp in r_combined.bound_potentials:
#         if isinstance(bp, potentials.LambdaPotential):
#             assert 0
#         elif isinstance(bp, potentials.CentroidRestraint):
#             assert 0
#         elif isinstance(bp, potentials.CoreRestraint):
#             core_count += 1
#             np.testing.assert_equal(bp.get_idxs(), core_pairs)
#         elif isinstance(bp, potentials.Nonbonded):

#             nb_count += 1

#             test_plane_idxs = bp.get_lambda_plane_idxs()
#             np.testing.assert_array_equal(test_plane_idxs, np.zeros_like(test_plane_idxs))

#             test_offset_idxs = bp.get_lambda_offset_idxs()
#             ref_offset_idxs = np.zeros_like(test_offset_idxs)
#             ref_offset_idxs[benzene.GetNumAtoms():] = 1
#             np.testing.assert_array_equal(ref_offset_idxs, test_offset_idxs)

#             # ensure exclusions are added correctly
#             combined_idxs = bp.get_exclusion_idxs()

#             left_idxs = r_benzene.bound_potentials[-1].get_exclusion_idxs()
#             right_idxs = r_phenol.bound_potentials[-1].get_exclusion_idxs()

#             n_left = benzene.GetNumAtoms()
#             n_right = phenol.GetNumAtoms()

#             assert (len(left_idxs) + len(right_idxs) + n_left * n_right) == len(combined_idxs)
#             test_set = np.sort(combined_idxs[len(left_idxs) + len(right_idxs):])

#             ref_set = []
#             for i in range(n_left):
#                 for j in range(n_right):
#                     ref_set.append((i, j+n_left))
#             ref_set = np.sort(np.array(ref_set, dtype=np.int32))

#             np.testing.assert_array_equal(test_set, ref_set)

#         bp.bound_impl(precision=np.float32)

#     assert nb_count == 1
#     assert core_count == 1


# def test_water_system_stage_0():

#     benzene = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1")) # a
#     phenol = Chem.AddHs(Chem.MolFromSmiles("Oc1ccccc1")) # b

#     AllChem.EmbedMolecule(benzene)
#     AllChem.EmbedMolecule(phenol)

#     ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_ccc.py').read())
#     r_benzene = Recipe.from_rdkit(benzene, ff_handlers)
#     r_phenol = Recipe.from_rdkit(phenol, ff_handlers)

#     print("combinining benzene with phenol")
#     r_combined = r_benzene.combine(r_phenol)


#     core_pairs = np.array([
#         [0,1],
#         [1,2],
#         [2,3],
#         [3,4],
#         [4,5]
#     ], dtype=np.int32)
#     core_pairs[:, 1] += benzene.GetNumAtoms()

#     b_idxs = np.arange(phenol.GetNumAtoms()) + benzene.GetNumAtoms()

#     com_k = 100.0
#     core_k = 200.0
#     rbfe.stage_0(r_combined, b_idxs, core_pairs, com_k, core_k)

#     system, host_coords, box, topology = builders.build_water_system(4.0)

#     # writer = pdb_writer.PDBWriter([topology, benzene, phenol], 'debug.pdb')

#     r_host = Recipe.from_openmm(system)
#     r_final = r_host.combine(r_combined)

#     # minimize coordinates of host + ligand A
#     ha_coords = np.concatenate([
#         host_coords,
#         get_romol_conf(benzene)
#     ])

#     ha_coords = minimize(r_host, r_benzene, ha_coords, box)

#     # writer.write_frame(x0*10)

#     # production run at various values of lambda
#     avg_du_dls = []
#     for lamb in [0.0, 0.5, 1.0]:
#         print("production run with lamb", lamb)
#         u_impls = []
#         for bp in r_final.bound_potentials:
#             # print(bp)
#             u_impls.append(bp.bound_impl(precision=np.float32))

#         seed = np.random.randint(np.iinfo(np.int32).max)

#         masses = np.concatenate([r_host.masses, r_benzene.masses, r_phenol.masses])

#         intg = LangevinIntegrator(
#             300.0,
#             1.5e-3,
#             1.0,
#             masses,
#             seed
#         ).impl()

#         x0 = np.concatenate([
#             ha_coords,
#             get_romol_conf(phenol)
#         ])

#         v0 = np.zeros_like(x0)

#         ctxt = custom_ops.Context(
#             x0,
#             v0,
#             box,
#             intg,
#             u_impls
#         )

#         # equilibration
#         for step in range(15000):
#             ctxt.step(lamb)

#         # writer.write_frame(ctxt.get_x_t()*10)
#         # writer.close()

#         du_dl_obs = custom_ops.AvgPartialUPartialLambda(u_impls, 10)
#         ctxt.add_observable(du_dl_obs)

#         # add observable for <du/dl>
#         for step in range(30000):
#             ctxt.step(lamb)

#         print(du_dl_obs.avg_du_dl())

#         avg_du_dls.append(du_dl_obs.avg_du_dl())

#         assert np.any(np.abs(ctxt.get_x_t()) > 100) == False
#         assert np.any(np.isnan(ctxt.get_x_t())) == False
#         assert np.any(np.isinf(ctxt.get_x_t())) == False

#     assert np.all(np.diff(avg_du_dls) < 0)
#     assert avg_du_dls[0] > avg_du_dls[1]
#     assert avg_du_dls[1] > avg_du_dls[2]


# def test_water_system_stage_1():

#     benzene = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1")) # a
#     phenol = Chem.AddHs(Chem.MolFromSmiles("Oc1ccccc1")) # b

#     AllChem.EmbedMolecule(benzene)
#     AllChem.EmbedMolecule(phenol)

#     ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_ccc.py').read())
#     r_benzene = Recipe.from_rdkit(benzene, ff_handlers)
#     r_phenol = Recipe.from_rdkit(phenol, ff_handlers)

#     print("combinining benzene with phenol")
#     r_combined = r_benzene.combine(r_phenol)


#     core_pairs = np.array([
#         [0,1],
#         [1,2],
#         [2,3],
#         [3,4],
#         [4,5]
#     ], dtype=np.int32)
#     core_pairs[:, 1] += benzene.GetNumAtoms()

#     a_idxs = np.arange(benzene.GetNumAtoms())
#     b_idxs = np.arange(phenol.GetNumAtoms()) + benzene.GetNumAtoms()

#     core_k = 200.0

#     rbfe.stage_1(r_combined, a_idxs, b_idxs, core_pairs, core_k)

#     system, host_coords, box, topology = builders.build_water_system(4.0)

#     # writer = pdb_writer.PDBWriter([topology, benzene, phenol], 'debug.pdb')

#     r_host = Recipe.from_openmm(system)
#     r_final = r_host.combine(r_combined)

#     # minimize coordinates of host + ligand A
#     ha_coords = np.concatenate([
#         host_coords,
#         get_romol_conf(benzene)
#     ])

#     ha_coords = minimize(r_host, r_benzene, ha_coords, box)

#     # assert 0
#     # production run at various values of lambda
#     avg_du_dls = []
#     for lamb in [0.0, 0.25, 0.5, 1.5]:
#         print("production run with lamb", lamb)
#         u_impls = []
#         for bp in r_final.bound_potentials:
#             # print(bp)
#             u_impls.append(bp.bound_impl(precision=np.float32))

#         seed = np.random.randint(np.iinfo(np.int32).max)

#         masses = np.concatenate([r_host.masses, r_benzene.masses, r_phenol.masses])

#         intg = LangevinIntegrator(
#             300.0,
#             1.5e-3,
#             1.0,
#             masses,
#             seed
#         ).impl()

#         x0 = np.concatenate([
#             ha_coords,
#             get_romol_conf(phenol)
#         ])


#         # writer.write_frame(x0*10)
#         # writer.close()


#         v0 = np.zeros_like(x0)

#         ctxt = custom_ops.Context(
#             x0,
#             v0,
#             box,
#             intg,
#             u_impls
#         )

#         # secondary minimization

#         for l in np.linspace(0.35, lamb, 500):
#             # print(ctxt.get_u_t())
#             ctxt.step(l)

#         # equilibration
#         for step in range(20000):
#             # print(step, ctxt.get_u_t())
#             ctxt.step(lamb)

#         # print(ctxt.get_x_t())
#         # assert 0
#         # writer.write_frame(ctxt.get_x_t()*10)
#         # writer.close()

#         du_dl_obs = custom_ops.AvgPartialUPartialLambda(u_impls, 10)
#         ctxt.add_observable(du_dl_obs)

#         # add observable for <du/dl>
#         for step in range(50000):
#             ctxt.step(lamb)

#         print(du_dl_obs.avg_du_dl())

#         avg_du_dls.append(du_dl_obs.avg_du_dl())

#         assert np.any(np.abs(ctxt.get_x_t()) > 100) == False
#         assert np.any(np.isnan(ctxt.get_x_t())) == False
#         assert np.any(np.isinf(ctxt.get_x_t())) == False

#     assert np.all(np.abs(avg_du_dls[0]) < 1e-3)
#     assert np.all(np.abs(avg_du_dls[-1]) < 1e-3)
#     for du_dl in avg_du_dls[1:-1]:
#         assert np.all(du_dl > 1.0)
