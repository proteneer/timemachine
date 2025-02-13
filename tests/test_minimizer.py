from time import time

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from timemachine.constants import MAX_FORCE_NORM
from timemachine.fe.free_energy import AbsoluteFreeEnergy
from timemachine.fe.model_utils import get_vacuum_val_and_grad_fn
from timemachine.fe.topology import BaseTopology
from timemachine.fe.utils import get_romol_conf, read_sdf, read_sdf_mols_by_name
from timemachine.ff import Forcefield
from timemachine.md import builders, minimizer
from timemachine.md.barostat.utils import compute_box_volume
from timemachine.md.minimizer import equilibrate_host_barker, make_host_du_dx_fxn
from timemachine.potentials import NonbondedPairList
from timemachine.potentials.jax_utils import distance_on_pairs, idxs_within_cutoff
from timemachine.utils import path_to_internal_file


@pytest.mark.parametrize(
    "pdb_path, sdf_path, mol_a_name, mol_b_name, run_one_test",
    [
        pytest.param(
            path_to_internal_file("timemachine.testsystems.data", "hif2a_nowater_min.pdb"),
            path_to_internal_file("timemachine.testsystems.data", "ligands_40.sdf"),
            "43",
            "234",
            False,
            marks=pytest.mark.nightly(reason="slow"),
        ),
        pytest.param(
            path_to_internal_file("timemachine.datasets.fep_benchmark.pfkfb3", "6hvi_prepared.pdb"),
            path_to_internal_file("timemachine.datasets.fep_benchmark.pfkfb3", "ligands.sdf"),
            "20",
            "43",
            False,
            marks=pytest.mark.nightly(reason="slow"),
        ),
        pytest.param(
            path_to_internal_file("timemachine.datasets.fep_benchmark.pfkfb3", "6hvi_prepared.pdb"),
            path_to_internal_file("timemachine.datasets.fep_benchmark.pfkfb3", "ligands.sdf"),
            "41",
            "43",
            False,
            marks=pytest.mark.nightly(reason="slow"),
        ),
        pytest.param(
            path_to_internal_file("timemachine.datasets.fep_benchmark.pfkfb3", "6hvi_prepared.pdb"),
            path_to_internal_file("timemachine.datasets.fep_benchmark.pfkfb3", "ligands.sdf"),
            "34",
            "37",
            False,
            marks=pytest.mark.nightly(reason="slow"),
        ),
        pytest.param(
            path_to_internal_file("timemachine.datasets.fep_benchmark.pfkfb3", "6hvi_prepared.pdb"),
            path_to_internal_file("timemachine.datasets.fep_benchmark.pfkfb3", "ligands.sdf"),
            "26",
            "37",
            False,
            marks=pytest.mark.nightly(reason="slow"),
        ),
        pytest.param(
            path_to_internal_file("timemachine.testsystems.data", "hif2a_nowater_min.pdb"),
            path_to_internal_file("timemachine.testsystems.data", "ligands_40.sdf"),
            "43",
            "234",
            True,
        ),
    ],
)
def test_fire_minimize_host_protein(pdb_path, sdf_path, mol_a_name, mol_b_name, run_one_test):
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    with sdf_path as ligand_path:
        mols_by_name = read_sdf_mols_by_name(ligand_path)
    mol_a = mols_by_name[mol_a_name]
    mol_b = mols_by_name[mol_b_name]

    with pdb_path as host_path:
        for mols in [[mol_a], [mol_b], [mol_a, mol_b]]:
            host_config = builders.build_protein_system(str(host_path), ff.protein_ff, ff.water_ff, mols=mols)
            x_host = minimizer.fire_minimize_host(mols, host_config, ff)
            assert x_host.shape == host_config.conf.shape


def test_fire_minimize_host_solvent():
    ff = Forcefield.load_default()
    with path_to_internal_file("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        all_mols = read_sdf(path_to_ligand)
    mol_a = all_mols[1]
    mol_b = all_mols[4]

    for mols in [[mol_a], [mol_b], [mol_a, mol_b]]:
        host_config = builders.build_water_system(4.0, ff.water_ff, mols=mols)
        x_host = minimizer.fire_minimize_host(mols, host_config, ff)
        assert x_host.shape == host_config.conf.shape


@pytest.mark.parametrize("host_name", ["solvent", pytest.param("complex", marks=pytest.mark.nightly(reason="slow"))])
@pytest.mark.parametrize("mol_pair", [("20", "43")])
def test_pre_equilibrate_host_pfkfb3(host_name, mol_pair):
    ff = Forcefield.load_default()
    mol_a_name, mol_b_name = mol_pair
    with path_to_internal_file("timemachine.datasets.fep_benchmark.pfkfb3", "ligands.sdf") as path_to_ligand:
        mols_by_name = read_sdf_mols_by_name(path_to_ligand)
    mol_a = mols_by_name[mol_a_name]
    mol_b = mols_by_name[mol_b_name]
    mols = [mol_a, mol_b]
    if host_name == "solvent":
        host_config = builders.build_water_system(4.0, ff.water_ff, mols=mols)
    else:
        with path_to_internal_file("timemachine.datasets.fep_benchmark.pfkfb3", "6hvi_prepared.pdb") as pdb_path:
            host_config = builders.build_protein_system(str(pdb_path), ff.protein_ff, ff.water_ff, mols=mols)
    x_host, x_box = minimizer.pre_equilibrate_host(mols, host_config, ff)
    assert x_host.shape == host_config.conf.shape
    box_vol_before = compute_box_volume(host_config.box)
    box_vol_after = compute_box_volume(x_box)
    # assert box_vol_after < box_vol_before
    assert box_vol_after < 1.1 * box_vol_before


def test_fire_minimize_host_adamantane():
    """With cagey molecules, can trap water molecules inside of them. Verify that molecule can be minimized
    in water without issue"""
    ff = Forcefield.load_default()
    mol = Chem.AddHs(Chem.MolFromSmiles("C1C3CC2CC(CC1C2)C3"))
    AllChem.EmbedMolecule(mol, randomSeed=2024)
    # If don't delete the relevant water this minimization fails
    host_config = builders.build_water_system(4.0, ff.water_ff, mols=[mol])
    x_host = minimizer.fire_minimize_host([mol], host_config, ff)
    assert x_host.shape == host_config.conf.shape


@pytest.mark.nightly(reason="Currently not used in practice")
def test_equilibrate_host_barker():
    ff = Forcefield.load_default()
    with path_to_internal_file("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        all_mols = read_sdf(path_to_ligand)
    mol_a = all_mols[1]
    mol_b = all_mols[4]

    with path_to_internal_file("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as path_to_pdb:
        host_config = builders.build_protein_system(str(path_to_pdb), ff.protein_ff, ff.water_ff, mols=[mol_a, mol_b])

    # TODO[requirements-gathering]:
    #   do we really want to minimize here ("equilibrate to temperature ~= 0"),
    #   or do we want to equilibrate ("equilibrate to temperature = 300")?
    #   and if we run MD @ temperature = 300 initialized from x_host, how long does it take to "heat back up"?
    room_temperature = 300.0
    zero_temperature = 0.0

    # equilibrate_host_barker and fire_minimize_host methods will throw if the minimization failed

    setups = {"A and B simultaneously": [mol_a, mol_b], "A alone": [mol_a], "B alone": [mol_b]}

    for key in setups:
        print(f"minimizing host given {key}...")
        mols = setups[key]
        host_du_dx_fxn = make_host_du_dx_fxn(mols, host_config, ff)

        print(f"using unadjusted Barker proposal @ temperature = {room_temperature} K...")
        t0 = time()
        x_host = equilibrate_host_barker(mols, host_config, ff, temperature=room_temperature)
        assert x_host.shape == host_config.conf.shape
        t1 = time()
        max_frc = np.linalg.norm(host_du_dx_fxn(x_host), axis=-1).max()
        print(f"\tforce norm after room-temperature equilibration: {max_frc:.3f} kJ/mol / nm")
        print(f"\tmax distance traveled = {np.linalg.norm(np.array(host_config.conf) - x_host, axis=-1).max():.3f} nm")
        print(f"\tdone in {(t1 - t0):.3f} s")

        print(f"using unadjusted Barker proposal @ temperature = {zero_temperature} K...")
        t0 = time()
        x_host = equilibrate_host_barker(mols, host_config, ff, temperature=zero_temperature)
        assert x_host.shape == host_config.conf.shape
        t1 = time()

        max_frc = np.linalg.norm(host_du_dx_fxn(x_host), axis=-1).max()

        print(f"\tforce norm after low-temperature 'equilibration': {max_frc:.3f} kJ/mol / nm")
        print(f"\tmax distance traveled = {np.linalg.norm(np.array(host_config.conf) - x_host, axis=-1).max():.3f} nm")
        print(f"\tdone in {(t1 - t0):.3f} s")


@pytest.mark.parametrize(
    "minimizer_config",
    [
        minimizer.FireMinimizationConfig(100),
        minimizer.ScipyMinimizationConfig("BFGS"),
        minimizer.ScipyMinimizationConfig("L-BFGS-B"),
    ],
)
def test_local_minimize_water_box(minimizer_config):
    """
    Test that we can locally relax a box of water by selecting some random indices.
    """
    ff = Forcefield.load_default()
    host_config = builders.build_water_system(4.0, ff.water_ff)
    box0 = host_config.box
    box0 += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes at the boundary
    host_fns = host_config.host_system.get_U_fns()
    x0 = host_config.conf
    val_and_grad_fn = minimizer.get_val_and_grad_fn(host_fns, box0)

    free_idxs = [0, 2, 3, 6, 7, 9, 15, 16]
    frozen_idxs = set(range(len(x0))).difference(set(free_idxs))
    frozen_idxs = list(frozen_idxs)

    u_init, g_init = val_and_grad_fn(x0)

    x_opt = minimizer.local_minimize(x0, box0, val_and_grad_fn, free_idxs, minimizer_config)

    np.testing.assert_array_equal(x0[frozen_idxs], x_opt[frozen_idxs])
    assert np.linalg.norm(x0[free_idxs] - x_opt[free_idxs]) > 0.01

    # Verify that the value and grad return the exact same result even after
    # being used for minimization
    u_init_test, g_init_test = val_and_grad_fn(x0)
    assert u_init == u_init_test
    np.testing.assert_array_equal(g_init, g_init_test)


@pytest.mark.parametrize("seed", [2024])
@pytest.mark.parametrize(
    "minimizer_config",
    [
        minimizer.FireMinimizationConfig(100),
        minimizer.ScipyMinimizationConfig("BFGS"),
        minimizer.ScipyMinimizationConfig("L-BFGS-B"),
    ],
)
def test_local_minimize_restrained_subset(seed, minimizer_config):
    """
    Test that we can minimize systems and only restrain subsets of the atoms.
    """
    rng = np.random.default_rng(seed)
    ff = Forcefield.load_default()

    host_config = builders.build_water_system(4.0, ff.water_ff)
    host_fns = host_config.host_system.get_U_fns()
    x0 = host_config.conf
    box0 = host_config.box
    box0 += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes at the boundary

    val_and_grad_fn = minimizer.get_val_and_grad_fn(host_fns, box0)

    free_idxs = rng.choice(np.arange(len(x0), dtype=np.int32), size=128, replace=False)
    frozen_idxs = set(range(len(x0))).difference(set(free_idxs))
    frozen_idxs = list(frozen_idxs)

    with pytest.raises(AssertionError, match="Restraint k be greater than 0.0 if restrained indices provided"):
        minimizer.local_minimize(x0, box0, val_and_grad_fn, free_idxs, minimizer_config, restrained_idxs=frozen_idxs)

    # Set a large k, to ensure movement of restrained idxs is minimal
    k = 500_000.0

    with pytest.raises(AssertionError, match="Restrained indices must be a subset of local indices"):
        minimizer.local_minimize(
            x0, box0, val_and_grad_fn, free_idxs, minimizer_config, restraint_k=k, restrained_idxs=frozen_idxs
        )

    restrained_idxs = rng.choice(free_idxs, replace=False, size=len(free_idxs) // 2)
    unrestrained_idxs = np.array(list(set(free_idxs).difference(restrained_idxs)))

    x_opt = minimizer.local_minimize(
        x0, box0, val_and_grad_fn, free_idxs, minimizer_config, restraint_k=k, restrained_idxs=restrained_idxs
    )

    np.testing.assert_array_equal(x0[frozen_idxs], x_opt[frozen_idxs])
    # All free atoms should have moved
    assert np.linalg.norm(x0[free_idxs] - x_opt[free_idxs]) > 0.0
    # Restrained atoms should have moved very slightly
    assert np.linalg.norm(x0[restrained_idxs] - x_opt[restrained_idxs]) < 0.011
    # Unrestrained atoms should have moved more
    assert np.linalg.norm(x0[unrestrained_idxs] - x_opt[unrestrained_idxs]) > 0.011


@pytest.mark.parametrize("seed", [2024])
@pytest.mark.parametrize(
    "minimizer_config",
    [
        minimizer.FireMinimizationConfig(100),
        minimizer.ScipyMinimizationConfig("BFGS"),
        minimizer.ScipyMinimizationConfig("L-BFGS-B"),
    ],
)
def test_local_minimize_restrained_waters_trigger_failure(seed, minimizer_config):
    """Construct a water box and attempt to minimize a benzene mol into it without removing clashy waters. Verify
    that if the water atoms are restrained that minimization fails and without restraints the minimization succeeds.
    """
    benzene = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    AllChem.EmbedMolecule(benzene, randomSeed=seed)

    ff = Forcefield.load_default()

    # Use non-zero lambda to ensure forces are large to start
    lamb = 0.1

    # Setup a water box without a void for the mol
    host_config = builders.build_water_system(4.0, ff.water_ff)
    host_config.box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes at the boundary
    box = host_config.box
    host_coords = host_config.conf

    bt = BaseTopology(benzene, ff)
    afe = AbsoluteFreeEnergy(benzene, bt)
    unbound_potentials, sys_params, masses = afe.prepare_host_edge(ff, host_config, lamb)
    coords = afe.prepare_combined_coords(host_coords=host_coords)

    bps = [pot.bind(p) for pot, p in zip(unbound_potentials, sys_params)]
    val_and_grad_fn = minimizer.get_val_and_grad_fn(bps, box)
    # Forces should be beyond the max force norm to start
    _, g_init = val_and_grad_fn(coords)
    assert np.max(np.linalg.norm(g_init, axis=-1)) > MAX_FORCE_NORM

    ligand_idxs = np.arange(benzene.GetNumAtoms()) + len(host_coords)

    free_idxs = idxs_within_cutoff(coords, coords[ligand_idxs], box, cutoff=0.5).tolist()

    # Set a large k, making it difficult to move waters out of the way if restrained
    k = 500_000.0

    # Restraining all atoms should trigger a failure
    with pytest.raises(minimizer.MinimizationError):
        minimizer.local_minimize(coords, box, val_and_grad_fn, free_idxs, minimizer_config, restraint_k=k)

    unrestrained_idxs = np.array(list(set(free_idxs).difference(ligand_idxs)))

    # Only restraining the ligand should work, since water is free to move
    minimized_coords = minimizer.local_minimize(
        coords, box, val_and_grad_fn, free_idxs, minimizer_config, restraint_k=k, restrained_idxs=ligand_idxs
    )
    np.testing.assert_allclose(minimized_coords[ligand_idxs], coords[ligand_idxs], atol=0.005)

    # At least one water atom will need to have moved an angstrom to allow the ligand to be in the water
    assert np.any(np.linalg.norm(minimized_coords[unrestrained_idxs] - coords[unrestrained_idxs], axis=-1) > 0.1)


def test_local_minimize_water_box_with_bounds():
    """
    Test that we can locally relax a box of water using L-BFGS-B with bounds
    """
    ff = Forcefield.load_default()

    host_config = builders.build_water_system(4.0, ff.water_ff)
    host_config.box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes at the boundary
    box0 = host_config.box
    x0 = host_config.conf
    host_fns = host_config.host_system.get_U_fns()
    val_and_grad_fn = minimizer.get_val_and_grad_fn(host_fns, box0)

    free_idxs = [0, 2, 3, 6, 7, 9, 15, 16]
    frozen_idxs = set(range(len(x0))).difference(set(free_idxs))
    frozen_idxs = list(frozen_idxs)

    allowed_diff = 0.01

    # Set up the bounds
    lower_bounds = np.array(x0[free_idxs].reshape(-1)) - allowed_diff
    upper_bounds = np.array(x0[free_idxs].reshape(-1)) + allowed_diff
    bounds = list(zip(lower_bounds, upper_bounds))

    minimizer_config = minimizer.ScipyMinimizationConfig("L-BFGS-B", bounds=bounds)

    u_init, g_init = val_and_grad_fn(x0)

    x_opt = minimizer.local_minimize(x0, box0, val_and_grad_fn, free_idxs, minimizer_config)

    np.testing.assert_array_equal(x0[frozen_idxs], x_opt[frozen_idxs])
    assert np.linalg.norm(x0[free_idxs] - x_opt[free_idxs]) > 0.01
    assert np.all(x0[free_idxs] - allowed_diff <= x_opt[free_idxs])
    assert np.all(x_opt[free_idxs] <= x0[free_idxs] + allowed_diff)

    # Verify that the value and grad return the exact same result even after
    # being used for minimization
    u_init_test, g_init_test = val_and_grad_fn(x0)
    assert u_init == u_init_test
    np.testing.assert_array_equal(g_init, g_init_test)


@pytest.mark.nocuda
@pytest.mark.parametrize(
    "minimizer_config",
    [
        minimizer.FireMinimizationConfig(100),
        minimizer.ScipyMinimizationConfig("BFGS"),
        minimizer.ScipyMinimizationConfig("L-BFGS-B"),
    ],
)
@pytest.mark.parametrize("restraint_k", [0.0, 3_000.0])
def test_local_minimize_strained_ligand(minimizer_config, restraint_k):
    """
    Test that we can minimize a ligand in vacuum using local_minimize when the ligand is strained.
    """
    ff = Forcefield.load_default()

    mol = Chem.MolFromMolBlock(
        """minimization_failure
     RDKit          3D

 21 20  0  0  1  0            999 V2000
   -0.1708   -0.4217   -0.0397 C   0  0  0  0  0  0
    1.1755   -0.4328   -0.5490 C   0  0  0  0  0  0
    2.1284    0.7717   -0.6365 C   0  0  0  0  0  0
   -0.7315    0.8032    0.6201 C   0  0  0  0  0  0
   -0.3466    0.7426    2.1084 C   0  0  0  0  0  0
   -2.2581    0.7702    0.4329 C   0  0  0  0  0  0
   -0.1070    2.0269   -0.0722 C   0  0  0  0  0  0
    2.9723    0.5294    0.0097 H   0  0  0  0  0  0
    1.6881    1.6858   -0.2200 H   0  0  0  0  0  0
    0.7363    0.7175    2.2360 H   0  0  0  0  0  0
   -0.7241    1.6083    2.6542 H   0  0  0  0  0  0
   -0.7506   -0.1512    2.5852 H   0  0  0  0  0  0
   -2.5268    0.7646   -0.6241 H   0  0  0  0  0  0
   -2.6925   -0.1231    0.8832 H   0  0  0  0  0  0
   -2.7356    1.6373    0.8912 H   0  0  0  0  0  0
   -0.3416    2.0412   -1.1372 H   0  0  0  0  0  0
   -0.4720    2.9597    0.3596 H   0  0  0  0  0  0
    0.9797    2.0221    0.0209 H   0  0  0  0  0  0
    2.4585    0.9322   -1.6628 H   0  0  0  0  0  0
   -0.7897   -1.3021   -0.1304 H   0  0  0  0  0  0
    1.6057   -1.3521   -0.9180 H   0  0  0  0  0  0
  1  2  2  0  0  0
  1  4  1  0  0  0
  1 20  1  0  0  0
  2  3  1  0  0  0
  2 21  1  0  0  0
  3  8  1  0  0  0
  3  9  1  0  0  0
  3 19  1  0  0  0
  4  5  1  0  0  0
  4  6  1  0  0  0
  4  7  1  0  0  0
  5 10  1  0  0  0
  5 11  1  0  0  0
  5 12  1  0  0  0
  6 13  1  0  0  0
  6 14  1  0  0  0
  6 15  1  0  0  0
  7 16  1  0  0  0
  7 17  1  0  0  0
  7 18  1  0  0  0
M  END
$$$$
""",
        removeHs=False,
    )

    val_and_grad_fn = get_vacuum_val_and_grad_fn(mol, ff)

    x0 = get_romol_conf(mol)
    box0 = np.eye(3) * 100.0

    # All atoms will be free
    free_idxs = np.arange(mol.GetNumAtoms())

    u_init, g_init = val_and_grad_fn(x0)

    x_opt = minimizer.local_minimize(x0, box0, val_and_grad_fn, free_idxs, minimizer_config, restraint_k=restraint_k)

    assert np.linalg.norm(x0 - x_opt) > 0.01

    # Verify that the value and grad return the exact same result even after
    # being used for minimization
    u_init_test, g_init_test = val_and_grad_fn(x0)
    assert u_init == u_init_test
    np.testing.assert_array_equal(g_init, g_init_test)


@pytest.mark.xfail(raises=minimizer.MinimizationError, reason="Doesn't work currently, but should eventually")
def test_minimizer_failure_toy_system():
    """https://github.com/proteneer/timemachine/pull/1346 introduced a minimization scheme that minimizes
    at lambda = 0.1 and at 0.0. The need for lambda = 0.1 was due to certain ligands being very close to
    protein residues, this test replicates the failure case with a small toy system to allow investigation in the future
    """

    # Atoms 2933 and 38815 from PFKFB3 with ligands 20, 43
    nb_params = np.array(
        [
            [0.9323587, 0.13247664, 0.25629826, 0.0],
            [0.52629348, 0.12916129, 0.26202344, 0.0],
        ]
    )
    beta = 2.0
    cutoff = 1.2
    pairlist = NonbondedPairList([0, 1], [1.0, 1.0], beta, cutoff)

    # Small delta in coordinate space that results in failure
    dx = 0.04
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [dx, dx, dx],
        ]
    )
    box = np.eye(3) * 100.0

    initial_distance = distance_on_pairs(coords[None, 0], coords[None, 1], box)

    bound_impl = pairlist.bind(nb_params).to_gpu(np.float32).bound_impl

    du_dx = lambda x: bound_impl.execute(x, box, compute_u=False)[0]
    initial_force_norms = np.linalg.norm(du_dx(coords))

    minimized_coords = minimizer.fire_minimize(coords, du_dx, minimizer.FireMinimizationConfig(100))

    final_distance = distance_on_pairs(minimized_coords[None, 0], minimized_coords[None, 1], box)
    assert not np.isclose(initial_distance, final_distance, atol=2e-4)
    assert initial_force_norms > np.linalg.norm(du_dx(minimized_coords))
    minimizer.check_force_norm(minimized_coords)
