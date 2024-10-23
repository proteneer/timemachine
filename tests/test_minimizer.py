from importlib import resources
from time import time

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from timemachine.fe.free_energy import HostConfig
from timemachine.fe.model_utils import get_vacuum_val_and_grad_fn
from timemachine.fe.utils import get_romol_conf, read_sdf, read_sdf_mols_by_name
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.md import builders, minimizer
from timemachine.md.barostat.utils import compute_box_volume
from timemachine.md.minimizer import equilibrate_host_barker, make_host_du_dx_fxn
from timemachine.potentials import NonbondedPairList
from timemachine.potentials.jax_utils import distance_on_pairs


@pytest.mark.parametrize(
    "pdb_path, sdf_path, mol_a_name, mol_b_name",
    [
        (
            resources.path("timemachine.testsystems.data", "hif2a_nowater_min.pdb"),
            resources.path("timemachine.testsystems.data", "ligands_40.sdf"),
            "43",
            "234",
        ),
        (
            resources.path("timemachine.datasets.fep_benchmark.pfkfb3", "6hvi_prepared.pdb"),
            resources.path("timemachine.datasets.fep_benchmark.pfkfb3", "ligands.sdf"),
            "20",
            "43",
        ),
        (
            resources.path("timemachine.datasets.fep_benchmark.pfkfb3", "6hvi_prepared.pdb"),
            resources.path("timemachine.datasets.fep_benchmark.pfkfb3", "ligands.sdf"),
            "41",
            "43",
        ),
        (
            resources.path("timemachine.datasets.fep_benchmark.pfkfb3", "6hvi_prepared.pdb"),
            resources.path("timemachine.datasets.fep_benchmark.pfkfb3", "ligands.sdf"),
            "34",
            "37",
        ),
        (
            resources.path("timemachine.datasets.fep_benchmark.pfkfb3", "6hvi_prepared.pdb"),
            resources.path("timemachine.datasets.fep_benchmark.pfkfb3", "ligands.sdf"),
            "26",
            "37",
        ),
    ],
)
def test_fire_minimize_host_protein(pdb_path, sdf_path, mol_a_name, mol_b_name):
    ff = Forcefield.load_default()
    mols_by_name = read_sdf_mols_by_name(sdf_path)
    mol_a = mols_by_name[mol_a_name]
    mol_b = mols_by_name[mol_b_name]

    for mols in [[mol_a], [mol_b], [mol_a, mol_b]]:
        complex_system, complex_coords, complex_box, complex_top, num_water_atoms = builders.build_protein_system(
            str(pdb_path), ff.protein_ff, ff.water_ff, mols=mols
        )
        host_config = HostConfig(complex_system, complex_coords, complex_box, num_water_atoms, complex_top)
        x_host = minimizer.fire_minimize_host(mols, host_config, ff)
        assert x_host.shape == complex_coords.shape


def test_fire_minimize_host_solvent():
    ff = Forcefield.load_default()
    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        all_mols = read_sdf(path_to_ligand)
    mol_a = all_mols[1]
    mol_b = all_mols[4]

    for mols in [[mol_a], [mol_b], [mol_a, mol_b]]:
        solvent_system, solvent_coords, solvent_box, solvent_top = builders.build_water_system(
            4.0, ff.water_ff, mols=mols
        )
        host_config = HostConfig(solvent_system, solvent_coords, solvent_box, len(solvent_coords), solvent_top)
        x_host = minimizer.fire_minimize_host(mols, host_config, ff)
        assert x_host.shape == solvent_coords.shape


@pytest.mark.parametrize("host_name", ["solvent", "complex"])
@pytest.mark.parametrize("mol_pair", [("20", "43")])
def test_pre_equilibrate_host_pfkfb3(host_name, mol_pair):
    ff = Forcefield.load_default()
    mol_a_name, mol_b_name = mol_pair
    with resources.path("timemachine.datasets.fep_benchmark.pfkfb3", "ligands.sdf") as path_to_ligand:
        mols_by_name = read_sdf_mols_by_name(path_to_ligand)
    mol_a = mols_by_name[mol_a_name]
    mol_b = mols_by_name[mol_b_name]
    mols = [mol_a, mol_b]
    if host_name == "solvent":
        solvent_system, solvent_coords, solvent_box, solvent_top = builders.build_water_system(
            4.0, ff.water_ff, mols=mols
        )
        host_config = HostConfig(solvent_system, solvent_coords, solvent_box, len(solvent_coords), solvent_top)
    else:
        with resources.path("timemachine.datasets.fep_benchmark.pfkfb3", "6hvi_prepared.pdb") as pdb_path:
            complex_system, complex_coords, complex_box, complex_top, num_water_atoms = builders.build_protein_system(
                str(pdb_path), ff.protein_ff, ff.water_ff, mols=mols
            )
        host_config = HostConfig(complex_system, complex_coords, complex_box, num_water_atoms, complex_top)
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
    solvent_system, solvent_coords, solvent_box, solvent_top = builders.build_water_system(4.0, ff.water_ff, mols=[mol])
    host_config = HostConfig(solvent_system, solvent_coords, solvent_box, len(solvent_coords), solvent_top)
    x_host = minimizer.fire_minimize_host([mol], host_config, ff)
    assert x_host.shape == solvent_coords.shape


@pytest.mark.nightly(reason="Currently not used in practice")
def test_equilibrate_host_barker():
    ff = Forcefield.load_default()
    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        all_mols = read_sdf(path_to_ligand)
    mol_a = all_mols[1]
    mol_b = all_mols[4]

    with resources.path("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as path_to_pdb:
        complex_system, complex_coords, complex_box, complex_top, num_water_atoms = builders.build_protein_system(
            str(path_to_pdb), ff.protein_ff, ff.water_ff, mols=[mol_a, mol_b]
        )
        host_config = HostConfig(complex_system, complex_coords, complex_box, num_water_atoms, complex_top)

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
        assert x_host.shape == complex_coords.shape
        t1 = time()
        max_frc = np.linalg.norm(host_du_dx_fxn(x_host), axis=-1).max()
        print(f"\tforce norm after room-temperature equilibration: {max_frc:.3f} kJ/mol / nm")
        print(f"\tmax distance traveled = {np.linalg.norm(np.array(complex_coords) - x_host, axis=-1).max():.3f} nm")
        print(f"\tdone in {(t1 - t0):.3f} s")

        print(f"using unadjusted Barker proposal @ temperature = {zero_temperature} K...")
        t0 = time()
        x_host = equilibrate_host_barker(mols, host_config, ff, temperature=zero_temperature)
        assert x_host.shape == complex_coords.shape
        t1 = time()

        max_frc = np.linalg.norm(host_du_dx_fxn(x_host), axis=-1).max()

        print(f"\tforce norm after low-temperature 'equilibration': {max_frc:.3f} kJ/mol / nm")
        print(f"\tmax distance traveled = {np.linalg.norm(np.array(complex_coords) - x_host, axis=-1).max():.3f} nm")
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

    system, x0, box0, top = builders.build_water_system(4.0, ff.water_ff)
    host_fns, _ = openmm_deserializer.deserialize_system(system, cutoff=1.2)
    box0 += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes at the boundary

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


def test_local_minimize_water_box_with_bounds():
    """
    Test that we can locally relax a box of water using L-BFGS-B with bounds
    """
    ff = Forcefield.load_default()

    system, x0, box0, top = builders.build_water_system(4.0, ff.water_ff)
    host_fns, _ = openmm_deserializer.deserialize_system(system, cutoff=1.2)
    box0 += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes at the boundary

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
@pytest.mark.parametrize("restraint_k", [None, 3_000.0])
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
