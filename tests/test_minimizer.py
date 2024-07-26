from importlib import resources
from time import time

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from timemachine.fe.free_energy import HostConfig
from timemachine.fe.utils import get_mol_name, read_sdf
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
    all_mols = read_sdf(sdf_path)
    mol_a = next(m for m in all_mols if get_mol_name(m) == mol_a_name)
    mol_b = next(m for m in all_mols if get_mol_name(m) == mol_b_name)

    for mols in [[mol_a], [mol_b], [mol_a, mol_b]]:
        complex_system, complex_coords, complex_box, _, num_water_atoms = builders.build_protein_system(
            str(pdb_path), ff.protein_ff, ff.water_ff, mols=mols
        )
        host_config = HostConfig(complex_system, complex_coords, complex_box, num_water_atoms)
        x_host = minimizer.fire_minimize_host(mols, host_config, ff)
        assert x_host.shape == complex_coords.shape


def test_fire_minimize_host_solvent():
    ff = Forcefield.load_default()
    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        all_mols = read_sdf(path_to_ligand)
    mol_a = all_mols[1]
    mol_b = all_mols[4]

    for mols in [[mol_a], [mol_b], [mol_a, mol_b]]:
        solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(4.0, ff.water_ff, mols=mols)
        host_config = HostConfig(solvent_system, solvent_coords, solvent_box, len(solvent_coords))
        x_host = minimizer.fire_minimize_host(mols, host_config, ff)
        assert x_host.shape == solvent_coords.shape


@pytest.mark.parametrize("host_name", ["solvent", "complex"])
@pytest.mark.parametrize("mol_pair", [("20", "43")])
def test_pre_equilibrate_host_pfkfb3(host_name, mol_pair):
    ff = Forcefield.load_default()
    mol_a_name, mol_b_name = mol_pair
    with resources.path("timemachine.datasets.fep_benchmark.pfkfb3", "ligands.sdf") as path_to_ligand:
        all_mols = read_sdf(path_to_ligand)
    mol_a = next(m for m in all_mols if get_mol_name(m) == mol_a_name)
    mol_b = next(m for m in all_mols if get_mol_name(m) == mol_b_name)
    mols = [mol_a, mol_b]
    if host_name == "solvent":
        solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(4.0, ff.water_ff, mols=mols)
        host_config = HostConfig(solvent_system, solvent_coords, solvent_box, len(solvent_coords))
    else:
        with resources.path("timemachine.datasets.fep_benchmark.pfkfb3", "6hvi_prepared.pdb") as pdb_path:
            complex_system, complex_coords, complex_box, _, num_water_atoms = builders.build_protein_system(
                str(pdb_path), ff.protein_ff, ff.water_ff, mols=mols
            )
        host_config = HostConfig(complex_system, complex_coords, complex_box, num_water_atoms)
    x_host, x_box = minimizer.pre_equilibrate_host(mols, host_config, ff)
    assert x_host.shape == host_config.conf.shape
    assert compute_box_volume(x_box) < compute_box_volume(host_config.box)


def test_fire_minimize_host_adamantane():
    """With cagey molecules, can trap water molecules inside of them. Verify that molecule can be minimized
    in water without issue"""
    ff = Forcefield.load_default()
    mol = Chem.AddHs(Chem.MolFromSmiles("C1C3CC2CC(CC1C2)C3"))
    AllChem.EmbedMolecule(mol, randomSeed=2024)
    # If don't delete the relevant water this minimization fails
    solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(4.0, ff.water_ff, mols=[mol])
    host_config = HostConfig(solvent_system, solvent_coords, solvent_box, len(solvent_coords))
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
        complex_system, complex_coords, complex_box, _, num_water_atoms = builders.build_protein_system(
            str(path_to_pdb), ff.protein_ff, ff.water_ff, mols=[mol_a, mol_b]
        )
        host_config = HostConfig(complex_system, complex_coords, complex_box, num_water_atoms)

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


def test_local_minimize_water_box():
    """
    Test that we can locally relax a box of water by selecting some random indices.
    """
    ff = Forcefield.load_default()

    system, x0, box0, _ = builders.build_water_system(4.0, ff.water_ff)
    host_fns, _ = openmm_deserializer.deserialize_system(system, cutoff=1.2)
    box0 += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes at the boundary

    val_and_grad_fn = minimizer.get_val_and_grad_fn(host_fns, box0)

    free_idxs = [0, 2, 3, 6, 7, 9, 15, 16]
    frozen_idxs = set(range(len(x0))).difference(set(free_idxs))
    frozen_idxs = list(frozen_idxs)

    u_init, g_init = val_and_grad_fn(x0)

    x_opt = minimizer.local_minimize(x0, val_and_grad_fn, free_idxs)

    np.testing.assert_array_equal(x0[frozen_idxs], x_opt[frozen_idxs])
    assert np.linalg.norm(x0[free_idxs] - x_opt[free_idxs]) > 0.01

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

    minimized_coords = minimizer.fire_minimize(coords, du_dx, 100)

    final_distance = distance_on_pairs(minimized_coords[None, 0], minimized_coords[None, 1], box)
    assert not np.isclose(initial_distance, final_distance, atol=2e-4)
    assert initial_force_norms > np.linalg.norm(du_dx(minimized_coords))
    minimizer.check_force_norm(minimized_coords)
