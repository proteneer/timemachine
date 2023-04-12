from importlib import resources
from time import time

import numpy as np

from timemachine.fe.utils import read_sdf
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.md import builders, minimizer
from timemachine.md.minimizer import equilibrate_host_barker, make_host_du_dx_fxn


def test_minimizer():
    ff = Forcefield.load_default()

    with resources.path("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as path_to_pdb:
        complex_system, complex_coords, complex_box, _ = builders.build_protein_system(
            str(path_to_pdb), ff.protein_ff, ff.water_ff
        )

    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        all_mols = read_sdf(path_to_ligand)

    mol_a = all_mols[1]
    mol_b = all_mols[4]

    # TODO[requirements-gathering]:
    #   do we really want to minimize here ("equilibrate to temperature ~= 0"),
    #   or do we want to equilibrate ("equilibrate to temperature = 300")?
    #   and if we run MD @ temperature = 300 initialized from x_host, how long does it take to "heat back up"?
    room_temperature = 300.0
    zero_temperature = 0.0

    # equilibrate_host_barker and minimize_host_4d methods will throw if the minimization failed

    setups = {"A and B simultaneously": [mol_a, mol_b], "A alone": [mol_a], "B alone": [mol_b]}

    for key in setups:
        print(f"minimizing host given {key}...")
        mols = setups[key]
        host_du_dx_fxn = make_host_du_dx_fxn(mols, complex_system, complex_coords, ff, complex_box)

        print(f"using unadjusted Barker proposal @ temperature = {room_temperature} K...")
        t0 = time()
        x_host = equilibrate_host_barker(
            mols, complex_system, complex_coords, ff, complex_box, temperature=room_temperature
        )
        t1 = time()
        max_frc = np.linalg.norm(host_du_dx_fxn(x_host), axis=-1).max()
        print(f"\tforce norm after room-temperature equilibration: {max_frc:.3f} kJ/mol / nm")
        print(f"\tmax distance traveled = {np.linalg.norm(np.array(complex_coords) - x_host, axis=-1).max():.3f} nm")
        print(f"\tdone in {(t1 - t0):.3f} s")

        print(f"using unadjusted Barker proposal @ temperature = {zero_temperature} K...")
        t0 = time()
        x_host = equilibrate_host_barker(
            mols, complex_system, complex_coords, ff, complex_box, temperature=zero_temperature
        )
        t1 = time()

        max_frc = np.linalg.norm(host_du_dx_fxn(x_host), axis=-1).max()

        print(f"\tforce norm after low-temperature 'equilibration': {max_frc:.3f} kJ/mol / nm")
        print(f"\tmax distance traveled = {np.linalg.norm(np.array(complex_coords) - x_host, axis=-1).max():.3f} nm")
        print(f"\tdone in {(t1 - t0):.3f} s")

        t0 = time()
        print("using 4D FIRE annealing")
        x_host = minimizer.minimize_host_4d(mols, complex_system, complex_coords, ff, complex_box)
        max_frc = np.linalg.norm(host_du_dx_fxn(x_host), axis=-1).max()
        print(f"\tmax force norm after 4D FIRE annealing: {max_frc:.3f} kJ/mol / nm")
        print(f"\tmax distance traveled = {np.linalg.norm(np.array(complex_coords) - x_host, axis=-1).max():.3f} nm")
        t1 = time()
        print(f"\tdone in {(t1 - t0):.3f} s")


def test_equilibrate_host():
    ff = Forcefield.load_default()
    host_system, host_coords, host_box, _ = builders.build_water_system(4.0, ff.water_ff)

    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        mols = read_sdf(path_to_ligand)

    mol = mols[0]

    coords, box = minimizer.equilibrate_host(mol, host_system, host_coords, 300, 1.0, ff, host_box, 25, seed=2022)
    assert coords.shape[0] == host_coords.shape[0] + mol.GetNumAtoms()
    assert coords.shape[1] == host_coords.shape[1]
    assert box.shape == host_box.shape


def test_local_minimize_water_box():
    """
    Test that we can locally relax a box of water by selecting some random indices.
    """
    ff = Forcefield.load_default()

    system, x0, box0, _ = builders.build_water_system(4.0, ff.water_ff)
    x0 = np.array(builders.strip_units(x0))
    host_fns, _ = openmm_deserializer.deserialize_system(system, cutoff=1.2)
    box0 += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes at the boundary

    bound_impls = [p.to_gpu(np.float32).bound_impl for p in host_fns]
    val_and_grad_fn = minimizer.get_val_and_grad_fn(bound_impls, box0)

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
