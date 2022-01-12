from jax import grad, value_and_grad, config, jacfwd, jacrev

config.update("jax_enable_x64", True)

import numpy as np

from rdkit import Chem
from scipy.optimize import minimize, check_grad

from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers

from fe import free_energy, topology
from fe import estimator

from parallel.client import CUDAPoolClient

from md import builders, minimizer

from timemachine.lib import LangevinIntegrator, MonteCarloBarostat

from fe.functional import construct_differentiable_interface, construct_differentiable_interface_fast
from md.barostat.utils import get_bond_list, get_group_indices
from testsystems.relative import hif2a_ligand_pair


def test_absolute_free_energy():

    suppl = Chem.SDMolSupplier("tests/data/ligands_40.sdf", removeHs=False)
    all_mols = [x for x in suppl]
    mol = all_mols[1]

    complex_system, complex_coords, _, _, complex_box, _ = builders.build_protein_system(
        "tests/data/hif2a_nowater_min.pdb"
    )

    # build the water system.
    solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(4.0)

    ff = Forcefield(deserialize_handlers(open("ff/params/smirnoff_1_1_0_ccc.py").read()))

    ff_params = ff.get_ordered_params()

    seed = 2021

    lambda_schedule = np.linspace(0, 1.0, 4)
    equil_steps = 1000
    prod_steps = 1000

    afe = free_energy.AbsoluteFreeEnergy(mol, ff)

    def absolute_model(ff_params):

        dGs = []

        for host_system, host_coords, host_box in [
            (complex_system, complex_coords, complex_box),
            (solvent_system, solvent_coords, solvent_box),
        ]:

            # minimize the host to avoid clashes
            host_coords = minimizer.minimize_host_4d([mol], host_system, host_coords, ff, host_box)

            unbound_potentials, sys_params, masses, coords = afe.prepare_host_edge(ff_params, host_system, host_coords)

            harmonic_bond_potential = unbound_potentials[0]
            group_idxs = get_group_indices(get_bond_list(harmonic_bond_potential))

            x0 = coords
            v0 = np.zeros_like(coords)
            client = CUDAPoolClient(1)
            temperature = 300.0
            pressure = 1.0

            integrator = LangevinIntegrator(temperature, 1.5e-3, 1.0, masses, seed)

            barostat = MonteCarloBarostat(x0.shape[0], pressure, temperature, group_idxs, 25, seed)

            model = estimator.FreeEnergyModel(
                unbound_potentials,
                client,
                host_box,
                x0,
                v0,
                integrator,
                lambda_schedule,
                equil_steps,
                prod_steps,
                barostat,
            )

            dG, _ = estimator.deltaG(model, sys_params)
            dGs.append(dG)

        return dGs[0] - dGs[1]

    dG = absolute_model(ff_params)
    assert np.abs(dG) < 1000.0


def test_relative_free_energy():
    # test that we can properly build a single topology host guest system and
    # that we can run a few steps in a stable way. This tests runs both the complex
    # and the solvent stages.

    suppl = Chem.SDMolSupplier("tests/data/ligands_40.sdf", removeHs=False)
    all_mols = [x for x in suppl]
    mol_a = all_mols[1]
    mol_b = all_mols[4]

    core = np.array(
        [
            [0, 0],
            [2, 2],
            [1, 1],
            [6, 6],
            [5, 5],
            [4, 4],
            [3, 3],
            [15, 16],
            [16, 17],
            [17, 18],
            [18, 19],
            [19, 20],
            [20, 21],
            [32, 30],
            [26, 25],
            [27, 26],
            [7, 7],
            [8, 8],
            [9, 9],
            [10, 10],
            [29, 11],
            [11, 12],
            [12, 13],
            [14, 15],
            [31, 29],
            [13, 14],
            [23, 24],
            [30, 28],
            [28, 27],
            [21, 22],
        ]
    )

    complex_system, complex_coords, _, _, complex_box, _ = builders.build_protein_system(
        "tests/data/hif2a_nowater_min.pdb"
    )

    # build the water system.
    solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(4.0)

    ff = Forcefield(deserialize_handlers(open("ff/params/smirnoff_1_1_0_ccc.py").read()))

    ff_params = ff.get_ordered_params()

    seed = 2021

    lambda_schedule = np.linspace(0, 1.0, 4)
    equil_steps = 1000
    prod_steps = 1000

    single_topology = topology.SingleTopology(mol_a, mol_b, core, ff)
    rfe = free_energy.RelativeFreeEnergy(single_topology)

    def vacuum_model(ff_params):

        unbound_potentials, sys_params, masses, coords = rfe.prepare_vacuum_edge(ff_params)

        x0 = coords
        v0 = np.zeros_like(coords)
        client = CUDAPoolClient(1)
        box = np.eye(3, dtype=np.float64) * 100

        harmonic_bond_potential = unbound_potentials[0]
        group_idxs = get_group_indices(get_bond_list(harmonic_bond_potential))

        x0 = coords
        v0 = np.zeros_like(coords)
        client = CUDAPoolClient(1)
        temperature = 300.0
        pressure = 1.0

        integrator = LangevinIntegrator(temperature, 1.5e-3, 1.0, masses, seed)

        barostat = MonteCarloBarostat(x0.shape[0], pressure, temperature, group_idxs, 25, seed)
        model = estimator.FreeEnergyModel(
            unbound_potentials, client, box, x0, v0, integrator, lambda_schedule, equil_steps, prod_steps, barostat
        )

        return estimator.deltaG(model, sys_params)[0]

    dG = vacuum_model(ff_params)
    assert np.abs(dG) < 1000.0

    def binding_model(ff_params):

        dGs = []

        for host_system, host_coords, host_box in [
            (complex_system, complex_coords, complex_box),
            (solvent_system, solvent_coords, solvent_box),
        ]:

            # minimize the host to avoid clashes
            host_coords = minimizer.minimize_host_4d([mol_a], host_system, host_coords, ff, host_box)

            unbound_potentials, sys_params, masses, coords = rfe.prepare_host_edge(ff_params, host_system, host_coords)

            x0 = coords
            v0 = np.zeros_like(coords)
            client = CUDAPoolClient(1)

            harmonic_bond_potential = unbound_potentials[0]
            group_idxs = get_group_indices(get_bond_list(harmonic_bond_potential))

            temperature = 300.0
            pressure = 1.0

            integrator = LangevinIntegrator(temperature, 1.5e-3, 1.0, masses, seed)

            barostat = MonteCarloBarostat(x0.shape[0], pressure, temperature, group_idxs, 25, seed)

            model = estimator.FreeEnergyModel(
                unbound_potentials,
                client,
                host_box,
                x0,
                v0,
                integrator,
                lambda_schedule,
                equil_steps,
                prod_steps,
                barostat,
            )

            dG, _ = estimator.deltaG(model, sys_params)
            dGs.append(dG)

        return dGs[0] - dGs[1]

    dG = binding_model(ff_params)
    assert np.abs(dG) < 1000.0


def assert_shapes_consistent(U, coords, sys_params, box, lam):
    """assert U, grad(U) have the right shapes"""
    # can call U and get right shape
    energy = U(coords, sys_params, box, lam)
    assert energy.shape == ()

    # can call grad(U) and get right shape
    du_dx, du_dp, du_dl = grad(U, argnums=(0, 1, 3))(coords, sys_params, box, lam)
    assert du_dx.shape == coords.shape
    for (p, p_prime) in zip(sys_params, du_dp):
        assert p.shape == p_prime.shape
    assert du_dl.shape == ()


def assert_fwd_rev_consistent(f, x):
    """assert jacfwd(f)(x) == jacrev(f)(x)"""

    # forward-mode agrees with reverse-mode
    fwd_result = jacfwd(f)(x)
    rev_result = jacrev(f)(x)

    np.testing.assert_array_almost_equal(fwd_result, rev_result)


def assert_no_second_derivative(f, x):
    """assert an exception is raised if we try to access second derivatives of f"""

    # shouldn't be able to take second derivatives
    def div_f(x):
        return np.sum(grad(f)(x))

    problem = None
    try:
        # another grad should be a no-no
        grad(div_f)(x)
    except Exception as e:
        problem = e
    assert type(problem) == TypeError


def assert_ff_optimizable(U, coords, sys_params, box, lam):
    """define a differentiable loss function in terms of U, assert it can be minimized,
    and return initial params, optimized params, and the loss function"""

    nb_params = sys_params[-1]
    nb_params_shape = nb_params.shape

    def loss(nb_params):
        concat_params = sys_params[:-1] + [nb_params]
        return (U(coords, concat_params, box, lam) - 666) ** 2

    x_0 = nb_params.flatten()

    def flat_loss(flat_nb_params):
        return loss(flat_nb_params.reshape(nb_params_shape))

    def fun(flat_nb_params):
        v, g = value_and_grad(flat_loss)(flat_nb_params)
        return float(v), np.array(g)

    # minimization successful
    result = minimize(fun, x_0, jac=True, tol=0)
    x_opt = result.x
    assert flat_loss(x_opt) < 1e-10

    return x_0, x_opt, flat_loss


def test_functional():
    """Assert that
    * derivatives of U w.r.t. x, params, and lam accessible by grad(U) are of the correct shape,
    * a differentiable loss function in terms of U can be minimized,
    * forward-mode and reverse-mode differentiation of loss agree,
    * an exception is raised if we try to do something that requires second derivatives,
    * grad(nonlinear_function_in_terms_of_U) agrees with finite-difference
    """

    ff_params = hif2a_ligand_pair.ff.get_ordered_params()
    unbound_potentials, sys_params, _, coords = hif2a_ligand_pair.prepare_vacuum_edge(ff_params)
    box = np.eye(3) * 100
    lam = 0.5

    for precision in [np.float32, np.float64]:
        U = construct_differentiable_interface(unbound_potentials, precision)

        # U, grad(U) have the right shapes
        assert_shapes_consistent(U, coords, sys_params, box, lam)

        # can scipy.optimize a differentiable Jax function that calls U
        x_0, x_opt, flat_loss = assert_ff_optimizable(U, coords, sys_params, box, lam)

        # jacfwd agrees with jacrev
        assert_fwd_rev_consistent(flat_loss, x_0)

        # calling grad twice shouldn't be allowed
        assert_no_second_derivative(flat_loss, x_0)

        # check grad by comparison to forward finite-difference
        if precision == np.float64:

            def low_dim_f(perturb: float) -> float:
                """low-dimensional input so that finite-difference isn't too expensive"""

                # scaling perturbation down by 1e-4 so that f(1.0) isn't 10^30ish...
                return flat_loss(x_opt + 1e-4 * perturb) ** 2

            perturbations = np.linspace(-1, 1, 10)

            for perturb in perturbations:
                abs_err = check_grad(low_dim_f, grad(low_dim_f), perturb, epsilon=1e-4)
                assert abs_err < 1e-3


def test_construct_differentiable_interface_fast():
    """Assert that the computation of U and its derivatives using the
    C++ code path produces equivalent results to doing the
    summation in Python"""

    ff_params = hif2a_ligand_pair.ff.get_ordered_params()
    unbound_potentials, sys_params, _, coords = hif2a_ligand_pair.prepare_vacuum_edge(ff_params)
    box = np.eye(3) * 100
    lam = 0.5

    for precision in [np.float32, np.float64]:
        U_ref = construct_differentiable_interface(unbound_potentials, precision)
        U = construct_differentiable_interface_fast(unbound_potentials, sys_params, precision)
        args = (coords, sys_params, box, lam)
        np.testing.assert_array_equal(U(*args), U_ref(*args))

        argnums = (0, 1, 3)
        grad_U_ref = grad(U_ref, argnums=argnums)(*args)
        grad_U = grad(U, argnums=argnums)(*args)

        np.testing.assert_array_equal(grad_U[0], grad_U_ref[0])

        assert len(grad_U[1]) == len(grad_U_ref[1])
        for dU_dp, dU_dp_ref in zip(grad_U[1], grad_U_ref[1]):
            np.testing.assert_array_equal(dU_dp, dU_dp_ref)

        np.testing.assert_array_equal(grad_U[2], grad_U_ref[2])
