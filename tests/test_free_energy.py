from importlib import resources

import numpy as np
import pytest
from jax import grad, jacfwd, jacrev, value_and_grad
from rdkit import Chem
from scipy.optimize import check_grad, minimize

from timemachine.constants import DEFAULT_FF
from timemachine.fe import free_energy, topology, utils
from timemachine.fe.functional import construct_differentiable_interface, construct_differentiable_interface_fast
from timemachine.fe.single_topology import SingleTopology
from timemachine.ff import Forcefield
from timemachine.md import builders
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology


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


def assert_ff_optimizable(U, coords, sys_params, box, lam, tol=1e-10):
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
    assert flat_loss(x_opt) < tol

    return x_0, x_opt, flat_loss


def test_functional():
    """Assert that
    * derivatives of U w.r.t. x, params, and lam accessible by grad(U) are of the correct shape,
    * a differentiable loss function in terms of U can be minimized,
    * forward-mode and reverse-mode differentiation of loss agree,
    * an exception is raised if we try to do something that requires second derivatives,
    * grad(nonlinear_function_in_terms_of_U) agrees with finite-difference,
    * requesting derivative w.r.t. box causes a runtime error
    """

    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_from_file(DEFAULT_FF)
    st = SingleTopology(mol_a, mol_b, core, forcefield)

    vac_sys = st.setup_intermediate_state(0.5)
    x_a = utils.get_romol_conf(st.mol_a)
    x_b = utils.get_romol_conf(st.mol_b)
    coords = st.combine_confs(x_a, x_b)
    box = np.eye(3) * 100
    lam = 0.5

    potentials = vac_sys.get_U_fns()
    sys_params = [np.array(bp.params) for bp in potentials]

    tol_at_precision = {np.float32: 2.5e-10, np.float64: 1e-10}
    for precision, tol in tol_at_precision.items():
        U = construct_differentiable_interface(potentials, precision)

        # U, grad(U) have the right shapes
        assert_shapes_consistent(U, coords, sys_params, box, lam)

        # can scipy.optimize a differentiable Jax function that calls U
        x_0, x_opt, flat_loss = assert_ff_optimizable(U, coords, sys_params, box, lam, tol)

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

        # grad w.r.t. box shouldn't be allowed
        with pytest.raises(RuntimeError) as e:
            _ = grad(U, argnums=2)(coords, sys_params, box, lam)
        assert "box" in str(e).lower()


def test_construct_differentiable_interface_fast():
    """Assert that the computation of U and its derivatives using the
    C++ code path produces equivalent results to doing the
    summation in Python"""
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_from_file(DEFAULT_FF)
    st = SingleTopology(mol_a, mol_b, core, forcefield)
    vac_sys = st.setup_intermediate_state(0.5)
    x_a = utils.get_romol_conf(st.mol_a)
    x_b = utils.get_romol_conf(st.mol_b)
    coords = st.combine_confs(x_a, x_b)
    box = np.eye(3) * 100
    lam = 0.5

    potentials = vac_sys.get_U_fns()
    sys_params = [np.array(bp.params) for bp in potentials]

    for precision in [np.float32, np.float64]:
        U_ref = construct_differentiable_interface(potentials, precision)
        U = construct_differentiable_interface_fast(potentials, sys_params, precision)
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


def test_absolute_vacuum():
    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        mol = next(Chem.SDMolSupplier(str(path_to_ligand), removeHs=False))

    ff = Forcefield.load_from_file("smirnoff_1_1_0_ccc.py")
    ff_params = ff.get_params()

    bt = topology.BaseTopology(mol, ff)
    afe = free_energy.AbsoluteFreeEnergy(mol, bt)

    unbound_potentials, sys_params, masses = afe.prepare_vacuum_edge(ff_params)
    assert np.all(masses == utils.get_mol_masses(mol))
    np.testing.assert_array_almost_equal(afe.prepare_combined_coords(), utils.get_romol_conf(mol))


def test_vacuum_and_solvent_edge_types():
    """Ensure that the values returned by the vacuum and solvent edges are all of the same type."""
    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        mol = next(Chem.SDMolSupplier(str(path_to_ligand), removeHs=False))

    ff = Forcefield.load_from_file("smirnoff_1_1_0_ccc.py")
    solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(3.0, ff.water_ff)

    ff_params = ff.get_params()

    bt = topology.BaseTopology(mol, ff)
    afe = free_energy.AbsoluteFreeEnergy(mol, bt)

    vacuum_unbound_potentials, vacuum_sys_params, vacuum_masses = afe.prepare_vacuum_edge(ff_params)

    solvent_unbound_potentials, solvent_sys_params, solvent_masses = afe.prepare_host_edge(ff_params, solvent_system)

    assert type(vacuum_unbound_potentials) == type(solvent_unbound_potentials)
    assert type(vacuum_sys_params) == type(solvent_sys_params)
    assert type(vacuum_masses) == type(solvent_masses)
