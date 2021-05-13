import jax
from jax import grad, value_and_grad, config, jacfwd, jacrev
config.update("jax_enable_x64", True)

import numpy as np

from rdkit import Chem
from scipy.optimize import minimize

from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers

from fe import free_energy, topology
from fe import estimator

from parallel.client import CUDAPoolClient

from md import builders, minimizer

from timemachine.lib import LangevinIntegrator

from fe.functional import construct_differentiable_interface
from testsystems.relative import hif2a_ligand_pair


def test_absolute_free_energy():

    suppl = Chem.SDMolSupplier('tests/data/ligands_40.sdf', removeHs=False)
    all_mols = [x for x in suppl]
    mol = all_mols[1]

    complex_system, complex_coords, _, _, complex_box, _ = builders.build_protein_system('tests/data/hif2a_nowater_min.pdb')

    # build the water system.
    solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(4.0)

    ff = Forcefield(deserialize_handlers(open('ff/params/smirnoff_1_1_0_ccc.py').read()))

    ff_params = ff.get_ordered_params()
    ff_handles = ff.get_ordered_handles()

    seed = 2021

    lambda_schedule = np.linspace(0, 1.0, 4)
    equil_steps = 1000
    prod_steps = 1000

    afe = free_energy.AbsoluteFreeEnergy(mol, ff)

    def absolute_model(ff_params):

        dGs = []

        for host_system, host_coords, host_box in [
            (complex_system, complex_coords, complex_box),
            (solvent_system, solvent_coords, solvent_box)]:

            # minimize the host to avoid clashes
            host_coords = minimizer.minimize_host_4d([mol], host_system, host_coords, ff, host_box)

            unbound_potentials, sys_params, masses, coords = afe.prepare_host_edge(ff_params, host_system, host_coords)

            x0 = coords
            v0 = np.zeros_like(coords)
            client = CUDAPoolClient(1)

            integrator = LangevinIntegrator(
                300.0,
                1.5e-3,
                1.0,
                masses,
                seed
            )

            model = estimator.FreeEnergyModel(
                unbound_potentials,
                client,
                host_box,
                x0,
                v0,
                integrator,
                lambda_schedule,
                equil_steps,
                prod_steps
            )

            dG, _ = estimator.deltaG(model, sys_params)
            dGs.append(dG)


        return dGs[0] - dGs[1]

    # automatic chaining of vjps
    vg_fn = jax.value_and_grad(absolute_model)
    dG, ff_grads = vg_fn(ff_params) # dG and ff_params_grad
    for g, h in zip(ff_grads, ff_handles):
        assert g.shape == h.params.shape
        assert np.all(np.abs(g) < 10000)

    assert np.abs(dG) < 1000.0

def test_relative_free_energy():
    # test that we can properly build a single topology host guest system and
    # that we can run a few steps in a stable way. This tests runs both the complex
    # and the solvent stages.

    suppl = Chem.SDMolSupplier('tests/data/ligands_40.sdf', removeHs=False)
    all_mols = [x for x in suppl]
    mol_a = all_mols[1]
    mol_b = all_mols[4]

    core = np.array([
        [ 0,  0],
        [ 2,  2],
        [ 1,  1],
        [ 6,  6],
        [ 5,  5],
        [ 4,  4],
        [ 3,  3],
        [15, 16],
        [16, 17],
        [17, 18],
        [18, 19],
        [19, 20],
        [20, 21],
        [32, 30],
        [26, 25],
        [27, 26],
        [ 7,  7],
        [ 8,  8],
        [ 9,  9],
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
        [21, 22]
    ])

    complex_system, complex_coords, _, _, complex_box, _ = builders.build_protein_system('tests/data/hif2a_nowater_min.pdb')

    # build the water system.
    solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(4.0)

    ff = Forcefield(deserialize_handlers(open('ff/params/smirnoff_1_1_0_ccc.py').read()))

    ff_params = ff.get_ordered_params()
    ff_handles = ff.get_ordered_handles()

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
        box = np.eye(3, dtype=np.float64)*100

        integrator = LangevinIntegrator(
            300.0,
            1.5e-3,
            1.0,
            masses,
            seed
        )

        model = estimator.FreeEnergyModel(
            unbound_potentials,
            client,
            box,
            x0,
            v0,
            integrator,
            lambda_schedule,
            equil_steps,
            prod_steps
        )

        return estimator.deltaG(model, sys_params)[0]

    vg_fn = jax.value_and_grad(vacuum_model)
    dG, ff_grads = vg_fn(ff_params) # dG and ff_params_grad
    for g, h in zip(ff_grads, ff_handles):
        assert g.shape == h.params.shape
        assert np.all(np.abs(g) < 10000)
    assert np.abs(dG) < 1000.0


    def binding_model(ff_params):

        dGs = []

        for host_system, host_coords, host_box in [
            (complex_system, complex_coords, complex_box),
            (solvent_system, solvent_coords, solvent_box)]:

            # minimize the host to avoid clashes
            host_coords = minimizer.minimize_host_4d([mol_a], host_system, host_coords, ff, host_box)

            unbound_potentials, sys_params, masses, coords = rfe.prepare_host_edge(ff_params, host_system, host_coords)

            x0 = coords
            v0 = np.zeros_like(coords)
            client = CUDAPoolClient(1)

            integrator = LangevinIntegrator(
                300.0,
                1.5e-3,
                1.0,
                masses,
                seed
            )

            model = estimator.FreeEnergyModel(
                unbound_potentials,
                client,
                host_box,
                x0,
                v0,
                integrator,
                lambda_schedule,
                equil_steps,
                prod_steps
            )

            dG, _ = estimator.deltaG(model, sys_params)
            dGs.append(dG)

        return dGs[0] - dGs[1]

    # automatic chaining of vjps
    vg_fn = jax.value_and_grad(binding_model)
    dG, ff_grads = vg_fn(ff_params) # dG and ff_params_grad
    for g, h in zip(ff_grads, ff_handles):
        assert g.shape == h.params.shape
        assert np.all(np.abs(g) < 10000)

    assert np.abs(dG) < 1000.0


def test_functional():
    """Assert that derivatives of U w.r.t. x, params, and lam accessible by grad(U) are of the correct shape.
    Also assert that a differentiable loss function in terms of U can be minimized, and that
    forward-mode and reverse-mode differentiation of loss agree."""

    ff_params = hif2a_ligand_pair.ff.get_ordered_params()
    unbound_potentials, sys_params, _, coords = hif2a_ligand_pair.prepare_vacuum_edge(ff_params)
    box = np.eye(3) * 100
    lam = 0.5

    for precision in [np.float32, np.float64]:
        U = construct_differentiable_interface(unbound_potentials, precision)

        # can call U and get right shape
        energy = U(coords, sys_params, box, lam)
        assert energy.shape == ()

        # can call grad(U) and get right shape
        du_dx, du_dp, du_dl = grad(U, argnums=(0, 1, 3))(coords, sys_params, box, lam)
        assert du_dx.shape == coords.shape
        for (p, p_prime) in zip(sys_params, du_dp):
            assert p.shape == p_prime.shape
        assert du_dl.shape == ()

        # can scipy.optimize a differentiable Jax function that calls U
        nb_params = sys_params[-1]
        nb_params_shape = nb_params.shape

        def loss(nb_params):
            concat_params = sys_params[:-1] + [nb_params]
            return (U(coords, concat_params, box, lam) - 666) ** 2

        x0 = nb_params.flatten()

        def flat_loss(flat_nb_params):
            return loss(flat_nb_params.reshape(nb_params_shape))

        def fun(flat_nb_params):
            v, g = value_and_grad(flat_loss)(flat_nb_params)
            return float(v), np.array(g)

        # forward-mode agrees with reverse-mode
        fwd_result = jacfwd(flat_loss)(x0)
        rev_result = jacrev(flat_loss)(x0)

        np.testing.assert_array_almost_equal(fwd_result, rev_result)

        # minimization successful
        result = minimize(fun, x0, jac=True, tol=0)
        assert flat_loss(result.x) < 1e-10
