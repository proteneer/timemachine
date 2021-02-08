import jax

import numpy as np

from rdkit import Chem

from ff.handlers import openmm_deserializer
from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers

from fe import free_energy, topology
from fe import estimator

from parallel.client import CUDAPoolClient

from md import builders, minimizer

from timemachine.lib import LangevinIntegrator, custom_ops


def test_absolute_free_energy():

    suppl = Chem.SDMolSupplier('tests/data/ligands_40.sdf', removeHs=False)
    all_mols = [x for x in suppl]
    mol = all_mols[1]

    complex_system, complex_coords, _, _, complex_box, _ = builders.build_protein_system('tests/data/hif2a_nowater_min.pdb')
    complex_box += np.eye(3)*0.1 # BFGS this later

    # build the water system.
    solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(4.0)
    solvent_box += np.eye(3)*0.1 # BFGS this later

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

            dG = estimator.deltaG(model, sys_params)
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
    complex_box += np.eye(3)*0.1 # BFGS this later

    # build the water system.
    solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(4.0)
    solvent_box += np.eye(3)*0.1 # BFGS this later

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

        return estimator.deltaG(model, sys_params)

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

            dG = estimator.deltaG(model, sys_params)
            dGs.append(dG)

        return dGs[0] - dGs[1]

    # automatic chaining of vjps
    vg_fn = jax.value_and_grad(binding_model)
    dG, ff_grads = vg_fn(ff_params) # dG and ff_params_grad
    for g, h in zip(ff_grads, ff_handles):
        assert g.shape == h.params.shape
        assert np.all(np.abs(g) < 10000)

    assert np.abs(dG) < 1000.0
