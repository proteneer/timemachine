import unittest
import jax
import numpy as np


from fe import estimator
from timemachine.lib import LangevinIntegrator
from timemachine.lib import potentials
from parallel.client import CUDAPoolClient


def get_harmonic_bond(n_atoms, n_bonds):
    atom_idxs = np.arange(n_atoms)
    params = np.random.rand(n_bonds, 2).astype(np.float64)
    bond_idxs = []
    for _ in range(n_bonds):
        bond_idxs.append(np.random.choice(atom_idxs, size=2, replace=False))
    bond_idxs = np.array(bond_idxs, dtype=np.int32)
    lamb_mult = np.random.randint(-5, 5, size=n_bonds, dtype=np.int32)
    lamb_offset = np.random.randint(-5, 5, size=n_bonds, dtype=np.int32)
    return potentials.HarmonicBond(bond_idxs, lamb_mult, lamb_offset), params

def get_harmonic_angle(n_atoms, n_bonds):
    atom_idxs = np.arange(n_atoms)
    params = np.random.rand(n_bonds, 2).astype(np.float64)
    bond_idxs = []
    for _ in range(n_bonds):
        bond_idxs.append(np.random.choice(atom_idxs, size=3, replace=False))
    bond_idxs = np.array(bond_idxs, dtype=np.int32)
    lamb_mult = np.random.randint(-5, 5, size=n_bonds, dtype=np.int32)
    lamb_offset = np.random.randint(-5, 5, size=n_bonds, dtype=np.int32)
    return potentials.HarmonicAngle(bond_idxs, lamb_mult, lamb_offset), params

def test_free_energy_estimator():

    n_atoms = 5
    x0 = np.random.rand(n_atoms, 3)
    v0 = np.zeros_like(x0)

    n_bonds = 3
    n_angles = 4

    hb_pot, hb_params = get_harmonic_bond(n_atoms, n_bonds)
    ha_pot, ha_params = get_harmonic_angle(n_atoms, n_angles)

    sys_params = [hb_params, ha_params]
    unbound_potentials = [hb_pot, ha_pot]

    masses = np.random.rand(n_atoms)

    box = np.eye(3, dtype=np.float64)

    seed = 2021

    integrator = LangevinIntegrator(300, 1.5e-3, 1.0, masses, seed)

    lambda_schedule = np.linspace(0, 1.0, 4)

    for client in [None, CUDAPoolClient(1)]:

        mdl = estimator.FreeEnergyModel(
            unbound_potentials,
            client,
            box,
            x0,
            v0,
            integrator,
            lambda_schedule,
            100,
            100
        )

        value_and_grad_fn = jax.value_and_grad(estimator.deltaG, argnums=1)
        dG, sys_grad = value_and_grad_fn(mdl, sys_params) # run fwd, store result, and run bwd

        grad_fn = jax.grad(estimator.deltaG, argnums=1)
        grad = grad_fn(mdl, sys_params)

        assert len(grad) == 2
        assert grad[0].shape == sys_params[0].shape
        assert grad[1].shape == sys_params[1].shape