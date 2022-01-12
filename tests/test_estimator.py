import numpy as np


from fe import estimator_abfe
from timemachine.lib import LangevinIntegrator, potentials, MonteCarloBarostat
from parallel.client import CUDAPoolClient
from md.barostat.utils import get_bond_list, get_group_indices


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


def get_harmonic_restraints(n_atoms, n_restraints):
    assert n_restraints * 2 <= n_atoms
    params = np.random.rand(n_restraints, 2).astype(np.float64)
    bond_idxs_src = []
    bond_idxs_dst = []

    atom_idxs_src = np.arange(n_atoms // 2)
    atom_idxs_dst = np.arange(n_atoms // 2) + n_atoms // 2

    bond_idxs_src = np.random.choice(atom_idxs_src, size=n_restraints, replace=False)
    bond_idxs_dst = np.random.choice(atom_idxs_dst, size=n_restraints, replace=False)

    bond_idxs = np.array([bond_idxs_src, bond_idxs_dst], dtype=np.int32).T
    lamb_mult = np.random.randint(-5, 5, size=n_restraints, dtype=np.int32)
    lamb_offset = np.random.randint(-5, 5, size=n_restraints, dtype=np.int32)
    return potentials.HarmonicBond(bond_idxs, lamb_mult, lamb_offset), params


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

    group_idxs = get_group_indices(get_bond_list(hb_pot))

    temperature = 300.0
    pressure = 1.0

    integrator = LangevinIntegrator(temperature, 1.5e-3, 1.0, masses, seed)

    barostat = MonteCarloBarostat(x0.shape[0], pressure, temperature, group_idxs, 25, seed)

    beta = 0.125

    lambda_schedule = np.linspace(0, 1.0, 4)

    def loss_fn(sys_params):

        endpoint_correct = False
        mdl = estimator_abfe.FreeEnergyModel(
            unbound_potentials,
            endpoint_correct,
            client,
            box,
            x0,
            v0,
            integrator,
            barostat,
            lambda_schedule,
            100,
            100,
            beta,
            "test",
        )

        dG, bar_dG_err, results = estimator_abfe.deltaG(mdl, sys_params)

        return dG ** 2

    for client in [None, CUDAPoolClient(1)]:
        dG = loss_fn(sys_params)


def test_free_energy_estimator_with_endpoint_correction():
    """
    Test that we generate correctly shaped derivatives in the estimator code
    when the endpoint correction is turned on. We expected that f([a,b,c,...])
    to generate derivatives df/da, df/db, df/dc, df/d... such that
    df/da.shape == a.shape, df/db.shape == b.shape, df/dc == c.shape, and etc.
    """

    n_atoms = 15
    x0 = np.random.rand(n_atoms, 3)
    v0 = np.zeros_like(x0)

    n_bonds = 3
    n_angles = 4
    n_restraints = 5

    hb_pot, hb_params = get_harmonic_bond(n_atoms, n_bonds)
    ha_pot, ha_params = get_harmonic_angle(n_atoms, n_angles)
    rs_pot, rs_params = get_harmonic_restraints(n_atoms, n_restraints)

    sys_params = [hb_params, ha_params, rs_params]
    unbound_potentials = [hb_pot, ha_pot, rs_pot]

    masses = np.random.rand(n_atoms)

    box = np.eye(3, dtype=np.float64)

    seed = 2021

    group_idxs = get_group_indices(get_bond_list(hb_pot))

    temperature = 300.0
    pressure = 1.0

    integrator = LangevinIntegrator(temperature, 1.5e-3, 1.0, masses, seed)

    barostat = MonteCarloBarostat(x0.shape[0], pressure, temperature, group_idxs, 25, seed)

    beta = 0.125

    lambda_schedule = np.linspace(0, 1.0, 4)

    def loss_fn(sys_params):

        endpoint_correct = True
        mdl = estimator_abfe.FreeEnergyModel(
            unbound_potentials,
            endpoint_correct,
            client,
            box,
            x0,
            v0,
            integrator,
            barostat,
            lambda_schedule,
            100,
            100,
            beta,
            "test",
        )

        dG, bar_dG_err, results = estimator_abfe.deltaG(mdl, sys_params)

        return dG ** 2

    for client in [None, CUDAPoolClient(1)]:
        dG = loss_fn(sys_params)
