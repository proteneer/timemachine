import numpy as np
from jax import grad, jit

from timemachine.constants import BOLTZ, DEFAULT_FF
from timemachine.fe.utils import to_md_units
from timemachine.ff import Forcefield
from timemachine.integrator import VelocityVerletIntegrator
from timemachine.lib.potentials import SummedPotential
from timemachine.md.enhanced import VacuumState
from timemachine.md.moves import HMCMove
from timemachine.testsystems.dhfr import setup_dhfr
from timemachine.testsystems.relative import hif2a_ligand_pair


def prepare_vacuum_hmc_move(mol, ff, dt=1e-3, n_steps=100, temperature=300.0, seed=2022):
    """
    Notes:
    ------
    * uses constant diagonal mass matrix, with masses[i] = mean(get_masses(mol)) for all i
    """

    # prepare U fxn, force fxn
    vacuum_state = VacuumState(mol, ff)
    vacuum_U_fxn = vacuum_state.U_full
    vacuum_force_fxn = lambda x: -grad(vacuum_U_fxn)(x)

    # use constant diagonal mass matrix
    _physical_masses = np.array([atom.GetMass() for atom in mol.GetAtoms()])
    vacuum_masses = np.mean(_physical_masses) * np.ones(len(_physical_masses))

    # define velocity verlet update fxn
    vacuum_intg = VelocityVerletIntegrator(vacuum_force_fxn, masses=vacuum_masses, dt=dt)

    @jit
    def vv_update(x0, v0):
        return vacuum_intg._update_via_fori_loop(x0, v0, n_steps=n_steps)

    # put it all together
    hmc_move = HMCMove(vacuum_U_fxn, vv_update, vacuum_masses, temperature, seed=seed)

    return hmc_move


def test_hmc_on_jax_system():
    """Run HMC on a ligand in vacuum (using Jax reference potentials), assert acceptance rate > 95% when dt = 1 fs"""
    seed = 2022

    np.random.seed(seed)

    # setup
    mol = hif2a_ligand_pair.mol_a
    ff = Forcefield.load_from_file(DEFAULT_FF)

    hmc_move = prepare_vacuum_hmc_move(mol, ff, dt=1e-3, n_steps=100, seed=seed)
    x0 = 0.1 * np.array(mol.GetConformer(0).GetPositions())

    # run a few iterations
    traj = [x0]
    for _ in range(100):
        traj.append(hmc_move.move(traj[-1]))

    # assert we moved substantially
    assert traj[-1].shape == x0.shape
    assert np.max(np.abs(traj[-1] - traj[0])) > 0.1

    # assert high acceptance fraction
    assert hmc_move.acceptance_fraction > 0.95

    # assert energy didn't blow up
    U_before = hmc_move.U_fxn(x0)
    U_after = hmc_move.U_fxn(traj[-1])
    assert abs(U_after - U_before) < 1000

    return hmc_move


def prepare_tm_hmc_move(bound_potentials, masses, box, lam=0.0, dt=1e-3, n_steps=100, temperature=300.0, seed=2022):

    # define potential and force fxn
    params = [f.params for f in bound_potentials]
    summed_potential = SummedPotential(bound_potentials, params)
    summed_potential.bind(np.hstack([p.flatten() for p in params]))
    summed_bound_impl = summed_potential.bound_impl(np.float32)

    def U(coords):
        return summed_bound_impl.execute(coords, box, lam)[-1]

    def force(coords):
        du_dx = summed_bound_impl.execute(coords, box, lam)[0]
        return -du_dx

    # define velocity verlet update
    def vv_update_fxn(coords, velocities):
        intg = VelocityVerletIntegrator(force, masses, dt=dt)
        xs, vs = intg.multiple_steps(coords, velocities, n_steps=n_steps)
        return xs[-1], vs[-1]

    # put it all together
    hmc_move = HMCMove(U, vv_update_fxn, masses, temperature, seed=seed)
    return hmc_move


def test_hmc_on_dhfr():
    """Run HMC on dhfr (using SummedPotential), assert acceptance rate > 50% when dt = 0.001 fs"""
    seed = 2022
    np.random.seed(seed)

    bound_potentials, mass_list, unitted_x0, box = setup_dhfr()
    x0 = np.array([[to_md_units(x), to_md_units(y), to_md_units(z)] for x, y, z in unitted_x0])

    # note: need a *very* small timestep for large systems!
    n_particles = len(mass_list)  # n = 23558
    dt = 0.001e-3  # ~2-4 orders of magnitude smaller than stable timestep for unadjusted Langevin!

    # anecdote: compared physical_masses, constant_masses, hmr_masses, but none make a dramatic difference here
    # -- going with a constant for simplicity
    physical_masses = np.array(mass_list)
    constant_masses = np.ones(n_particles) * np.mean(physical_masses)

    hmc_move = prepare_tm_hmc_move(bound_potentials, constant_masses, box, dt=dt, n_steps=10, seed=seed)

    # run a few iterations
    traj = [x0]
    for _ in range(100):
        traj.append(hmc_move.move(traj[-1]))

    # assert we moved at least a little
    assert traj[-1].shape == x0.shape
    assert np.max(np.abs(traj[-1] - traj[0])) > 0.0

    # assert we accepted at least a little
    assert hmc_move.acceptance_fraction > 0.5

    # assert energy didn't blow up
    U_before = hmc_move.U_fxn(x0)
    U_after = hmc_move.U_fxn(traj[-1])
    assert abs(U_after - U_before) < 1000


def prepare_quartic_hmc_move(num_oscillators=1000, dt=1.5e-3, n_steps=1000, temperature=300.0):
    masses = np.ones(num_oscillators)

    @jit
    def U(x):
        return np.sum(x ** 4)

    def force(x):
        return -grad(U)(x)

    intg = VelocityVerletIntegrator(force, masses=masses, dt=dt)

    @jit
    def vv_update(x0, v0):
        return intg._update_via_fori_loop(x0, v0, n_steps=n_steps)

    hmc_move = HMCMove(U, vv_update, masses, temperature)

    return hmc_move


def test_hmc_on_quartic_potential():
    """Run HMC on a system of independent quartic oscillators, assert canonical sampling"""

    np.random.seed(2022)

    num_oscillators = 10000

    x0 = np.random.randn(num_oscillators, 3)
    temperature = 300.0
    hmc_move = prepare_quartic_hmc_move(num_oscillators, dt=0.05, temperature=temperature)

    # run a few iterations
    traj = [x0]
    for _ in range(100):
        traj.append(hmc_move.move(traj[-1]))

    # assert we moved
    assert traj[-1].shape == x0.shape
    assert np.max(np.abs(traj[-1] - traj[0])) > 0.1

    # assert high acceptance fraction
    assert hmc_move.acceptance_fraction > 0.5, hmc_move.acceptance_fraction

    # assert good sampling
    samples = np.array(traj[10:]).flatten()

    # summarize using histogram
    y_empirical, edges = np.histogram(samples, bins=100, range=(-2, +2), density=True)
    x_grid = (edges[1:] + edges[:-1]) / 2

    # compare with e^{-U(x) / kB T} / Z
    threshold = 1e-4

    y = np.exp(-(x_grid ** 4) / (BOLTZ * temperature))
    y_ref = y / np.trapz(y, x_grid)

    histogram_mse = np.mean((y_ref - y_empirical) ** 2)

    assert histogram_mse < threshold
