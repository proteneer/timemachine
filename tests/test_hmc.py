import numpy as np
from jax import grad, jit

from timemachine.constants import DEFAULT_FF
from timemachine.ff import Forcefield
from timemachine.integrator import VelocityVerletIntegrator
from timemachine.md.enhanced import VacuumState
from timemachine.md.moves import HMCMove
from timemachine.testsystems.relative import hif2a_ligand_pair


def prepare_vacuum_hmc_move(mol, ff, dt=1e-3, n_steps=100, temperature=300.0):
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
    hmc_move = HMCMove(vacuum_U_fxn, vv_update, vacuum_masses, temperature)

    return hmc_move


def test_hmc_on_jax_system():
    """Run HMC on a ligand in vacuum (using Jax reference potentials), assert the output is not dramatically bad"""
    np.random.seed(2022)

    # setup
    mol = hif2a_ligand_pair.mol_a
    ff = Forcefield.load_from_file(DEFAULT_FF)

    hmc_move = prepare_vacuum_hmc_move(mol, ff, dt=1e-3, n_steps=100)
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
