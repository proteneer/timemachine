import numpy as np
import random

from system import forcefield
from timemachine.lib import custom_ops
from timemachine.integrator import langevin_coefficients

from timemachine import constants

from simtk.openmm.app import PDBFile

import scipy

def average_E_and_derivatives(reservoir):
    """
    Compute the average energy and derivatives

    Parameters
    ----------
    reservoir: list of reservoir
        [
            [E, dE_dx, dx_dp, dE_dp],
            [E, dE_dx, dx_dp, dE_dp],
            ...
        ]

    Returns
    -------
    Average energy, analytic total derivative, and thermodynamic gradient

    """
    running_sum_total_derivs = None
    running_sum_E = 0
    n_reservoir = len(reservoir)

    running_sum_dE_dp = None
    running_sum_EmultdE_dp = None

    for E, dE_dx, dx_dp, dE_dp, _ in reservoir:
        if running_sum_total_derivs is None:
            running_sum_total_derivs = np.zeros_like(dE_dp)
        if running_sum_dE_dp is None:
            running_sum_dE_dp = np.zeros_like(dE_dp)
        if running_sum_EmultdE_dp is None:
            running_sum_EmultdE_dp = np.zeros_like(dE_dp)

        # tensor contract [N,3] with [P, N, 3] and dE_d
        total_dE_dp = np.einsum('kl,mkl->m', dE_dx, dx_dp) + dE_dp
        running_sum_total_derivs += total_dE_dp
        running_sum_E += E


        running_sum_dE_dp += dE_dp
        running_sum_EmultdE_dp += E*dE_dp

    # compute the thermodynamic average:
    # boltz*(<E><dE/dp> - <E.dE/dp>)
    thermo_deriv = running_sum_E*running_sum_dE_dp - running_sum_EmultdE_dp

    return running_sum_E/n_reservoir, running_sum_total_derivs/n_reservoir, -constants.BOLTZ*(thermo_deriv/n_reservoir)/(100)



from collections import namedtuple

class FireDescentState(namedtuple(
    'FireDescentState',
    ['position', 'velocity', 'force', 'dt', 'alpha', 'n_pos'])):
  """A tuple containing state information for the Fire Descent minimizer.
  Attributes:
    position: The current position of particles. An ndarray of floats
      with shape [n, spatial_dimension].
    velocity: The current velocity of particles. An ndarray of floats
      with shape [n, spatial_dimension].
    force: The current force on particles. An ndarray of floats
      with shape [n, spatial_dimension].
    dt: A float specifying the current step size.
    alpha: A float specifying the current momentum.
    n_pos: The number of steps in the right direction, so far.
  """

  def __new__(cls, position, velocity, force, dt, alpha, n_pos):
    return super(FireDescentState, cls).__new__(
        cls, position, velocity, force, dt, alpha, n_pos)

def fire_descent(
    force_fn, dt_start=0.00025,
    dt_max=0.001, n_min=5, f_inc=1.1, f_dec=0.5, alpha_start=0.1, f_alpha=0.99):
    """Defines FIRE minimization.
    This code implements the "Fast Inertial Relaxation Engine" from [1].
    Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      [n, spatial_dimension].
    shift_fn: A function that displaces positions, R, by an amount dR. Both R
      and dR should be ndarrays of shape [n, spatial_dimension].
    quant: Either a quantity.Energy or a quantity.Force specifying whether
      energy_or_force is an energy or force respectively.
    dt_start: The initial step size during minimization as a float.
    dt_max: The maximum step size during minimization as a float.
    n_min: An integer specifying the minimum number of steps moving in the
      correct direction before dt and f_alpha should be updated.
    f_inc: A float specifying the fractional rate by which the step size
      should be increased.
    f_dec: A float specifying the fractional rate by which the step size
      should be decreased.
    alpha_start: A float specifying the initial momentum.
    f_alpha: A float specifying the fractional change in momentum.
    Returns:
    See above.
    [1] Bitzek, Erik, Pekka Koskinen, Franz Gahler, Michael Moseler,
      and Peter Gumbsch. "Structural relaxation made simple."
      Physical review letters 97, no. 17 (2006): 170201.
    """

    def init_fun(R, **kwargs):
        V = np.zeros_like(R)
        return FireDescentState(
            R, V, force_fn(R), dt_start, alpha_start, 0)

    def apply_fun(state, **kwargs):
        R, V, F_old, dt, alpha, n_pos = state
        R = R + dt * V + dt ** 2 * F_old
        F = force_fn(R)
        V = V + dt * 0.5 * (F_old + F)

        # NOTE(schsam): This will be wrong if F_norm ~< 1e-8.
        # TODO(schsam): We should check for forces below 1e-6. @ErrorChecking
        F_norm = np.sqrt(np.sum(F ** 2))
        V_norm = np.sqrt(np.sum(V ** 2))

        P = np.array(np.dot(np.reshape(F, (-1)), np.reshape(V, (-1))))

        V = V + alpha * (F * V_norm / F_norm - V)
        # print(alpha * (F * V_norm / F_norm - V))
        # NOTE(schsam): Can we clean this up at all?
        n_pos = np.where(P >= 0, n_pos + 1.0, 0)
        dt_choice = np.array([dt * f_inc, dt_max])
        dt = np.where(
            P > 0, np.where(n_pos > n_min, np.min(dt_choice), dt), dt)
        dt = np.where(P < 0, dt * f_dec, dt)
        alpha = np.where(
            P > 0, np.where(n_pos > n_min, alpha * f_alpha, alpha), alpha)
        alpha = np.where(P < 0, alpha_start, alpha)
        V = (P < 0) * np.zeros_like(V) + (P >= 0) * V

        return FireDescentState(R, V, F, dt, alpha, n_pos)

    return init_fun, apply_fun


def simple_fire_descent(
    force_fn, dt_start=0.00025,
    dt_max=0.001, n_min=1, f_inc=1.1, f_dec=0.5, alpha_start=0.1, f_alpha=0.99):
    """Defines FIRE minimization.
    This code implements the "Fast Inertial Relaxation Engine" from [1].
    Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      [n, spatial_dimension].
    shift_fn: A function that displaces positions, R, by an amount dR. Both R
      and dR should be ndarrays of shape [n, spatial_dimension].
    quant: Either a quantity.Energy or a quantity.Force specifying whether
      energy_or_force is an energy or force respectively.
    dt_start: The initial step size during minimization as a float.
    dt_max: The maximum step size during minimization as a float.
    n_min: An integer specifying the minimum number of steps moving in the
      correct direction before dt and f_alpha should be updated.
    f_inc: A float specifying the fractional rate by which the step size
      should be increased.
    f_dec: A float specifying the fractional rate by which the step size
      should be decreased.
    alpha_start: A float specifying the initial momentum.
    f_alpha: A float specifying the fractional change in momentum.
    Returns:
    See above.
    [1] Bitzek, Erik, Pekka Koskinen, Franz Gahler, Michael Moseler,
      and Peter Gumbsch. "Structural relaxation made simple."
      Physical review letters 97, no. 17 (2006): 170201.
    """

    def init_fun(R, **kwargs):
        V = np.zeros_like(R)
        return FireDescentState(
            R, V, force_fn(R), dt_start, alpha_start, 0)

    def apply_fun(state, **kwargs):
        R, V, F_old, dt, alpha, n_pos = state
        
        print(dt, alpha)

        R = R + dt * V + dt ** 2 * F_old
        F = force_fn(R)
        V = V + dt * 0.5 * (F_old + F)

        # NOTE(schsam): This will be wrong if F_norm ~< 1e-8.
        # TODO(schsam): We should check for forces below 1e-6. @ErrorChecking
        F_norm = np.sqrt(np.sum(F ** 2))
        V_norm = np.sqrt(np.sum(V ** 2))

        P = np.array(np.dot(np.reshape(F, (-1)), np.reshape(V, (-1))))
        V = V + alpha * (F * V_norm / F_norm - V)
        n_pos = np.where(P >= 0, n_pos + 1.0, 0)
        dt_choice = np.array([dt * f_inc, dt_max])

        dt = np.where(
            P > 0, np.where(n_pos > n_min, np.min(dt_choice), dt), dt)
        dt = np.where(P < 0, dt * f_dec, dt)
        alpha = np.where(
            P > 0, np.where(n_pos > n_min, alpha * f_alpha, alpha), alpha)
        alpha = np.where(P < 0, alpha_start, alpha)
        V = (P < 0) * np.zeros_like(V) + (P >= 0) * V

        return FireDescentState(R, V, F, dt, alpha, n_pos)

    return init_fun, apply_fun


def write(xyz, masses):
    buf = str(len(masses)) + '\n'
    buf += 'timemachine\n'
    for m, (x,y,z) in zip(masses, xyz):
        if int(round(m)) == 12:
            symbol = 'C'
        elif int(round(m)) == 14:
            symbol = 'N'
        elif int(round(m)) == 16:
            symbol = 'O'
        elif int(round(m)) == 32:
            symbol = 'S'
        elif int(round(m)) == 35:
            symbol = 'Cl'
        elif int(round(m)) == 1:
            symbol = 'H'
        else:
            raise Exception("Unknown mass:" + str(m))

        buf += symbol + ' ' + str(round(x,5)) + ' ' + str(round(y,5)) + ' ' +str(round(z,5)) + '\n'
    return buf

def run_simulation(
    potentials,
    params,
    param_groups,
    conf,
    masses,
    dp_idxs,
    n_samples,
    pdb):

    num_atoms = len(masses)

    potentials = forcefield.merge_potentials(potentials)
        
    dt = 0.00001
    ca, cb, cc = langevin_coefficients(
        temperature=25.0,
        dt=dt,
        friction=50,
        masses=masses
    )

    # m_dt, m_ca, m_cb, m_cc = dt, 0.5, cb, np.zeros_like(masses)
    m_dt, m_ca, m_cb, m_cc = dt, 0.9, np.ones_like(cb)/10000, np.zeros_like(masses)

    opt = custom_ops.LangevinOptimizer_f32(
        m_dt,
        m_ca,
        m_cb.astype(np.float32),
        m_cc.astype(np.float32)
    )

    v0 = np.zeros_like(conf)
    dp_idxs = dp_idxs.astype(np.int32)

    ctxt = custom_ops.Context_f32(
        potentials,
        opt,
        params.astype(np.float32),
        conf.astype(np.float32), # x0
        v0.astype(np.float32), # v0
        dp_idxs
    )

    tolerance = 10

    def mean_norm(conf):
        norm_x = np.dot(conf.reshape(-1), conf.reshape(-1))/num_atoms
        if norm_x < 1:
            norm_x = 1
            # raise ValueError("Starting norm is less than one")
        return np.sqrt(norm_x)

    epsilon = tolerance/mean_norm(conf)

    # outfile = open("md_langevin.pdb", "w")
    # PDBFile.writeHeader(pdb.topology, outfile)

    count = 0
    max_iter = 5000
    for i in range(max_iter):
        dt *= 1.005
        dt = min(dt, 0.02)
        opt.set_dt(dt)
        ctxt.step()

        # minimization_energies.append(E)
        # if len(minimization_energies) > window_size:
            # window_std = np.std(minimization_energies[-window_size:])
            # if window_std < 1.046/2:
                # break
        if i % 100 == 0:
            E = ctxt.get_E()
            dE_dx = ctxt.get_dE_dx()
            g_norm = mean_norm(dE_dx)
            x_norm = mean_norm(conf)
            # print("i", i, dt, E, "|g|", g_norm)
            if g_norm < epsilon * x_norm:
                break
            # print("shapes", pdb.topology.getNumAtoms(), state.position.shape)
            # PDBFile.writeModel(pdb.topology, ctxt.get_x()*10, outfile, count)
            count += 1
            # PDBFile.write()
            # fh.write(write(ctxt.get_x()*10, masses))
            # print("minimization", i, E)
        # if i > 5000:
            # break
    
    # PDBFile.writeFooter(pdb.topology, outfile)
    # outfile.flush()

    if i == max_iter-1:
        raise Exception("Energy minimization failed to converge in ", i, "steps")
    else:
        print("Minimization converged in", i, "steps to", E)

    R = [[
        ctxt.get_E(),
        ctxt.get_dE_dx(),
        ctxt.get_dx_dp(),
        ctxt.get_dE_dp(),
        0
    ]]

    return R, ctxt.get_x()
