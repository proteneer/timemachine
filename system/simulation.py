import numpy as np
import random

from system import forcefield
from timemachine.lib import custom_ops
from timemachine.integrator import langevin_coefficients

from timemachine import constants

from simtk.openmm.app import PDBFile, DCDFile, Topology

import scipy

from collections import namedtuple

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

class FireDescentState(namedtuple(
    'FireDescentState',
    ['position', 'velocity', 'force', 'dt', 'alpha', 'n_pos'])):
  """
  A tuple containing state information for the Fire Descent minimizer.
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
    force_fn, dt_start=0.00001,
    dt_max=0.00005, n_min=5, f_inc=1.1, f_dec=0.5, alpha_start=0.1, f_alpha=0.99):
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

#         def sigmoid(x):
#             k = 100
#             return 1 / (1 + np.exp(-k*x))
            
#         sig = sigmoid(1000 * P)
        

        # NOTE(schsam): Can we clean this up at all?
        n_pos = np.where(P >= 0, n_pos + 1.0, 0.0)
#         n_pos = (n_pos + 1) * sig
        dt_choice = np.array([dt * f_inc, dt_max])
        dt = np.where(
            P > 0, np.where(n_pos > n_min, np.min(dt_choice), dt), dt)
#         dt = min(dt_max, min(dt_max, dt * f_inc * sig + dt * (1-sig)) * sigmoid((n_pos - n_min) * 1000)+ dt * (1 - sigmoid((n_pos - n_min) * 1000)))
        dt = np.where(P < 0, dt * f_dec, dt)
#         dt = dt * sig + dt * f_dec * (1 - sig) 
        alpha = np.where(
            P > 0, np.where(n_pos > n_min, alpha * f_alpha, alpha), alpha)
#         alpha = alpha * f_alpha * sig + alpha * (1 - sig) * sigmoid((n_pos - n_min) * 1000) + alpha * (1 - sigmoid((n_pos - n_min) * 1000))
        alpha = np.where(P < 0, alpha_start, alpha)
#         alpha = alpha_start * (1 - sig) + alpha * sig
        V = (P < 0) * np.zeros_like(V) + (P >= 0) * V
#         V = (1 - sig) * np.zeros_like(V) + (sig) * V

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
    pdb=None,
    pdb_name=None
    ):

    num_atoms = len(masses)
    
#     num_host_atoms = 2035

#     custom_electrostatics = None
#     if num_atoms > 2035:
#         for p, args in potentials:
#             if p == custom_ops.Electrostatics_f32:
#                 old_scale_matrix = args[0]
#                 new_scale_matrix = np.zeros_like(old_scale_matrix)
#                 new_scale_matrix[:num_host_atoms, :num_host_atoms] = old_scale_matrix[:num_host_atoms, :num_host_atoms]
#                 new_scale_matrix[num_host_atoms:, num_host_atoms:] = old_scale_matrix[num_host_atoms:, num_host_atoms:]

#                 custom_electrostatics = p(new_scale_matrix.astype(np.float32), args[1])
    
    potentials = forcefield.merge_potentials(potentials)
        
    dt = 1e-3
    ca, cb, cc = langevin_coefficients(
        temperature=25.0,
        dt=dt,
        friction=50,
        masses=masses
    )

    # m_dt, m_ca, m_cb, m_cc = dt, 0.5, cb, np.zeros_like(masses)

    m_dt, m_ca, m_cb, m_cc = dt, 0.9, np.ones_like(cb)/10000, np.zeros_like(masses)
    
    opt = custom_ops.LangevinOptimizer_f64(
        m_dt,
        m_ca,
        m_cb.astype(np.float64),
        m_cc.astype(np.float64)
    )

    v0 = np.zeros_like(conf)
    dp_idxs = dp_idxs.astype(np.int32)

    ctxt = custom_ops.Context_f64(
        potentials,
        opt,
        params.astype(np.float64),
        conf.astype(np.float64), # x0
        v0.astype(np.float64), # v0
        dp_idxs
    )

    # Minimize the system and carry the gradient over
    # call system converged when the delta is .25 kcal)
    # max_iter = 25000
    # window_size = 150
    # minimization_energies = []
    # def fun(x0):
    #     conf = x0.reshape(num_atoms, 3)
    #     tot_E, tot_dE_dx = ctxt.debug_compute_dE_dx(
    #         conf.astype(np.float32),
    #     )

    #     return tot_E.astype(np.float64), tot_dE_dx.reshape(-1).astype(np.float64)

    # scipy.optimize.minimize(fun, conf.reshape(-1), method='L-BFGS-B', jac=True, options={"disp": 1})
    # # scipy.optimize.minimize(fun, conf.reshape(-1), method='Newton-CG', jac=True, hess=hess, options={"disp": True})

    # assert 0
    
#     tot_dE_dx = 0
    tolerance = 10.0

    def mean_norm(conf):
        norm_x = np.dot(conf.reshape(-1), conf.reshape(-1))/num_atoms
#         if norm_x < 1:
#             raise ValueError("Starting norm is less than one")
        return np.sqrt(norm_x)
    
    x_norm = mean_norm(conf)
    x_norm = np.where(x_norm < 1, 1, x_norm)
    epsilon = tolerance/x_norm

#     def force_fn(conf):
#         nonlocal tot_dE_dx
#         _, tot_dE_dx = ctxt.debug_compute_dE_dx(
#             conf.astype(np.float32),
#         )        
#         # print(tot_dE_dx)
#         return -tot_dE_dx

#     init_fn, update_fn = fire_descent(force_fn)

#     state = init_fn(conf)
    
#     if pdb is not None:
#         outfile = open('md.dcd','wb')
# #         outfile = open("md.pdb", "w")
#         dcd = DCDFile(outfile, pdb.topology, .0001)
# #         PDBFile.writeHeader(pdb.topology, outfile)
    
#     state = update_fn(state)
    
# #     count = 0

#     init_fn, update_fn = fire_descent(force_fn, dt_start=0.0005,dt_max=0.001)

#     for step in range(max_iter):
# #         print("step", step)
#         state = update_fn(state)
#         g_norm = mean_norm(tot_dE_dx)
#         if g_norm < epsilon:
#             break
#         if step % 400 == 0:
#             print("step", step, "energy", ctxt.get_E(), "|g|", g_norm, "eps", epsilon)
#         if pdb is not None: 
#             if step % 50 == 0:
# #                 print(state.position.shape, len(pdb.getPositions()))
# #                 assert 0
#                 dcd.writeModel(state.position)
# #                 PDBFile.writeModel(pdb.topology, state.position*10, outfile, count)
# #                 count += 1

    # PDBFile.writeHeader(simulation.topology, self._out)

#     PDBFile.writeHeader(pdb.topology, outfile)
#     count = 0
    
#     if pdb is not None:
#         outfile = open(pdb_name + '.dcd','wb')
#         dcd = DCDFile(outfile, pdb.topology, .0001)
    nan = False
    max_iter = 15000
    for i in range(max_iter):
        dt *= 1.01
        dt = min(dt, 0.02)
        opt.set_dt(dt)
        ctxt.step()
        # minimization_energies.append(E)
        # if len(minimization_energies) > window_size:
            # window_std = np.std(minimization_energies[-window_size:])
            # if window_std < 1.046/2:
                # break
#         if i % 100 == 0:
#             E = ctxt.get_E()
#             print("i", i, dt, E)
        if i > 100 and i % 100 == 0:
            if np.isnan(ctxt.get_E()):
                nan = True
                break
#                 raise Exception("energy is nan")
            dE_dx = ctxt.get_dE_dx()
            g_norm = mean_norm(dE_dx)
            x_norm = mean_norm(conf)
            if g_norm < epsilon:
                break
#         if pdb is not None:
#             if i % 50 == 0:
#                 dcd.writeModel(ctxt.get_x())   
            
#     if custom_electrostatics is not None:
        
# #     for p in potentials:
# #         conf = ctxt.get_x()
#         conf = conf.reshape(1,-1,3)
# #         if p == custom_ops.Electrostatics_f32:
            
#         res = custom_electrostatics.derivatives(conf.astype(np.float32), params.astype(np.float32), np.array([0]).astype(np.int32))
#         print("electrostatic interaction energy", res[0])
        
#     for p in potentials:
#         conf = ctxt.get_x()
#         conf = conf.reshape(1,-1,3)
#         res = p.derivatives(conf.astype(np.float32), params.astype(np.float32), np.array([0]).astype(np.int32))
#         print(p, res[0])

#     for step in range(max_iter):
#         ctxt.step()
#         g_norm = mean_norm(ctxt.get_dE_dx())
#         if g_norm < epsilon:
#             break
#         if m_dt < .001:
#             m_dt = m_dt * 1.1
#             opt.set_dt(m_dt)
#         if step % 1000 == 0:
#             print("step", step, "energy", ctxt.get_E(), "|g|", g_norm, "eps", epsilon)
#         minimization_energies.append(E)
#         if len(minimization_energies) > window_size:
#             window_std = np.std(minimization_energies[-window_size:])
#             if window_std < 1.046/2:
#                 break
#         if i % 25 == 0:
#             PDBFile.writeModel(pdb.topology, ctxt.get_x()*10, outfile, count)
#             count += 1
#             # PDBFile.write()
#             # fh.write(write(ctxt.get_x()*10, masses))
#             # print("minimization", i, E)
#         if i > 5000:
#             break
    
#     PDBFile.writeFooter(pdb.topology, outfile)
#     outfile.flush()

    if np.isnan(ctxt.get_E()):
        nan = True
#         raise Exception("energy is nan")
#     if i == max_iter-1:
#         print("Energy minimization failed to converge in ", i, "steps")
#         raise Exception("Energy minimization failed to converge in ", i, "steps")
#     else:
    print("Minimization converged in", i, "steps to", ctxt.get_E())

    # #modify integrator to do dynamics
    # opt.set_dt(dt)
    # opt.set_coeff_a(ca)
    # opt.set_coeff_b(cb)
    # opt.set_coeff_c(cc)

    # # dynamics via reservoir sampling
    # k = n_samples # number of samples we want to keep
    # R = []
    # count = 0

    # for count in range(n_steps):

    #     # closure around R, and ctxt
    #     def get_reservoir_item(step):
    #         E = ctxt.get_E()
    #         dE_dx = ctxt.get_dE_dx()
    #         dx_dp = ctxt.get_dx_dp()
    #         dE_dp = ctxt.get_dE_dp()
    #         min_dx = np.amin(dx_dp)
    #         max_dx = np.amax(dx_dp)
    #         lhs = np.einsum('kl,mkl->m', dE_dx, dx_dp)
    #         total_dE_dp = lhs + dE_dp

    #         # print(step, total_dE_dp)

    #         limits = 1e5
    #         # if min_dx < -limits or max_dx > limits:
    #             # raise Exception("Derivatives blew up:", min_dx, max_dx)
    #         return [E, dE_dx, dx_dp, dE_dp, step]

    #     if count < k:
    #         R.append(get_reservoir_item(count))
    #     else:
    #         j = random.randint(0, count)
    #         if j < k:
    #             R[j] = get_reservoir_item(count)
    #             np.set_printoptions(suppress=True)

    #     if count % 5000 == 0:
    #         print("count", count)

    #     ctxt.step()
    R = [[
        ctxt.get_E(),
        ctxt.get_dE_dx(),
        ctxt.get_dx_dp(),
        ctxt.get_dE_dp(),
        ctxt.get_x()
    ]]
    
#     print(np.amax(abs(ctxt.get_dE_dx())))
#     print(np.amax(abs(ctxt.get_dx_dp())))
#     print(np.amax(abs(ctxt.get_dE_dp())))
    
    if nan:
        print("energy is nan")
        R = [[
            np.nan,
            np.zeros_like(ctxt.get_dE_dx()),
            np.zeros_like(ctxt.get_dx_dp()),
            np.zeros_like(ctxt.get_dE_dp()),
            ctxt.get_x()
        ]]

    return R
