import numpy as np
import tensorflow as tf
import force
from constants import BOLTZ

class LangevinIntegrator():

    def __init__(self,
        masses,
        num_params,
        dt=0.0025,
        friction=1.0,
        temp=300.0,
        precision=tf.float64,
        disable_noise=False,
        buffer_size=None):
        """
        Langevin Integrator

        This is inspired by ReferenceStochasticDynamics.cpp from OpenMM.

        """
        self.dt = dt
        self.friction = friction # dissipation (how fast we forget)
        self.temperature = temp  # temperature

        self.disable_noise = disable_noise
        self.vscale = np.exp(-self.dt*self.friction)

        if self.friction == 0:
            self.fscale = self.dt
        else:
            self.fscale = (1-self.vscale)/self.friction

        kT = BOLTZ * self.temperature
        self.nscale = np.sqrt(kT*(1-self.vscale*self.vscale)) # noise scale
        self.normal = tf.distributions.Normal(loc=0.0, scale=1.0)
        self.invMasses = (1.0/masses).reshape((-1, 1))
        self.sqrtInvMasses = np.sqrt(self.invMasses)

        self.ca = tf.cast(self.vscale, dtype=precision)
        self.cbs = []
        for m in masses:
            self.cbs.append((self.fscale*self.dt)/m)

        self.cbs = tf.convert_to_tensor(self.cbs, dtype=precision)
        self.cbs = tf.reshape(self.cbs, shape=(1, -1, 1))

        if buffer_size is None:
            self.steps_to_converge = 4000 # guestimate later
        else:
            self.steps_to_converge = buffer_size

        # pre-compute scaled prefactors once at the beginning
        self.scale = (1-tf.pow(self.ca, tf.range(self.steps_to_converge, dtype=precision)+1))/(1-self.ca)
        self.scale = tf.reverse(self.scale, [0])
        self.scale = tf.reshape(self.scale, (-1, 1, 1, 1))

        self.num_atoms = len(masses)
        self.num_params = num_params

        # buffer for accumulated velocities
        self.v_t = tf.get_variable(
            "buffer_velocity",
            shape=(self.num_atoms, 3),
            dtype=precision,
            initializer=tf.initializers.zeros)

        # buffer for current step's dxdp
        self.dxdp_t = tf.get_variable(
            "buffer_dxdp",
            shape=(self.num_params, self.num_atoms, 3),
            dtype=precision,
            initializer=tf.initializers.zeros)

        # buffer for unconverged zetas
        self.buffer_zetas = tf.get_variable(
            "buffer_zetas",
            shape=(self.steps_to_converge, self.num_params, self.num_atoms, 3),
            dtype=precision,
            initializer=tf.initializers.zeros)

        # buffer for converged zetas
        self.converged_zetas = tf.get_variable(
            "converged_zetas",
            shape=(self.num_params, self.num_atoms, 3),
            dtype=precision,
            initializer=tf.initializers.zeros)


    def step(self, x_t, energies):
        """
        Advance x_t in to x_{t+1}
        """

        # buffers for each energy/force type
        gs = []
        hs = []
        es = []
        for e in energies:
            es.append(e.energy(x_t))
            gs.append(e.gradients(x_t))
            hs.append(e.hessians(x_t))

        tot_e = tf.reduce_sum(es)
        grads = tf.reduce_sum(gs, axis=0)
        hessians = tf.reduce_sum(hs, axis=0)

        num_dims = 3
        num_atoms = self.num_atoms

        if self.disable_noise:
            noise = 0
        else:
            noise = self.normal.sample((num_atoms, num_dims))

        new_v_t = self.vscale*self.v_t - self.fscale*self.invMasses*grads + self.nscale*self.sqrtInvMasses*noise
        self.v_t = tf.assign(self.v_t, new_v_t)

        # compute dxs
        dx = self.v_t * self.dt

        # The Algorithm:

        # 1. Given starting geometry and velocity x_t, v_t, dxdp_t, zetas_[0, t)
        # 2. Compute grads, hessians, and mixed partials: g_t, h_t, mp_t
        # 3. Using previous zetas_[0, t), first compute zeta_t for time t using h_t, mp_t, dxdp_t to compute zeta_t, appending
        #    zeta_t to list of zetas
        # 4. Return new geometry and dxdp

        # compute dxdp, derivative of the geometry with respect to parameters
        # This uses only [0, t) and doesn't require zeta of the current step

        mixed_partials = []
        for e in energies:
            mp = e.mixed_partials(x_t)
            mixed_partials.extend(mp)

        mixed_partials = tf.stack(mixed_partials) # [num_params, num_atoms, 3]

        # Rolling Algorithm, insertion is at the front, highest exponent is at the front:
        # buf size: 5         buf   start_idx
        
        # step 0: [-.-,-,-,0] -     0
        # step 1: [-,-,-,0,1] -     0
        # step 2: [-,-,0,1,2] -     0
        # step 3: [-,0,1,2,3] -     0
        # step 4: [0,1,2,3,4] -     0
        # step 5: [1,2,3,4,5] 0     1
        # step 6: [2,3,4,5,6] 0+1   2
        # step 7: [3,4,5,6,7] 0+1+2 3

        # 1. Accumulate first element in buffer into converged zetas
        # 2. Insert to the front
        # 3. Cyclic-roll buffer left

        # compute zeta_t for time t
        # a:  0 1 2 3            0 1 2
        # h: [N,3,N,3], dxdp_t: [p,N,3], contraction: [p,N,3], mp: [p,N,3]
        contraction = tf.einsum('ijkl,mkl->mij', hessians, self.dxdp_t)
        zeta_t = contraction + mixed_partials

        self.converged_zetas = tf.assign_add(self.converged_zetas, self.buffer_zetas[0])
        with tf.control_dependencies([self.converged_zetas, self.dxdp_t]):
            override = self.buffer_zetas[0].assign(zeta_t)
            self.buffer_zetas = tf.assign(
                self.buffer_zetas,
                tf.roll(override, shift=-1, axis=0)
            )

            # compute dxdp_{t+1} using unconverged zeta and converged zetas
            new_dxdp_t = self.buffer_zetas * self.scale
            new_dxdp_t = tf.reduce_sum(new_dxdp_t, axis=0)
            new_dxdp_t += self.converged_zetas * self.scale[0][0][0][0]
            new_dxdp_t *= -self.cbs

            self.dxdp_t = tf.assign(self.dxdp_t, new_dxdp_t)

            # (ytz): note we *MUST* return self.dxdp_t for this not to have a dependency hell
            return dx, self.dxdp_t

if __name__ == "__main__":

    hb = force.HarmonicBondForce()
    intg = LangevinIntegrator(
        masses=np.array([1.0, 12.0]),
        friction=10.0,
        dt=0.03,
        num_params=len(hb.params()))
    x0 = np.array([[1.0, 0.5, -0.5], [0.2, 0.1, -0.3]])

    x_ph = tf.placeholder(dtype=tf.float64, shape=(None, 3))

    dx, dxdps, zt, tmp = intg.step(x_ph, [hb])

    sess = tf.Session()
    sess.run(tf.initializers.global_variables())

    x = x0

    for step in range(10):
        dx_val, dp_val, zt_val, tmp_val = sess.run([dx, dxdps, zt, tmp], feed_dict={x_ph: x})
        x = x + dx_val
