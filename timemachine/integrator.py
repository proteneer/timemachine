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
        disable_noise=False):
        """
        Langevin Integrator

        This is a mirror implementation of ReferenceStochasticDynamics.cpp from OpenMM


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

        self.ca = tf.cast(self.vscale, dtype=tf.float32)
        self.cbs = []
        for m in masses:
            self.cbs.append(-self.fscale*self.dt/m)
        self.cbs = np.array(self.cbs, dtype=np.float32)
        self.cbs = tf.reshape(self.cbs, shape=(1, -1, 1))
        self.steps_to_converge = 4000 # guestimate later

        # pre-compute scaled prefactors once at the beginning
        self.scale = (1-tf.pow(self.ca, tf.range(self.steps_to_converge, dtype=tf.float32)+1))/(1-self.ca)
        self.scale = tf.reverse(self.scale, [0])
        self.scale = tf.reshape(self.scale, (-1, 1, 1, 1))


        num_atoms = 2
        self.v_t = tf.get_variable(
            "buffer_velocity",
            shape=(num_atoms, 3),
            dtype=np.float32,
            initializer=tf.initializers.zeros
            )
        # buffer for unconverged zetas
        self.buffer_zetas = tf.get_variable(
            "buffer_zetas",
            shape=(self.steps_to_converge, num_params, num_atoms, 3),
            dtype=np.float32,
            initializer=tf.initializers.zeros)

        # buffer for converged zetas
        self.converged_zetas = tf.get_variable(
            "converged_zetas",
            shape=(num_params, num_atoms, 3),
            dtype=np.float32,
            initializer=tf.initializers.zeros)


    def step(self, x, energies):

        # buffers for each energy/force type
        gs = []
        hs = []
        es = []
        for e in energies:
            es.append(e.energy(x))
            gs.append(e.gradients(x))
            hs.append(e.hessians(x))

        tot_e = tf.reduce_sum(es)
        grads = tf.reduce_sum(gs, axis=0)
        hessians = tf.reduce_sum(hs, axis=0)

        num_atoms = 2
        num_dims = 3

        # if self.v_t is None:
            # self.v_t = tf.zeros((num_atoms, num_dims))

        noise = 0
        # noise = self.normal.sample((num_atoms, num_dims))

        new_v_t = self.vscale*self.v_t - self.fscale*self.invMasses*grads + self.nscale*self.sqrtInvMasses*noise
        self.v_t = tf.assign(self.v_t, new_v_t)

        # compute dxs
        dx = self.v_t * self.dt

        # The Algorithm:

        # 1. Given starting geometry and velocity x_t and v_t, dxdp_{t-1}, zetas_[0, t)
        # 2. Compute grads, hessians, and mixed partials: g_t, h_t, mp_t
        # 3. Using previous zetas_[0, t), first compute dxdp_t and then use h_t, mp_t, dxdp_t to compute zeta_t, appending
        #    zeta_t to list of zetas
        # 4. Return new geometry and dxdp

        # compute dxdp, derivative of the geometry with respect to parameters
        # This uses only [0, t) and doesn't require zeta of the current step

        mixed_partials = []
        for e in energies:
            mixed_partials.extend(e.mixed_partials(x))

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

        # compute dxdp_t using unconverged zeta
        dxdp_t = self.buffer_zetas * self.scale
        dxdp_t = tf.reduce_sum(dxdp_t, axis=0)
        # add remainder from dxdp_t using converged zeta
        
        dxdp_t += self.converged_zetas * self.scale[0][0][0][0]

        dxdp_t *= -self.cbs * self.dt 

        # compute zeta_t for this particular step
        # a:  0 1 2 3            0 1 2
        # h: [N,3,N,3], dxdp_t: [p,N,3], contraction: [p,N,3], mp: [p,N,3]
        contraction = tf.einsum('ijkl,mkl->mij', hessians, dxdp_t)
        zeta_t = contraction + mixed_partials

        self.converged_zetas = tf.assign_add(self.converged_zetas, self.buffer_zetas[0])
        with tf.control_dependencies([self.converged_zetas, dxdp_t]):
            override = self.buffer_zetas[0].assign(zeta_t)
            self.buffer_zetas = tf.assign(self.buffer_zetas, tf.roll(override, shift=-1, axis=0))

        return dx, dxdp_t, gs, self.buffer_zetas

if __name__ == "__main__":

    hb = force.HarmonicBondForce()
    intg = LangevinIntegrator(
        masses=np.array([1.0, 12.0]),
        friction=10.0,
        dt=0.03,
        num_params=2)
    x0 = np.array([[1.0, 0.5, -0.5], [0.2, 0.1, -0.3]])

    x_ph = tf.placeholder(dtype=tf.float32, shape=(None, 3))

    dx, dxdps, zt, tmp = intg.step(x_ph, [hb])

    sess = tf.Session()
    sess.run(tf.initializers.global_variables())

    x = x0

    for step in range(10):
        dx_val, dp_val, zt_val, tmp_val = sess.run([dx, dxdps, zt, tmp], feed_dict={x_ph: x})
        x = x + dx_val
