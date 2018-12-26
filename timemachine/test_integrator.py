import numpy as np
import tensorflow as tf
import unittest
import force
from constants import BOLTZ
import integrator

class ReferenceLangevinIntegrator():

    def __init__(self, masses, dt=0.0025, friction=1.0, temp=300.0, disable_noise=False):
        self.dt = dt
        self.v_t = None
        self.friction = friction # dissipation speed (how fast we forget)
        self.temperature = temp           # temperature

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

    def step(self, grads):
        num_atoms = grads.shape[0]
        num_dims = grads.shape[1]

        if self.v_t is None:
            self.v_t = np.zeros((num_atoms, num_dims))

        noise = self.normal.sample((num_atoms, num_dims))
        noise = tf.cast(noise, dtype=grads.dtype)

        if self.disable_noise:
            print("disabling noise")
            noise = tf.zeros(noise.shape, dtype=grads.dtype)

        # grads shape: [N x 3]


        self.v_t = self.vscale*self.v_t - self.fscale*self.invMasses*grads + self.nscale*self.sqrtInvMasses*noise
        dx = self.v_t * self.dt
        return dx


class TestLangevinIntegrator(unittest.TestCase):

    # def test_gradient_descent():

    # def test_slow_convergence():

    def test_five_steps(self):
        masses = np.array([1.0, 12.0])
        friction = 10.0
        dt = 0.003
        num_params = 2
        temp = 300.0

        x_ph = tf.placeholder(dtype=tf.float32, shape=(2, 3))
        
        x0 = np.array([[1.0, 0.5, -0.5], [0.2, 0.1, -0.3]], dtype=np.float32)

        hb = force.HarmonicBondForce()
        ref_intg = ReferenceLangevinIntegrator(masses, dt, friction, temp, disable_noise=True)

        num_steps = 2

        x = x_ph

        all_tmps = []

        for step in range(num_steps):
            grads = hb.gradients(x)
            all_tmps.append(grads)
            dx = ref_intg.step(grads)
            x += dx

        ref_dxdp = tf.gradients(x, hb.params())
        test_intg = integrator.LangevinIntegrator(masses, 2, dt, friction, temp, disable_noise=True)

        dx, dxdps, gs, tmp = test_intg.step(x_ph, [hb])
        dxdps = tf.reduce_sum(dxdps, axis=[1,2])

        sess = tf.Session()
        sess.run(tf.initializers.global_variables())

        obs_x, obs_dxdp, g0, g1 = sess.run([x, ref_dxdp, all_tmps[0], all_tmps[1]], feed_dict={x_ph: x0})
        

        print("g0, g1", g0, g1, "obs_x", obs_x)

        x = x0

        for step in range(num_steps):
            dx_val, dp_val, gs_val, tmp_val = sess.run([dx, dxdps, gs, tmp], feed_dict={x_ph: x})
            print("test_g", gs_val)
            x += dx_val

        test_dxdp = dp_val

        print(obs_x, x)
        print(obs_dxdp, test_dxdp) # relative ratios are correct


if __name__ == "__main__":

    unittest.main()
