import numpy as np
import tensorflow as tf
import unittest

from timemachine.cpu_functionals import custom_ops


class ReferenceIntegrator():

    def __init__(self,
        grads,
        hessians,
        mixed_partials,
        coeff_a,
        coeff_bs,
        buffer_size,
        precision=tf.float64):

        self.num_params = mixed_partials.shape[0]
        self.num_atoms = mixed_partials.shape[1]
        self.dE_dx = grads
        self.d2E_dx2 = hessians
        self.d2E_dxdp = mixed_partials
        self.ca = coeff_a
        self.cbs = coeff_bs

        self.scale = (1-tf.pow(self.ca, tf.range(buffer_size, dtype=precision)+1))/(1-self.ca)
        self.scale = tf.reverse(self.scale, [0])
        self.scale = tf.reshape(self.scale, (-1, 1, 1, 1))

        self.initializers = []
    
        # buffer for current step's dxdp
        self.dxdp_t = tf.get_variable(
            "buffer_dxdp",
            shape=(self.num_params, self.num_atoms, 3),
            dtype=precision,
            initializer=tf.initializers.zeros)
        self.initializers.append(self.dxdp_t.initializer)

        # buffer for unconverged Dps
        self.buffer_Dx = tf.get_variable(
            "buffer_Dx",
            shape=(buffer_size, self.num_params, self.num_atoms, 3),
            dtype=precision,
            initializer=tf.initializers.zeros)
        self.initializers.append(self.buffer_Dx.initializer)

        # buffer for converged Dps
        self.converged_Dps = tf.get_variable(
            "converged_Dps",
            shape=(self.num_params, self.num_atoms, 3),
            dtype=precision,
            initializer=tf.initializers.zeros)

        self.initializers.append(self.converged_Dps.initializer)

        num_dims = 3
        num_atoms = self.num_atoms

        Dx_t = tf.einsum('ijkl,mkl->mij', self.d2E_dx2, self.dxdp_t)
        Dx_t += self.d2E_dxdp
        self.Dx_t = Dx_t

        all_ops = []

        converged_Dx_assign = tf.assign_add(self.converged_Dps, self.buffer_Dx[0])
        override = self.buffer_Dx[0].assign(Dx_t)
        buffer_Dx_assign = tf.assign(
            self.buffer_Dx,
            tf.roll(override, shift=-1, axis=0)
        )

        # compute dxdp_{t+1} using unconverged Dp and converged Dps
        new_dxdp_t = buffer_Dx_assign * self.scale
        new_dxdp_t = tf.reduce_sum(new_dxdp_t, axis=0)
        new_dxdp_t += converged_Dx_assign * self.scale[0][0][0][0]
        new_dxdp_t *= -self.cbs

        self.dxdp_t_assign = tf.assign(self.dxdp_t, new_dxdp_t)

class TestReduction(unittest.TestCase):

    def tearDown(self):
        # (ytz): needed to clear variables
        tf.reset_default_graph()

    def test_gpu_integrator(self):
        """
        Testing convergence of zetas.
        """

        masses = np.array([1.0, 12.0, 4.0, 2.0], dtype=np.float64)

        num_atoms = masses.shape[0]

        x_ph = tf.placeholder(shape=(num_atoms, 3), dtype=np.float64)
        grad_ph = tf.placeholder(shape=(num_atoms, 3), dtype=np.float64)
        hessian_ph = tf.placeholder(shape=(num_atoms, 3, num_atoms, 3), dtype=np.float64)
        mixed_partial_ph = tf.placeholder(shape=(5, num_atoms, 3), dtype=np.float64)

        num_params = mixed_partial_ph.shape[0]

        coeff_a = np.float64(0.5)
        coeff_bs = np.expand_dims(np.array(masses), axis=1)
        buffer_size = 3

        ref_intg = ReferenceIntegrator(
            grad_ph,
            hessian_ph,
            mixed_partial_ph,
            coeff_a,
            coeff_bs,
            buffer_size,
            precision=tf.float64)

        gpu_intg = custom_ops.Integrator_double(
            coeff_a,
            buffer_size,
            num_atoms,
            num_params,
            coeff_bs.reshape(-1).tolist()
        )

        sess = tf.Session()
        sess.run(tf.initializers.global_variables())

        num_steps = 10

        for step in range(num_steps):
            # -0.5 is to avoid exccess accumulation
            grad = np.random.rand(num_atoms,3).astype(dtype=np.float64)-0.5
            # this absolutely needs to be symmetric for things to work to avoid a transposition
            hessian = np.random.rand(num_atoms*3,num_atoms*3).astype(dtype=np.float64)-0.5
            hessian = (hessian + hessian.T)/2
            hessian = hessian.reshape((num_atoms, 3, num_atoms, 3))
            mixed_partial = np.random.rand(num_params,num_atoms,3).astype(dtype=np.float64)-0.5

            ref_dxdp_val = sess.run(ref_intg.dxdp_t_assign, feed_dict={
                grad_ph: grad,
                hessian_ph: hessian,
                mixed_partial_ph: mixed_partial})

            gpu_intg.step(hessian, mixed_partial)
            test_dxdp_val = gpu_intg.get_dxdp()

            np.testing.assert_allclose(
                test_dxdp_val,
                ref_dxdp_val.reshape(-1)
            )



if __name__ == "__main__":

    unittest.main()
