import numpy as np
import tensorflow as tf
import unittest

from tensorflow.python.ops.parallel_for.gradients import jacobian
from timemachine.functionals import bonded
from timemachine.cpu_functionals import custom_ops
from tests.test_integrator import ReferenceLangevinIntegrator

class ReferenceIntegrator():

    def __init__(self,
        x_t,
        v_t,
        noise,
        grads,
        hessians,
        mixed_partials,
        dt,
        coeff_a,
        coeff_bs,
        coeff_cs,
        buffer_size,
        precision=tf.float64):

        self.x_t = x_t
        self.v_t = v_t
        self.dt = dt

        self.num_params = mixed_partials.shape[0]
        self.num_atoms = mixed_partials.shape[1]
        self.dE_dx = grads
        self.d2E_dx2 = hessians
        self.d2E_dxdp = mixed_partials
        self.ca = coeff_a
        self.cbs = coeff_bs
        self.ccs = coeff_cs

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
        new_dxdp_t *= -self.cbs * self.dt

        self.dxdp_t_assign = tf.assign(self.dxdp_t, new_dxdp_t)

        self.new_v_t = self.ca*self.v_t - self.cbs*self.dE_dx + self.ccs*noise
        self.new_x_t = self.x_t + self.new_v_t * self.dt


class TestGPUIntegrator(unittest.TestCase):

    def tearDown(self):
        # (ytz): needed to clear variables
        tf.reset_default_graph()

    def test_pure_gpu(self):
        masses = np.array([1.0, 12.0, 4.0, 2.0], dtype=np.float64)

        num_atoms = masses.shape[0]

        x_ph = tf.placeholder(shape=(num_atoms, 3), dtype=np.float64)
        v_ph = tf.placeholder(shape=(num_atoms, 3), dtype=np.float64)
        noise_ph = tf.placeholder(shape=(num_atoms, 3), dtype=np.float64)
        grad_ph = tf.placeholder(shape=(num_atoms, 3), dtype=np.float64)
        hessian_ph = tf.placeholder(shape=(num_atoms, 3, num_atoms, 3), dtype=np.float64)
        mixed_partial_ph = tf.placeholder(shape=(5, num_atoms, 3), dtype=np.float64)

        num_params = mixed_partial_ph.shape[0]

        coeff_a = np.float64(0.5)
        coeff_bs = np.expand_dims(np.array(masses), axis=1)
        coeff_cs = np.expand_dims(1/np.array(masses), axis=1)
        buffer_size = 3
        dt = 0.03

        ref_intg = ReferenceIntegrator(
            x_ph,
            v_ph,
            noise_ph,
            grad_ph,
            hessian_ph,
            mixed_partial_ph,
            dt,
            coeff_a,
            coeff_bs,
            coeff_cs,
            buffer_size,
            precision=tf.float64)

        gpu_intg = custom_ops.Integrator_double(
            dt,
            buffer_size,
            num_atoms,
            num_params,
            coeff_a,
            coeff_bs.reshape(-1).tolist(),
            coeff_cs.reshape(-1).tolist()
        )

        bond_params_np = np.array([100.0, 2.0], dtype=np.float64)
        global_bond_param_idxs = np.arange(bond_params_np.shape[0], dtype=np.int32)
        bond_idxs = np.array([
            [0, 1],
            [1, 2]
        ], dtype=np.int32)
        bond_param_idxs = np.array([
            [0, 1],
            [0, 1],
        ])

        hb_gpu = custom_ops.HarmonicBondGPU_double(
            bond_params_np.reshape(-1).tolist(),
            global_bond_param_idxs.tolist(),
            bond_param_idxs.reshape(-1).tolist(),
            bond_idxs.reshape(-1).tolist(),
        )

        custom_ops.Context_double(
            [hb_gpu],
            gpu_intg
        )

    # def test_analytic_integration(self):
    #     """
    #     Testing against reference implementation.
    #     """
    #     masses = np.array([1.0, 12.0, 4.0])
    #     x0 = np.array([
    #         [1.0, 0.5, -0.5],
    #         [0.2, 0.1, -0.3],
    #         [0.5, 0.4, 0.3],
    #     ], dtype=np.float64)
    #     x0.setflags(write=False)

    #     bond_params_np = np.array([100.0, 2.0], dtype=np.float64)
    #     bond_params_tf = tf.convert_to_tensor(bond_params_np)

    #     bond_idxs = np.array([
    #         [0, 1],
    #         [1, 2]
    #     ], dtype=np.int32)

    #     bond_param_idxs = np.array([
    #         [0, 1],
    #         [0, 1],
    #     ])

    #     hb = bonded.HarmonicBond(
    #         bond_params_tf,
    #         bond_idxs,
    #         bond_param_idxs,
    #     )

    #     angle_params_np = np.array([75, 1.81], dtype=np.float64)
    #     angle_params_tf = tf.convert_to_tensor(angle_params_np)

    #     angle_idxs = np.array([[0,1,2]], dtype=np.int32)
    #     angle_param_idxs = np.array([[0,1,2]], dtype=np.int32)

    #     ha = bonded.HarmonicAngle(
    #         angle_params_tf,
    #         angle_idxs,
    #         angle_param_idxs,
    #         cos_angles=True
    #     )

    #     friction = 10.0
    #     dt = 0.01
    #     temp = 0.0
    #     num_atoms = len(masses)
    #     x_ph = tf.placeholder(dtype=tf.float64, shape=(num_atoms, 3))

    #     ref_intg = ReferenceLangevinIntegrator(masses, dt, friction, temp)

    #     num_steps = 5

    #     x = x_ph

    #     for step in range(num_steps):
    #         print("step", step)
    #         all_grads = []
    #         for nrg in [hb, ha]:
    #             all_grads.append(tf.gradients(nrg.energy(x), x)[0])
    #         all_grads = tf.stack(all_grads, axis=0)
    #         grads = tf.reduce_sum(all_grads, axis=0)
    #         dx = ref_intg.step(grads)
    #         x += dx

    #     ref_x_final_op = x

    #     # verify correctness of jacobians through time
    #     ref_dxdp_hb_op = jacobian(ref_x_final_op, hb.get_params(), use_pfor=False) # (N, 3, P)
    #     ref_dxdp_ha_op = jacobian(ref_x_final_op, ha.get_params(), use_pfor=False) # (N, 3, P)
    #     ref_dxdp_hb_op = tf.transpose(ref_dxdp_hb_op, perm=[2,0,1])
    #     ref_dxdp_ha_op = tf.transpose(ref_dxdp_ha_op, perm=[2,0,1])

    #     buffer_size = 100 # just make something large

    #     global_bond_param_idxs = np.arange(bond_params_np.shape[0], dtype=np.int32)
    #     global_angle_param_idxs = np.arange(angle_params_np.shape[0], dtype=np.int32) + global_bond_param_idxs.shape[0]
    #     total_params = bond_params_np.shape[0] + angle_params_np.shape[0]

    #     gpu_intg = custom_ops.Integrator_double(
    #         dt,
    #         buffer_size,
    #         num_atoms,
    #         total_params,
    #         ref_intg.coeff_a,
    #         ref_intg.coeff_bs.reshape(-1).tolist(),
    #         ref_intg.coeff_cs.reshape(-1).tolist()
    #     )

    #     hb_cpu = custom_ops.HarmonicBond_double(
    #         bond_params_np.reshape(-1).tolist(),
    #         global_bond_param_idxs.tolist(),
    #         bond_param_idxs.reshape(-1).tolist(),
    #         bond_idxs.reshape(-1).tolist(),
    #     )

    #     ha_cpu = custom_ops.HarmonicAngle_double(
    #         angle_params_np.reshape(-1).tolist(),
    #         global_angle_param_idxs.tolist(),
    #         angle_param_idxs.reshape(-1).tolist(),
    #         angle_idxs.reshape(-1).tolist(),
    #         True
    #     )

    #     gpu_intg.set_coordinates(x0.reshape(-1).tolist())
    #     gpu_intg.set_velocities(np.zeros_like(x0).reshape(-1).tolist())

    #     xt = x0

    #     for step in range(num_steps):
    #         cpu_e0, cpu_grad0, cpu_hess0, cpu_mixed0 = hb_cpu.total_derivative(xt, total_params)
    #         cpu_e1, cpu_grad1, cpu_hess1, cpu_mixed1 = ha_cpu.total_derivative(xt, total_params)

    #         cpu_e = np.array(cpu_e0) + np.array(cpu_e1)
    #         cpu_grad = np.array(cpu_grad0) + np.array(cpu_grad1)
    #         cpu_hess = np.array(cpu_hess0) + np.array(cpu_hess1)
    #         cpu_mixed = np.array(cpu_mixed0) + np.array(cpu_mixed1)

    #         gpu_intg.step(cpu_grad, cpu_hess, cpu_mixed)
    #         xt = np.reshape(gpu_intg.get_coordinates(), (num_atoms, 3))

    #     sess = tf.Session()
    #     sess.run(tf.initializers.global_variables())

    #     cpu_final_x_t_val = gpu_intg.get_coordinates()
    #     np.testing.assert_array_almost_equal(
    #         cpu_final_x_t_val,
    #         sess.run(ref_x_final_op, feed_dict={x_ph: x0}).reshape(-1),
    #         decimal=13)

    #     cpu_dxdp = np.array(gpu_intg.get_dxdp()).reshape(total_params, num_atoms, 3)
    #     ref_dxdp_bonds, ref_dxdp_angles = sess.run([ref_dxdp_hb_op, ref_dxdp_ha_op], feed_dict={x_ph: x0})

    #     np.testing.assert_array_almost_equal(cpu_dxdp[:2], ref_dxdp_bonds)
    #     np.testing.assert_array_almost_equal(cpu_dxdp[2:], ref_dxdp_angles)


    # def test_gpu_integrator(self):
    #     """
    #     Testing convergence of zetas.
    #     """

    #     masses = np.array([1.0, 12.0, 4.0, 2.0], dtype=np.float64)

    #     num_atoms = masses.shape[0]

    #     x_ph = tf.placeholder(shape=(num_atoms, 3), dtype=np.float64)
    #     v_ph = tf.placeholder(shape=(num_atoms, 3), dtype=np.float64)
    #     noise_ph = tf.placeholder(shape=(num_atoms, 3), dtype=np.float64)
    #     grad_ph = tf.placeholder(shape=(num_atoms, 3), dtype=np.float64)
    #     hessian_ph = tf.placeholder(shape=(num_atoms, 3, num_atoms, 3), dtype=np.float64)
    #     mixed_partial_ph = tf.placeholder(shape=(5, num_atoms, 3), dtype=np.float64)

    #     num_params = mixed_partial_ph.shape[0]

    #     coeff_a = np.float64(0.5)
    #     coeff_bs = np.expand_dims(np.array(masses), axis=1)
    #     coeff_cs = np.expand_dims(1/np.array(masses), axis=1)
    #     buffer_size = 3
    #     dt = 0.03

    #     ref_intg = ReferenceIntegrator(
    #         x_ph,
    #         v_ph,
    #         noise_ph,
    #         grad_ph,
    #         hessian_ph,
    #         mixed_partial_ph,
    #         dt,
    #         coeff_a,
    #         coeff_bs,
    #         coeff_cs,
    #         buffer_size,
    #         precision=tf.float64)

    #     gpu_intg = custom_ops.Integrator_double(
    #         dt,
    #         buffer_size,
    #         num_atoms,
    #         num_params,
    #         coeff_a,
    #         coeff_bs.reshape(-1).tolist(),
    #         coeff_cs.reshape(-1).tolist()
    #     )

    #     sess = tf.Session()
    #     sess.run(tf.initializers.global_variables())

    #     num_steps = 10

    #     x_t = np.random.rand(num_atoms,3).astype(dtype=np.float64)-0.5
    #     v_t = np.random.rand(num_atoms,3).astype(dtype=np.float64)-0.5

    #     gpu_intg.set_coordinates(x_t.reshape(-1).tolist());
    #     gpu_intg.set_velocities(v_t.reshape(-1).tolist());

    #     for step in range(num_steps):
    #         # -0.5 is to avoid exccess accumulation
    #         grad = np.random.rand(num_atoms,3).astype(dtype=np.float64)-0.5
    #         # this absolutely needs to be symmetric for things to work to avoid a transposition
    #         hessian = np.random.rand(num_atoms*3,num_atoms*3).astype(dtype=np.float64)-0.5
    #         hessian = (hessian + hessian.T)/2
    #         hessian = hessian.reshape((num_atoms, 3, num_atoms, 3))
    #         mixed_partial = np.random.rand(num_params,num_atoms,3).astype(dtype=np.float64)-0.5

    #         gpu_intg.step(grad, hessian, mixed_partial)
    #         test_dxdp_val = gpu_intg.get_dxdp()
    #         test_x_t_val = gpu_intg.get_coordinates()
    #         test_v_t_val = gpu_intg.get_velocities()
    #         test_noise = gpu_intg.get_noise()

    #         new_x, new_v, ref_dxdp_val = sess.run([ref_intg.new_x_t, ref_intg.new_v_t, ref_intg.dxdp_t_assign], feed_dict={
    #             x_ph: x_t,
    #             v_ph: v_t,
    #             noise_ph: np.array(test_noise).reshape(num_atoms, 3),
    #             grad_ph: grad,
    #             hessian_ph: hessian,
    #             mixed_partial_ph: mixed_partial
    #         })

    #         np.testing.assert_allclose(test_v_t_val, new_v.reshape(-1))
    #         np.testing.assert_allclose(test_x_t_val, new_x.reshape(-1))

    #         x_t = new_x
    #         v_t = new_v

    #         np.testing.assert_allclose(
    #             test_dxdp_val,
    #             ref_dxdp_val.reshape(-1)
    #         )

    #         # assert 0


if __name__ == "__main__":

    unittest.main()
