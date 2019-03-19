import numpy as np
import tensorflow as tf
import unittest

from tensorflow.python.ops.parallel_for.gradients import jacobian
from timemachine.functionals import bonded, nonbonded
from timemachine.cpu_functionals import custom_ops
from tests.test_integrator import ReferenceLangevinIntegrator

class TestGPUIntegrator(unittest.TestCase):

    def tearDown(self):
        # (ytz): needed to clear variables
        tf.reset_default_graph()



    def test_gpu_electrostatic_analytic_integration(self):
        """
        Testing that lower triangular hessians are working as intended. This is because for the
        non-bonded kernels we compute only the lower right triangular portion
        """
        x0 = np.array([
            [ 0.0637,   0.0126,   0.2203],
            [ 1.0573,  -0.2011,   1.2864],
            [ 2.3928,   1.2209,  -0.2230],
            [-0.6891,   1.6983,   0.0780],
            [-0.6312,  -1.6261,  -0.2601]
        ], dtype=np.float64)

        N = x0.shape[0]

        x_ph = tf.placeholder(shape=(N, 3), dtype=np.float64)

        params_np = np.array([1.3, 0.3], dtype=np.float64)
        params_tf = tf.convert_to_tensor(params_np)
        param_idxs = np.array([0, 1, 1, 1, 1], dtype=np.int32)
        scale_matrix = np.array([
            [  0,  1,  1,  1,0.5],
            [  1,  0,  0,  1,  1],
            [  1,  0,  0,  0,0.2],
            [  1,  1,  0,  0,  1],
            [0.5,  1,0.2,  1,  0],
        ], dtype=np.float64)

        cutoff = None
        crf = 0.0

        sess = tf.Session()
        sess.run(tf.initializers.global_variables())

        ref_nrg = nonbonded.Electrostatic(params_tf, param_idxs, scale_matrix, cutoff=cutoff, crf=crf)
        nrg_op = ref_nrg.energy(x_ph)

        es_gpu = custom_ops.ElectrostaticsGPU_double(
            params_np.reshape(-1).tolist(),
            list(range(params_np.shape[0])),
            param_idxs.reshape(-1).tolist(),
            scale_matrix.reshape(-1).tolist()
        )

        masses = np.array([6.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)

        friction = 100.0
        dt = 0.01
        temp = 0.0
        num_atoms = len(masses)
        x_ph = tf.placeholder(dtype=tf.float64, shape=(num_atoms, 3))

        ref_intg = ReferenceLangevinIntegrator(masses, dt, friction, temp)

        num_steps = 10

        x = x_ph

        for step in range(num_steps):
            print("step", step)
            all_grads = []
            for nrg in [ref_nrg]:
                all_grads.append(tf.gradients(nrg.energy(x), x)[0])
            all_grads = tf.stack(all_grads, axis=0)
            grads = tf.reduce_sum(all_grads, axis=0)
            dx = ref_intg.step(grads)
            x += dx

        ref_x_final_op = x

        # verify correctness of jacobians through time
        ref_dxdp_es_op = jacobian(ref_x_final_op, ref_nrg.get_params(), use_pfor=False) # (N, 3, P)
        # ref_dxdp_ha_op = jacobian(ref_x_final_op, ha.get_params(), use_pfor=False) # (N, 3, P)
        ref_dxdp_es_op = tf.transpose(ref_dxdp_es_op, perm=[2,0,1])
        # ref_dxdp_ha_op = tf.transpose(ref_dxdp_ha_op, perm=[2,0,1])

        buffer_size = 100 # just make something large

        total_params = params_np.shape[0]
        # global_angle_param_idxs = np.arange(angle_params_np.shape[0], dtype=np.int32) + global_bond_param_idxs.shape[0]
        # total_params = bond_params_np.shape[0] + angle_params_np.shape[0]

        gpu_intg = custom_ops.Integrator_double(
            dt,
            buffer_size,
            num_atoms,
            total_params,
            ref_intg.coeff_a,
            ref_intg.coeff_bs.reshape(-1).tolist(),
            ref_intg.coeff_cs.reshape(-1).tolist()
        )

        gpu_intg.set_coordinates(x0.reshape(-1).tolist())
        gpu_intg.set_velocities(np.zeros_like(x0).reshape(-1).tolist())

        xt = x0

        context = custom_ops.Context_double(
            [es_gpu],
            gpu_intg
        )

        for step in range(num_steps):
            context.step()

        sess = tf.Session()
        sess.run(tf.initializers.global_variables())
 
        cpu_final_x_t_val = gpu_intg.get_coordinates()
        # test consistency of geometry
        np.testing.assert_array_almost_equal(
            cpu_final_x_t_val,
            sess.run(ref_x_final_op, feed_dict={x_ph: x0}).reshape(-1),
            decimal=13)

        gpu_dxdp = np.array(gpu_intg.get_dxdp()).reshape(total_params, num_atoms, 3)
        ref_dxdp_es = sess.run(ref_dxdp_es_op, feed_dict={x_ph: x0})

        np.testing.assert_array_almost_equal(gpu_dxdp, ref_dxdp_es)
        # np.testing.assert_array_almost_equal(gpu_dxdp[2:], ref_dxdp_angles)


    def test_gpu_analytic_integration(self):

        masses = np.array([1.0, 12.0, 4.0])
        x0 = np.array([
            [1.0, 0.5, -0.5],
            [0.2, 0.1, -0.3],
            [0.5, 0.4, 0.3],
        ], dtype=np.float64)
        x0.setflags(write=False)

        bond_params_np = np.array([100.0, 2.0], dtype=np.float64)
        bond_params_tf = tf.convert_to_tensor(bond_params_np)

        bond_idxs = np.array([
            [0, 1],
            [1, 2]
        ], dtype=np.int32)

        bond_param_idxs = np.array([
            [0, 1],
            [0, 1],
        ])

        hb = bonded.HarmonicBond(
            bond_params_tf,
            bond_idxs,
            bond_param_idxs,
        )

        angle_params_np = np.array([75, 1.81], dtype=np.float64)
        angle_params_tf = tf.convert_to_tensor(angle_params_np)

        angle_idxs = np.array([[0,1,2]], dtype=np.int32)
        angle_param_idxs = np.array([[0,1,2]], dtype=np.int32)

        ha = bonded.HarmonicAngle(
            angle_params_tf,
            angle_idxs,
            angle_param_idxs,
            cos_angles=True
        )

        friction = 10.0
        dt = 0.01
        temp = 0.0
        num_atoms = len(masses)
        x_ph = tf.placeholder(dtype=tf.float64, shape=(num_atoms, 3))

        ref_intg = ReferenceLangevinIntegrator(masses, dt, friction, temp)

        num_steps = 5

        x = x_ph

        for step in range(num_steps):
            print("step", step)
            all_grads = []
            for nrg in [hb, ha]:
                all_grads.append(tf.gradients(nrg.energy(x), x)[0])
            all_grads = tf.stack(all_grads, axis=0)
            grads = tf.reduce_sum(all_grads, axis=0)
            dx = ref_intg.step(grads)
            x += dx

        ref_x_final_op = x

        # verify correctness of jacobians through time
        ref_dxdp_hb_op = jacobian(ref_x_final_op, hb.get_params(), use_pfor=False) # (N, 3, P)
        ref_dxdp_ha_op = jacobian(ref_x_final_op, ha.get_params(), use_pfor=False) # (N, 3, P)
        ref_dxdp_hb_op = tf.transpose(ref_dxdp_hb_op, perm=[2,0,1])
        ref_dxdp_ha_op = tf.transpose(ref_dxdp_ha_op, perm=[2,0,1])

        buffer_size = 100 # just make something large

        global_bond_param_idxs = np.arange(bond_params_np.shape[0], dtype=np.int32)
        global_angle_param_idxs = np.arange(angle_params_np.shape[0], dtype=np.int32) + global_bond_param_idxs.shape[0]
        total_params = bond_params_np.shape[0] + angle_params_np.shape[0]

        gpu_intg = custom_ops.Integrator_double(
            dt,
            buffer_size,
            num_atoms,
            total_params,
            ref_intg.coeff_a,
            ref_intg.coeff_bs.reshape(-1).tolist(),
            ref_intg.coeff_cs.reshape(-1).tolist()
        )

        hb_gpu = custom_ops.HarmonicBondGPU_double(
            bond_params_np.reshape(-1).tolist(),
            global_bond_param_idxs.tolist(),
            bond_param_idxs.reshape(-1).tolist(),
            bond_idxs.reshape(-1).tolist(),
        )

        ha_gpu = custom_ops.HarmonicAngleGPU_double(
            angle_params_np.reshape(-1).tolist(),
            global_angle_param_idxs.tolist(),
            angle_param_idxs.reshape(-1).tolist(),
            angle_idxs.reshape(-1).tolist(),
        )

        gpu_intg.set_coordinates(x0.reshape(-1).tolist())
        gpu_intg.set_velocities(np.zeros_like(x0).reshape(-1).tolist())

        xt = x0

        context = custom_ops.Context_double(
            [hb_gpu, ha_gpu],
            gpu_intg
        )

        # test inference first
        for step in range(num_steps):
            context.step(True)

        sess = tf.Session()
        sess.run(tf.initializers.global_variables())
 
        cpu_final_x_t_val = gpu_intg.get_coordinates()
        np.testing.assert_array_almost_equal(
            cpu_final_x_t_val,
            sess.run(ref_x_final_op, feed_dict={x_ph: x0}).reshape(-1),
            decimal=13)

        # reset state
        gpu_intg.reset()
        gpu_intg.set_coordinates(x0.reshape(-1).tolist())
        gpu_intg.set_velocities(np.zeros_like(x0).reshape(-1).tolist())

        for step in range(num_steps):
            context.step()

        cpu_final_x_t_val = gpu_intg.get_coordinates()
        np.testing.assert_array_almost_equal(
            cpu_final_x_t_val,
            sess.run(ref_x_final_op, feed_dict={x_ph: x0}).reshape(-1),
            decimal=13)

        gpu_dxdp = np.array(gpu_intg.get_dxdp()).reshape(total_params, num_atoms, 3)
        ref_dxdp_bonds, ref_dxdp_angles = sess.run([ref_dxdp_hb_op, ref_dxdp_ha_op], feed_dict={x_ph: x0})

        np.testing.assert_array_almost_equal(gpu_dxdp[:2], ref_dxdp_bonds)
        np.testing.assert_array_almost_equal(gpu_dxdp[2:], ref_dxdp_angles)



    def test_cpu_analytic_integration(self):
        """
        Testing against reference implementation.
        """
        masses = np.array([1.0, 12.0, 4.0])
        x0 = np.array([
            [1.0, 0.5, -0.5],
            [0.2, 0.1, -0.3],
            [0.5, 0.4, 0.3],
        ], dtype=np.float64)
        x0.setflags(write=False)

        bond_params_np = np.array([100.0, 2.0], dtype=np.float64)
        bond_params_tf = tf.convert_to_tensor(bond_params_np)

        bond_idxs = np.array([
            [0, 1],
            [1, 2]
        ], dtype=np.int32)

        bond_param_idxs = np.array([
            [0, 1],
            [0, 1],
        ])

        hb = bonded.HarmonicBond(
            bond_params_tf,
            bond_idxs,
            bond_param_idxs,
        )

        angle_params_np = np.array([75, 1.81], dtype=np.float64)
        angle_params_tf = tf.convert_to_tensor(angle_params_np)

        angle_idxs = np.array([[0,1,2]], dtype=np.int32)
        angle_param_idxs = np.array([[0,1,2]], dtype=np.int32)

        ha = bonded.HarmonicAngle(
            angle_params_tf,
            angle_idxs,
            angle_param_idxs,
            cos_angles=True
        )

        friction = 10.0
        dt = 0.01
        temp = 0.0
        num_atoms = len(masses)
        x_ph = tf.placeholder(dtype=tf.float64, shape=(num_atoms, 3))

        ref_intg = ReferenceLangevinIntegrator(masses, dt, friction, temp)

        num_steps = 5

        x = x_ph

        for step in range(num_steps):
            print("step", step)
            all_grads = []
            for nrg in [hb, ha]:
                all_grads.append(tf.gradients(nrg.energy(x), x)[0])
            all_grads = tf.stack(all_grads, axis=0)
            grads = tf.reduce_sum(all_grads, axis=0)
            dx = ref_intg.step(grads)
            x += dx

        ref_x_final_op = x

        # verify correctness of jacobians through time
        ref_dxdp_hb_op = jacobian(ref_x_final_op, hb.get_params(), use_pfor=False) # (N, 3, P)
        ref_dxdp_ha_op = jacobian(ref_x_final_op, ha.get_params(), use_pfor=False) # (N, 3, P)
        ref_dxdp_hb_op = tf.transpose(ref_dxdp_hb_op, perm=[2,0,1])
        ref_dxdp_ha_op = tf.transpose(ref_dxdp_ha_op, perm=[2,0,1])

        buffer_size = 100 # just make something large

        global_bond_param_idxs = np.arange(bond_params_np.shape[0], dtype=np.int32)
        global_angle_param_idxs = np.arange(angle_params_np.shape[0], dtype=np.int32) + global_bond_param_idxs.shape[0]
        total_params = bond_params_np.shape[0] + angle_params_np.shape[0]

        gpu_intg = custom_ops.Integrator_double(
            dt,
            buffer_size,
            num_atoms,
            total_params,
            ref_intg.coeff_a,
            ref_intg.coeff_bs.reshape(-1).tolist(),
            ref_intg.coeff_cs.reshape(-1).tolist()
        )

        hb_cpu = custom_ops.HarmonicBond_double(
            bond_params_np.reshape(-1).tolist(),
            global_bond_param_idxs.tolist(),
            bond_param_idxs.reshape(-1).tolist(),
            bond_idxs.reshape(-1).tolist(),
        )

        ha_cpu = custom_ops.HarmonicAngle_double(
            angle_params_np.reshape(-1).tolist(),
            global_angle_param_idxs.tolist(),
            angle_param_idxs.reshape(-1).tolist(),
            angle_idxs.reshape(-1).tolist(),
            True
        )

        gpu_intg.set_coordinates(x0.reshape(-1).tolist())
        gpu_intg.set_velocities(np.zeros_like(x0).reshape(-1).tolist())

        xt = x0

        for step in range(num_steps):
            cpu_e0, cpu_grad0, cpu_hess0, cpu_mixed0 = hb_cpu.total_derivative(xt, total_params)
            cpu_e1, cpu_grad1, cpu_hess1, cpu_mixed1 = ha_cpu.total_derivative(xt, total_params)

            cpu_e = np.array(cpu_e0) + np.array(cpu_e1)
            cpu_grad = np.array(cpu_grad0) + np.array(cpu_grad1)
            cpu_hess = np.array(cpu_hess0) + np.array(cpu_hess1)
            cpu_mixed = np.array(cpu_mixed0) + np.array(cpu_mixed1)

            gpu_intg.step(cpu_grad, cpu_hess, cpu_mixed)
            xt = np.reshape(gpu_intg.get_coordinates(), (num_atoms, 3))

        sess = tf.Session()
        sess.run(tf.initializers.global_variables())

        cpu_final_x_t_val = gpu_intg.get_coordinates()
        np.testing.assert_array_almost_equal(
            cpu_final_x_t_val,
            sess.run(ref_x_final_op, feed_dict={x_ph: x0}).reshape(-1),
            decimal=13)

        cpu_dxdp = np.array(gpu_intg.get_dxdp()).reshape(total_params, num_atoms, 3)
        ref_dxdp_bonds, ref_dxdp_angles = sess.run([ref_dxdp_hb_op, ref_dxdp_ha_op], feed_dict={x_ph: x0})

        np.testing.assert_array_almost_equal(cpu_dxdp[:2], ref_dxdp_bonds)
        np.testing.assert_array_almost_equal(cpu_dxdp[2:], ref_dxdp_angles)


if __name__ == "__main__":

    unittest.main()
