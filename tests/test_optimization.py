import numpy as np
import tensorflow as tf
import unittest

from timemachine import bonded_force
from timemachine.constants import BOLTZ, VIBRATIONAL_CONSTANT
from timemachine import integrator
from timemachine import observable
from timemachine.reservoir_sampler import ReservoirSampler

class TestOptimization(unittest.TestCase):

    def tearDown(self):
        tf.reset_default_graph()

    def test_optimize_ensemble(self):
        """
        Testing that we can optimize ensembles with different integrator and ff settings relative to a canonical one.
        """

        # 1. Generate a water ensemble by doing 5000 steps of MD starting from an idealized geometry.
        # 2. Reservoir sample along the trajectory.
        # 3. Generate a loss function used sum of square distances.
        # 4. Compute parameter derivatives.

        x_opt = np.array([
            [-0.0070, -0.0100, 0.0000],
            [-0.1604,  0.4921, 0.0000],
            [ 0.5175,  0.0128, 0.0000],
        ], dtype=np.float64) # idealized geometry

        x_opt.setflags(write=False)

        
        masses = np.array([8.0, 1.0, 1.0], dtype=np.float64)
        num_atoms = len(masses)

        ideal_bond = 0.52
        ideal_angle = 1.81

        bond_params = [
            tf.get_variable("OH_kb", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(100.0)),
            tf.get_variable("OH_b0", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(ideal_bond)),
        ]

        hb = bonded_force.HarmonicBondForce(
            params=bond_params,
            bond_idxs=np.array([[0,1],[0,2]], dtype=np.int32),
            param_idxs=np.array([[0,1],[0,1]], dtype=np.int32)
        )

        angle_params = [
            tf.get_variable("HOH_ka", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(75.0)),
            tf.get_variable("HOH_a0", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(ideal_angle)),
        ]

        ha = bonded_force.HarmonicAngleForce(
            params=angle_params,
            angle_idxs=np.array([[1,0,2]], dtype=np.int32),
            param_idxs=np.array([[0,1]], dtype=np.int32)
        )

        # standard MD used to generate a canonical ensemble
        friction = 1.0
        dt = 0.0025
        temp = 300.0
        num_steps = 20000

        x_ph = tf.placeholder(name="input_geom", dtype=tf.float64, shape=(num_atoms, 3))
        intg = integrator.LangevinIntegrator(
            masses, x_ph, [hb, ha], dt, friction, temp)

        reservoir_size = 200

        ref_dx_op, _ = intg.step_op(inference=True)

        sess = tf.Session()
        sess.run(tf.initializers.global_variables())

        def reference_generator():
            x = np.copy(x_opt) # (ytz): Do not remove this copy else you'll spend hours tracking down bugs.
            for step in range(num_steps):
                if step % 100 == 0:
                    print(step)
                dx_val = sess.run(ref_dx_op, feed_dict={x_ph: x})
                x += dx_val
                # this copy is important here, otherwise we're just adding the same reference
                # since += modifies the object in_place
                yield np.copy(x), step # wip: decouple


        def generate_reference_ensemble():
            intg.reset(sess)
            rs = ReservoirSampler(reference_generator(), reservoir_size)
            rs.sample_all()

            ensemble = []
            for x, t in rs.R:
                ensemble.append(x)
            stacked_ensemble = np.stack(ensemble, axis=0)
            ensemble_ph = tf.placeholder(shape=(reservoir_size, num_atoms, 3), dtype=np.float64)

            ref_d2ij_op = observable.sorted_squared_distances(ensemble_ph)
            return ref_d2ij_op, stacked_ensemble, ensemble_ph

        a1, inp1, ensemble_ph1 = generate_reference_ensemble()
        a2, inp2, ensemble_ph2 = generate_reference_ensemble()

        # compute MSE of two _identical_ simulations except for the RNG
        loss = tf.reduce_sum(tf.pow(a1-a2, 2))/reservoir_size # 0.003

        mutual_MSE = sess.run(loss, feed_dict={ensemble_ph1: inp1, ensemble_ph2: inp2})

        print("Optimal MSE", mutual_MSE)

        # on average, two identical ensembles should yield the same result
        assert mutual_MSE < 0.05

        # Use completely different integration parameters
        friction = 10.0
        dt = 0.05
        temp = 100
        num_steps = 500 # we need to make the num_steps variable and only stop once we've converged

        x_ph = tf.placeholder(name="input_geom", dtype=tf.float64, shape=(num_atoms, 3))

        with tf.variable_scope("bad"):
            bad_intg = integrator.LangevinIntegrator(
                masses, x_ph, [hb, ha], dt, friction, temp, buffer_size=None)

        bad_dx_op, bad_dxdp_op = bad_intg.step_op()

        d0_ph = tf.placeholder(shape=tuple(), dtype=tf.float64)     
        d1_ph = tf.placeholder(shape=tuple(), dtype=tf.float64)        
        d2_ph = tf.placeholder(shape=tuple(), dtype=tf.float64)        
        d3_ph = tf.placeholder(shape=tuple(), dtype=tf.float64)        

        grads_and_vars = []
        for dp, var in zip([d0_ph, d1_ph, d2_ph, d3_ph], bond_params+angle_params):        
            grads_and_vars.append((dp, var))

        # param_optimizer = tf.train.AdamOptimizer(0.02)
        # param_optimizer = tf.train.AdamOptimizer(0.1) # unstable
        param_optimizer = tf.train.RMSPropOptimizer(0.1)
        train_op = param_optimizer.apply_gradients(grads_and_vars)
        sess.run(tf.initializers.global_variables())

        # it turns out the force constants don't really matter a whole lot in this case, but the
        # ideal lengths/angles do matter.

        # Use completely different forcefield parameters, changing bond constants and lengths.
        # See if we can recover the original parameters again.
        sess.run([
            tf.assign(bond_params[0], 60),
            tf.assign(bond_params[1], 1.3),
            tf.assign(angle_params[0], 43),
            tf.assign(angle_params[1], 2.1)
        ])

        print("Starting params", sess.run(bond_params+angle_params))

        for epoch in range(10000):

            def sub_optimal_generator():
                x = np.copy(x_opt)
                for step in range(num_steps):
                    dx_val, dxdp_val = sess.run([bad_dx_op, bad_dxdp_op], feed_dict={x_ph: x})
                    x += dx_val
                    yield np.copy(x), dxdp_val, step # wip: decouple

            bad_intg.reset(sess)
            rs = ReservoirSampler(sub_optimal_generator(), reservoir_size)
            rs.sample_all()
            
            bad_ensemble = []
            bad_ensemble_grads = []
            for x, dxdp, t in rs.R:
                bad_ensemble.append(x)
                bad_ensemble_grads.append(dxdp)
            stacked_bad_ensemble = np.stack(bad_ensemble, axis=0)
            stacked_bad_ensemble_grads = np.stack(bad_ensemble_grads, axis=0)

            # compute ensemble's average angle and bond length
            bad_ensemble_ph = tf.placeholder(shape=(reservoir_size, num_atoms, 3), dtype=np.float64)

            ref_d2ij_op = observable.sorted_squared_distances(bad_ensemble_ph)

            # b1, bnp1, bbad_ensemble_ph1 = generate_sub_optimal_bad_ensemble()
            loss_op = tf.reduce_sum(tf.pow(a1-ref_d2ij_op, 2))/reservoir_size # 0.003
            dLdx_op = tf.gradients(loss_op, bad_ensemble_ph)
            loss, dLdx_val = sess.run([loss_op, dLdx_op], feed_dict={
                ensemble_ph1: inp1,
                bad_ensemble_ph: stacked_bad_ensemble
            }) # MSE 12.953122852970827

            dLdx_dxdp = np.multiply(np.expand_dims(dLdx_val[0], 1), stacked_bad_ensemble_grads)
            reduced_dLdp = np.sum(dLdx_dxdp, axis=tuple([0,2,3]))

            sess.run(train_op, feed_dict={      
                 d0_ph: reduced_dLdp[0],        
                 d1_ph: reduced_dLdp[1],        
                 d2_ph: reduced_dLdp[2],        
                 d3_ph: reduced_dLdp[3],        
             })

            if loss < mutual_MSE*5:
                # succesfully converged (should take about 600 epochs)
                return

            print("loss", loss, "epoch", epoch, "current params", sess.run(bond_params+angle_params))


        assert 0

    @unittest.skip("Skipping broken water frequency test")
    def test_optimize_water_frequencies(self):
        masses = np.array([8.0, 1.0, 1.0])
        x_opt = np.array([
            [-0.0070, -0.0100, 0.0000],
            [-0.1604,  0.4921, 0.0000],
            [ 0.5175,  0.0128, 0.0000],
        ], dtype=np.float64)
        x_opt.setflags(write=False) # idealized geometry

        bonds = x_opt - x_opt[0, :]
        bond_lengths = np.linalg.norm(bonds[1:, :], axis=1)

        num_atoms = len(masses)

        starting_bond = 1.47 # Guessestimate starting (true x_opt: 0.52)
        starting_angle = 0.07 # Guessestimate ending (true x_opt: 1.81)

        bond_params = [
            tf.get_variable("OH_kb", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(100.0)),
            tf.get_variable("OH_b0", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(starting_bond)),
        ]

        hb = bonded_force.HarmonicBondForce(
            params=bond_params,
            bond_idxs=np.array([[0,1],[0,2]], dtype=np.int32),
            param_idxs=np.array([[0,1],[0,1]], dtype=np.int32)
        )

        angle_params = [
            tf.get_variable("HOH_ka", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(75.0)),
            tf.get_variable("HOH_a0", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(starting_angle)),
        ]

        ha = bonded_force.HarmonicAngleForce(
            params=angle_params,
            angle_idxs=np.array([[1,0,2]], dtype=np.int32),
            param_idxs=np.array([[0,1]], dtype=np.int32)
        )

        friction = 10.0
        dt = 0.005
        temp = 0.0

        x_ph = tf.placeholder(name="input_geom", dtype=tf.float64, shape=(num_atoms, 3))
        intg = integrator.LangevinIntegrator(
            masses, x_ph, [hb, ha], dt, friction, temp)

        dx_op, dxdp_op = intg.step_op()

        num_steps = 500

        # param_optimizer = tf.train.AdamOptimizer(0.02)
        param_optimizer = tf.train.RMSPropOptimizer(0.02)

        def loss(pred_x):

            test_eigs = observable.vibrational_eigenvalues(pred_x, masses, [hb, ha])
            true_freqs = [0,0,0,40.63,59.383,66.44,1799.2,3809.46,3943] # from http://gaussian.com/vib/
            true_eigs = [(x/VIBRATIONAL_CONSTANT)**2 for x in true_freqs]
            return tf.sqrt(tf.reduce_sum(tf.pow(true_eigs - test_eigs, 2))), test_eigs

        # geometry we arrive at at time t=inf
        x_final_ph = tf.placeholder(dtype=tf.float64, shape=(num_atoms, 3))
        loss_op, test_eigs_op = loss(x_final_ph)
        dLdx = tf.gradients(loss_op, x_final_ph)

        grads_and_vars = intg.grads_and_vars(dLdx[0])
        train_op = param_optimizer.apply_gradients(grads_and_vars)

        sess = tf.Session()
        sess.run(tf.initializers.global_variables())

        num_epochs = 750

        for e in range(num_epochs):
            x = np.copy(x_opt)
            intg.reset(sess) # clear integration buffers
            for step in range(num_steps):
                dx_val, dxdp_val = sess.run([dx_op, dxdp_op], feed_dict={x_ph: x})
                x += dx_val

            _, loss, evs = sess.run([train_op, loss_op, test_eigs_op], feed_dict={x_final_ph: x})
            print("starting epoch", e, "loss", loss, "current params", sess.run(bond_params+angle_params), evs)

        params = sess.run(bond_params+angle_params)
        np.testing.assert_almost_equal(params[1], 0.52, decimal=2)
        np.testing.assert_almost_equal(params[3], 1.81, decimal=1)

    def test_optimize_single_structure(self):
        """
        Testing optimization of a single structure.
        """
        masses = np.array([8.0, 1.0, 1.0])
        x0 = np.array([
            [-0.0070, -0.0100, 0.0000],
            [-1.1426,  0.5814, 0.0000],
            [ 0.4728, -0.2997, 0.0000],
        ], dtype=np.float64) # starting geometry
        x0.setflags(write=False)

        x_opt = np.array([
            [-0.0070, -0.0100, 0.0000],
            [-0.1604,  0.4921, 0.0000],
            [ 0.5175,  0.0128, 0.0000],
        ], dtype=np.float64) # idealized geometry
        x_opt.setflags(write=False)

        bonds = x_opt - x_opt[0, :]
        bond_lengths = np.linalg.norm(bonds[1:, :], axis=1)

        num_atoms = len(masses)

        starting_bond = 0.8 # Guessestimate starting (true x_opt: 0.52)
        starting_angle = 2.1 # Guessestimate ending (true x_opt: 1.81)

        bond_params = [
            tf.get_variable("OH_kb", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(100.0)),
            tf.get_variable("OH_b0", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(starting_bond)),
        ]

        hb = bonded_force.HarmonicBondForce(
            params=bond_params,
            bond_idxs=np.array([[0,1],[0,2]], dtype=np.int32),
            param_idxs=np.array([[0,1],[0,1]], dtype=np.int32)
        )

        angle_params = [
            tf.get_variable("HOH_ka", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(75.0)),
            tf.get_variable("HOH_a0", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(starting_angle)),
        ]

        ha = bonded_force.HarmonicAngleForce(
            params=angle_params,
            angle_idxs=np.array([[1,0,2]], dtype=np.int32),
            param_idxs=np.array([[0,1]], dtype=np.int32)
        )

        friction = 10.0
        dt = 0.005
        # temp = 50.0
        temp = 0.0

        x_ph = tf.placeholder(name="input_geom", dtype=tf.float64, shape=(num_atoms, 3))
        intg = integrator.LangevinIntegrator(
            masses, x_ph, [hb, ha], dt, friction, temp)

        dx_op, dxdp_op = intg.step_op()

        num_steps = 500

        # param_optimizer = tf.train.AdamOptimizer(0.02)
        param_optimizer = tf.train.RMSPropOptimizer(0.01)

        def loss(pred_x):

            # Compute pairwise distances
            def dij(x):
                v01 = x[0]-x[1]
                v02 = x[0]-x[2]
                v12 = x[1]-x[2]
                return tf.stack([tf.norm(v01), tf.norm(v02), tf.norm(v12)])

            return tf.norm(dij(x_opt) - dij(pred_x))

        # geometry we arrive at at time t=inf
        x_final_ph = tf.placeholder(dtype=tf.float64, shape=(num_atoms, 3))
        dLdx = tf.gradients(loss(x_final_ph), x_final_ph)

        grads_and_vars = intg.grads_and_vars(dLdx)
        train_op = param_optimizer.apply_gradients(grads_and_vars)

        sess = tf.Session()
        sess.run(tf.initializers.global_variables())

        num_epochs = 750

        for e in range(num_epochs):
            print("starting epoch", e, "current params", sess.run(bond_params+angle_params))
            x = np.copy(x0)
            intg.reset(sess) # clear integration buffers
            for step in range(num_steps):
                dx_val, dxdp_val = sess.run([dx_op, dxdp_op], feed_dict={x_ph: x})
                x += dx_val

            sess.run(train_op, feed_dict={x_final_ph: x})

        params = sess.run(bond_params+angle_params)
        np.testing.assert_almost_equal(params[1], 0.52, decimal=2)
        np.testing.assert_almost_equal(params[3], 1.81, decimal=1)
           
if __name__ == "__main__":
    unittest.main()