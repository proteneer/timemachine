import functools
import unittest

import numpy as np

import jax
import jax.numpy as jnp
from jax.config import config; config.update("jax_enable_x64", True)

from timemachine.lib import custom_ops
from timemachine.lib import ops
from timemachine.potentials import bonded, nonbonded, alchemy

from common import GradientTest
from common import prepare_bonded_system, prepare_restraints, prepare_nonbonded_system

class TestContext(unittest.TestCase):

    # def setup_system(self, N):

    #     np.random.seed(125)

    #     P_bonds = 4
    #     P_angles = 6
    #     P_torsions = 13
 
    #     B = 35
    #     A = 36
    #     T = 37

    #     D = 3

    #     x0 = np.random.rand(N,D).astype(dtype=np.float64)*2

    #     precision = np.float64

    #     params, ref_bonded_potentials_0, test_bonded_potentials_0 = prepare_bonded_system(
    #         x0,
    #         P_bonds,
    #         P_angles,
    #         P_torsions,
    #         B,
    #         A,
    #         T,
    #         precision
    #     )

    #     params, ref_bonded_potentials_1, test_bonded_potentials_1 = prepare_bonded_system(
    #         x0,
    #         P_bonds,
    #         P_angles,
    #         P_torsions,
    #         B,
    #         A,
    #         T,
    #         precision,
    #         params=params
    #     )


    #     masses = np.random.rand(N)

    #     E = 1
    #     P_charges = 4
    #     P_lj = 5
    #     P_exc = 7

    #     cutoff = 1.5

    #     params, ref_nonbonded_potentials_0, test_nonbonded_potentials_0 = prepare_nonbonded_system(
    #         x0,
    #         E,
    #         P_charges,
    #         P_lj,
    #         P_exc,
    #         params=params,
    #         p_scale=10.0,
    #         cutoff=cutoff,
    #         precision=precision
    #     )

    #     params, ref_nonbonded_potentials_1, test_nonbonded_potentials_1 = prepare_nonbonded_system(
    #         x0,
    #         E,
    #         P_charges,
    #         P_lj,
    #         P_exc,
    #         params=params,
    #         p_scale=10.0,
    #         cutoff=cutoff,
    #         precision=precision
    #     )

    #     ref_potentials = [] #((ref_bonded_potentials_0, ref_bonded_potentials_1), (ref_nonbonded_potentials_0, ref_nonbonded_potentials_1))
    #     test_potentials = [] #((test_bonded_potentials_0, test_bonded_potentials_1), (test_nonbonded_potentials_0, test_nonbonded_potentials_1))

    #     for a, b in zip(ref_bonded_potentials_0, ref_bonded_potentials_1):
    #         ref_potentials.append((a,b))
    #     for a, b in zip(ref_nonbonded_potentials_0, ref_nonbonded_potentials_1):
    #         ref_potentials.append((a,b))

    #     for a, b in zip(test_bonded_potentials_0, test_bonded_potentials_1):
    #         test_potentials.append((a,b))
    #     for a, b in zip(test_nonbonded_potentials_0, test_nonbonded_potentials_1):
    #         test_potentials.append((a,b))


    #     return x0, params, masses, ref_potentials, test_potentials


    # def test_reverse_mode_lambda(self):
    #     """
    #     This test ensures that we can reverse-mode differentiate
    #     observables that are dU_dlambdas of each state. We provide
    #     adjoints with respect to each computed dU/dLambda.
    #     """
    #     np.random.seed(4321)

    #     N = 1023

    #     x0, params, masses, _, test_tuples = self.setup_system(N)

    #     test_fns = []
    #     for test_a, _ in test_tuples:
    #         test_fns.append(test_a)

    #     v0 = np.random.rand(x0.shape[0], x0.shape[1])
    #     N = len(masses)

    #     num_steps = 15
    #     lambda_schedule = np.random.rand(num_steps)
    #     cas = np.random.rand(num_steps)
    #     cbs = np.random.rand(len(masses))/10
    #     ccs = np.zeros_like(cbs)

    #     step_sizes = np.random.rand(num_steps)

    #     stepper0 = custom_ops.AlchemicalStepper_f64(
    #         test_fns,
    #         lambda_schedule
    #     )

    #     seed = 1234

    #     ctxt0 = custom_ops.ReversibleContext_f64(
    #         stepper0,
    #         x0,
    #         v0,
    #         cas,
    #         cbs,
    #         ccs,
    #         step_sizes,
    #         params,
    #         seed
    #     )

    #     # run 5 steps forward
    #     ctxt0.forward_mode()
    #     last_0 = ctxt0.get_last_coords()

    #     stepper1 = custom_ops.AlchemicalStepper_f64(
    #         test_fns,
    #         lambda_schedule
    #     )

    #     ctxt1 = custom_ops.ReversibleContext_f64(
    #         stepper1,
    #         x0,
    #         v0,
    #         cas,
    #         cbs,
    #         ccs,
    #         step_sizes,
    #         params,
    #         seed
    #     )

    #     ctxt1.forward_mode()
    #     last_1 = ctxt1.get_last_coords()

    #     np.testing.assert_equal(last_0, last_1)




    # def setup_system(self, N):

    #     np.random.seed(125)

    #     B = 35
    #     A = 36
    #     T = 37

    #     D = 3

    #     x0 = np.random.rand(N,D).astype(dtype=np.float64)*2

    #     precision = np.float64

    #     bond_params, ref_bond, test_bond = prepare_bonded_system(
    #         x0,
    #         B,
    #         A,
    #         T,
    #         precision
    #     )

    #     masses = np.random.rand(N)

    #     return x0, bond_params, masses, ref_potentials, test_potentials

    def test_reverse_mode_lambda(self):
        """
        This test ensures that we can reverse-mode differentiate
        observables that are dU_dlambdas of each state. We provide
        adjoints with respect to each computed dU/dLambda.
        """

        np.random.seed(4321)


        N = 5
        B = 5
        A = 0
        T = 0
        D = 3

        x0 = np.random.rand(N,D).astype(dtype=np.float64)*2

        precision = np.float64
 
        (bond_params, ref_bond), test_bond = prepare_bonded_system(
            x0,
            B,
            A,
            T,
            precision
        )

        (restr_params, ref_restr), test_restr = prepare_restraints(
            x0,
            B,
            precision
        )

        E = 2

        lambda_plane_idxs = np.random.randint(low=0, high=2, size=N, dtype=np.int32)
        lambda_offset_idxs = np.random.randint(low=0, high=2, size=N, dtype=np.int32)

        (charge_params, lj_params), ref_nb_fn, test_nb_ctor = prepare_nonbonded_system(
            x0,
            E,
            lambda_plane_idxs,
            lambda_offset_idxs,
            p_scale=10.0,
            cutoff=1000.0,
            precision=precision       
        )

        test_nb = test_nb_ctor()

        masses = np.random.rand(N)

        v0 = np.random.rand(x0.shape[0], x0.shape[1])
        N = len(masses)

        num_steps = 5
        lambda_schedule = np.random.rand(num_steps)
        cas = np.random.rand(num_steps)
        cbs = np.random.rand(len(masses))/10
        ccs = np.zeros_like(cbs)

        step_sizes = np.random.rand(num_steps)

        def loss_fn(du_dls):
            return jnp.sum(du_dls*du_dls)/du_dls.shape[0]

        def sum_loss_fn(du_dls):
            du_dls = np.sum(du_dls, axis=0)
            return jnp.sum(du_dls*du_dls)/du_dls.shape[0]            

        def integrate_once_through(
            x_t,
            v_t,
            bond_params,
            restr_params,
            charge_params,
            lj_params):

            ref_bond_impl = functools.partial(ref_bond, params=bond_params)
            ref_restr_impl = functools.partial(ref_restr, params=restr_params)
            ref_nb_impl = functools.partial(ref_nb_fn, charge_params=charge_params, lj_params=lj_params)

            def ref_total_nrg_fn(*args):
                nrgs = []
                for fn in [ref_bond_impl, ref_restr_impl, ref_nb_impl]:
                    nrgs.append(fn(*args))
                return jnp.sum(nrgs)

            dU_dx_fn = jax.grad(ref_total_nrg_fn, argnums=(0,))
            dU_dl_fn = jax.grad(ref_total_nrg_fn, argnums=(1,))

            all_du_dls = []
            for step in range(num_steps):
                lamb = lambda_schedule[step]
                du_dl = dU_dl_fn(x_t, lamb)[0]
                all_du_dls.append(du_dl)
                dt = step_sizes[step]
                cb_tmp = np.expand_dims(cbs, axis=-1)                
                v_t = cas[step]*v_t + cb_tmp*dU_dx_fn(x_t, lamb)[0]
                x_t = x_t + v_t*dt
                # note that we do not calculate the du_dl of the last frame.

            all_du_dls = jnp.stack(all_du_dls)
            return loss_fn(all_du_dls)

        # when we have multiple parameters, we need to set this up correctly
        ref_loss = integrate_once_through(
            x0,
            v0,
            bond_params,
            restr_params,
            charge_params,
            lj_params
        )

        grad_fn = jax.grad(integrate_once_through, argnums=(2, 3))
        ref_dl_dp_bond, ref_dl_dp_restr = grad_fn(
            x0,
            v0,
            bond_params,
            restr_params,
            charge_params,
            lj_params
        )

        stepper = custom_ops.AlchemicalStepper_f64(
            [test_bond, test_restr, test_nb],
            lambda_schedule
        )

        seed = 1234

        ctxt = custom_ops.ReversibleContext_f64(
            stepper,
            x0,
            v0,
            cas,
            cbs,
            ccs,
            step_sizes,
            seed
        )

        # run 5 steps forward
        ctxt.forward_mode()
        test_du_dls = stepper.get_du_dl()
        test_loss = sum_loss_fn(test_du_dls)
        loss_grad_fn = jax.grad(sum_loss_fn, argnums=(0,))
        du_dl_adjoint = loss_grad_fn(test_du_dls)[0]

        # limit of precision is due to the settings in fixed_point.hpp
        # np.testing.assert_almost_equal(test_loss, ref_loss, decimal=7)
        np.testing.assert_allclose(test_loss, ref_loss, rtol=1e-6)
        stepper.set_du_dl_adjoint(du_dl_adjoint)
        ctxt.set_x_t_adjoint(np.zeros_like(x0))
        ctxt.backward_mode()

        test_dl_dp = test_bond.get_du_dp_tangents()
        np.testing.assert_allclose(test_dl_dp, ref_dl_dp_bond, rtol=1e-6)

        test_dl_dp = test_restr.get_du_dp_tangents()
        np.testing.assert_allclose(test_dl_dp, ref_dl_dp_restr, rtol=1e-6)

if __name__ == "__main__":
    unittest.main()