import gc
import unittest
import weakref

import jax
import numpy as np
import pytest
from common import prepare_nb_system

from timemachine import constants
from timemachine.ff import Forcefield
from timemachine.integrator import langevin_coefficients
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat, VelocityVerletIntegrator, custom_ops
from timemachine.lib.potentials import SummedPotential
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.md.enhanced import get_solvent_phase_system
from timemachine.testsystems.ligands import get_biphenyl

pytestmark = [pytest.mark.memcheck]


class TestContext(unittest.TestCase):
    def test_multiple_steps_store_interval(self):
        np.random.seed(2022)

        N = 8
        D = 3

        x0 = np.random.rand(N, D).astype(dtype=np.float64) * 2

        E = 2

        params, potential = prepare_nb_system(x0, E, p_scale=3.0, cutoff=1.0)
        test_nrg = potential.to_gpu()

        masses = np.random.rand(N)
        v0 = np.random.rand(x0.shape[0], x0.shape[1])

        temperature = 300
        dt = 2e-3
        friction = 0.0

        box = np.eye(3) * 3.0
        intg = custom_ops.LangevinIntegrator(masses, temperature, dt, friction, 1234)

        bp = test_nrg.bind(params).bound_impl(precision=np.float64)
        bps = [bp]

        ctxt = custom_ops.Context(x0, v0, box, intg, bps)
        test_xs, test_boxes = ctxt.multiple_steps(10, 10)
        assert len(test_xs) == 1
        assert len(test_xs) == len(test_boxes)
        # We should not get out the input frame
        assert np.any(np.not_equal(x0, test_xs[0]))

        # The current coordinates should match, as the number of steps and the interval match
        np.testing.assert_array_equal(test_xs[0], ctxt.get_x_t())
        _, _ = bps[0].execute(test_xs[0], test_boxes[0])

        # Given an interval greater than the number of steps, return empty arrays
        test_xs, test_boxes = ctxt.multiple_steps(10, 100)
        assert len(test_xs) == 0
        assert len(test_boxes) == 0

        # Given interval of 0, return the last frame
        test_xs, test_boxes = ctxt.multiple_steps(10, 0)
        assert len(test_xs) == 1
        assert len(test_boxes) == 1

        np.testing.assert_array_equal(test_xs[0], ctxt.get_x_t())
        _, _ = bps[0].execute(test_xs[0], test_boxes[0])

    def test_multiple_steps_U_store_interval(self):
        np.random.seed(2022)

        N = 8
        D = 3

        x0 = np.random.rand(N, D).astype(dtype=np.float64) * 2

        E = 2

        params, potential = prepare_nb_system(x0, E, p_scale=3.0, cutoff=1.0)
        test_nrg = potential.to_gpu()

        masses = np.random.rand(N)
        v0 = np.random.rand(x0.shape[0], x0.shape[1])

        temperature = 300
        dt = 2e-3
        friction = 0.0

        box = np.eye(3) * 3.0
        intg = custom_ops.LangevinIntegrator(masses, temperature, dt, friction, 1234)

        bp = test_nrg.bind(params).bound_impl(precision=np.float64)
        bps = [bp]

        ctxt = custom_ops.Context(x0, v0, box, intg, bps)
        test_us, test_xs, test_boxes = ctxt.multiple_steps_U(10, 10, 10)
        assert len(test_xs) == 1
        assert test_us.shape == (1,)
        assert len(test_xs) == len(test_boxes)
        # We should not get out the input frame
        assert np.any(np.not_equal(x0, test_xs[0]))

        # The current coordinates should match, as the number of steps and the interval match
        np.testing.assert_array_equal(test_xs[0], ctxt.get_x_t())
        _, test_frame_u = bps[0].execute(test_xs[0], test_boxes[0])
        np.testing.assert_array_equal(test_us[0], test_frame_u)

        # Given an interval greater than the number of steps, return empty arrays
        test_us, test_xs, test_boxes = ctxt.multiple_steps_U(10, 100, 100)
        assert len(test_xs) == 0
        assert len(test_us) == 0
        assert len(test_boxes) == 0

        # Given interval of 0, return the last frame
        test_us, test_xs, test_boxes = ctxt.multiple_steps_U(10, 0, 0)
        assert len(test_xs) == 1
        assert test_us.shape == (1,)
        assert len(test_boxes) == 1

        np.testing.assert_array_equal(test_xs[0], ctxt.get_x_t())
        _, test_frame_u = bps[0].execute(test_xs[0], test_boxes[0])
        np.testing.assert_array_equal(test_us[0], test_frame_u)

    def test_set_and_get(self):
        """
        This test the setters and getters in the context.
        """

        np.random.seed(4321)

        N = 8
        D = 3

        x0 = np.random.rand(N, D).astype(dtype=np.float64) * 2

        E = 2

        params, potential = prepare_nb_system(x0, E, p_scale=3.0, cutoff=1.0)
        test_nrg = potential.to_gpu()

        masses = np.random.rand(N)
        v0 = np.random.rand(x0.shape[0], x0.shape[1])

        temperature = 300
        dt = 2e-3
        friction = 0.0

        box = np.eye(3) * 3.0
        intg = custom_ops.LangevinIntegrator(masses, temperature, dt, friction, 1234)

        bp = test_nrg.bind(params).bound_impl(precision=np.float64)
        bps = [bp]

        ctxt = custom_ops.Context(x0, v0, box, intg, bps)

        np.testing.assert_equal(ctxt.get_x_t(), x0)
        np.testing.assert_equal(ctxt.get_v_t(), v0)
        np.testing.assert_equal(ctxt.get_box(), box)

        new_x = np.random.rand(N, 3)
        ctxt.set_x_t(new_x)

        np.testing.assert_equal(ctxt.get_x_t(), new_x)

    def test_fwd_mode(self):
        """
        This test ensures that we can reverse-mode differentiate
        observables that are dU_dlambdas of each state. We provide
        adjoints with respect to each computed dU/dLambda.
        """

        np.random.seed(4321)

        N = 8
        D = 3

        x0 = np.random.rand(N, D).astype(dtype=np.float64) * 2

        E = 2

        params, potential = prepare_nb_system(
            x0,
            E,
            p_scale=3.0,
            # cutoff=0.5,
            cutoff=1.0,
        )
        ref_nrg_fn = potential.to_reference()
        test_nrg = potential.to_gpu()

        masses = np.random.rand(N)

        v0 = np.random.rand(x0.shape[0], x0.shape[1])

        num_steps = 12
        temperature = 300
        dt = 1.5e-3
        friction = 0.0
        ca, cbs, ccs = langevin_coefficients(temperature, dt, friction, masses)

        # not convenient to simulate identical trajectories otherwise
        assert (ccs == 0).all()

        def integrate_once_through(x_t, v_t, box, params):

            dU_dx_fn = jax.grad(ref_nrg_fn, argnums=(0,))
            dU_dp_fn = jax.grad(ref_nrg_fn, argnums=(1,))

            all_du_dps = []
            all_xs = []
            all_du_dxs = []
            all_us = []

            def compute_reference_values():
                u = ref_nrg_fn(x_t, params, box)
                all_us.append(u)
                du_dp = dU_dp_fn(x_t, params, box)[0]
                all_du_dps.append(du_dp)
                du_dx = dU_dx_fn(x_t, params, box)[0]
                all_du_dxs.append(du_dx)
                all_xs.append(x_t)

            for step in range(num_steps):
                compute_reference_values()

                noise = np.random.randn(*v_t.shape)
                force = -all_du_dxs[-1]
                v_mid = v_t + np.expand_dims(cbs, axis=-1) * force

                v_t = ca * v_mid + np.expand_dims(ccs, axis=-1) * noise
                x_t += 0.5 * dt * (v_mid + v_t)

            # Compute them for the last set of coords
            compute_reference_values()
            return all_xs, all_du_dxs, all_du_dps, all_us

        box = np.eye(3) * 3.0

        # when we have multiple parameters, we need to set this up correctly
        ref_all_xs, ref_all_du_dxs, ref_all_du_dps, ref_all_us = integrate_once_through(x0, v0, box, params)

        intg = custom_ops.LangevinIntegrator(masses, temperature, dt, friction, 1234)

        bp = test_nrg.bind(params).bound_impl(precision=np.float64)
        bps = [bp]

        ctxt = custom_ops.Context(x0, v0, box, intg, bps)
        ctxt.initialize()
        for step in range(num_steps):
            print("comparing step", step)
            test_x_t = ctxt.get_x_t()
            np.testing.assert_allclose(test_x_t, ref_all_xs[step])
            test_du_dx_t, _ = bp.execute(test_x_t, box)
            ctxt.step()
            # np.testing.assert_allclose(test_u_t, ref_all_us[step])
            np.testing.assert_allclose(test_du_dx_t, ref_all_du_dxs[step])
        ctxt.finalize()
        # test the multiple_steps method
        ctxt_2 = custom_ops.Context(x0, v0, box, intg, bps)

        x_interval = 2
        start_box = ctxt_2.get_box()
        test_xs, test_boxes = ctxt_2.multiple_steps(num_steps, x_interval)
        end_box = ctxt_2.get_box()

        np.testing.assert_allclose(test_xs, ref_all_xs[x_interval::x_interval])
        np.testing.assert_array_equal(start_box, end_box)
        for i in range(test_boxes.shape[0]):
            np.testing.assert_array_equal(start_box, test_boxes[i])
        self.assertEqual(test_boxes.shape[0], test_xs.shape[0])
        self.assertEqual(test_boxes.shape[1], D)
        self.assertEqual(test_boxes.shape[2], test_xs.shape[2])

        # test the multiple_steps_U method
        ctxt_3 = custom_ops.Context(x0, v0, box, intg, bps)

        u_interval = 3

        test_us, test_xs, test_boxes = ctxt_3.multiple_steps_U(num_steps, u_interval, x_interval)
        np.testing.assert_array_almost_equal(ref_all_us[u_interval::u_interval], test_us)

        np.testing.assert_array_almost_equal(ref_all_xs[x_interval::x_interval], test_xs)

    def test_multiple_steps_local_validation(self):
        seed = 2022
        np.random.seed(seed)

        N = 8
        D = 3

        coords = np.random.rand(N, D).astype(dtype=np.float64) * 2
        box = np.eye(3) * 3.0
        masses = np.random.rand(N)

        E = 2

        params, potential = prepare_nb_system(
            coords,
            E,
            p_scale=3.0,
            cutoff=1.0,
        )
        nb_pot = potential.to_gpu()

        temperature = 300
        dt = 1.5e-3
        friction = 0.0
        radius = 1.2

        # Select a single particle to use as the reference, will be frozen
        local_idxs = np.array([len(coords) - 1], dtype=np.int32)

        v0 = np.zeros_like(coords)
        bps = [nb_pot.bind(params).bound_impl(np.float32)]

        reference_values = []
        for bp in bps:
            reference_values.append(bp.execute(coords, box))

        intg = LangevinIntegrator(temperature, dt, friction, masses, seed)

        intg_impl = intg.impl()

        verlet = VelocityVerletIntegrator(dt, masses)
        verlet_impl = verlet.impl()

        # If the integrator is not a thermostat, should fail
        ctxt = custom_ops.Context(coords, v0, box, verlet_impl, bps)
        with pytest.raises(RuntimeError, match="integrator must be LangevinIntegrator."):
            ctxt.multiple_steps_local(100, local_idxs, radius=radius)

        # Construct context with no potentials, local MD should fail.
        ctxt = custom_ops.Context(coords, v0, box, intg_impl, [])
        with pytest.raises(RuntimeError, match="unable to find a NonbondedAllPairs potential"):
            ctxt.multiple_steps_local(100, local_idxs, radius=radius)

        # If you have multiple nonbonded potentials, should fail
        ctxt = custom_ops.Context(coords, v0, box, intg_impl, bps * 2)
        with pytest.raises(RuntimeError, match="found multiple NonbondedAllPairs potentials"):
            ctxt.multiple_steps_local(100, local_idxs, radius=radius)

        # Verify that indices are correctly checked
        ctxt = custom_ops.Context(coords, v0, box, intg_impl, bps)
        with pytest.raises(RuntimeError, match="indices can't be empty"):
            ctxt.multiple_steps_local(100, np.array([], dtype=np.int32), radius=radius)

        with pytest.raises(RuntimeError, match="index values must be less than N"):
            ctxt.multiple_steps_local(100, np.array([N * 2], dtype=np.int32), radius=radius)

        with pytest.raises(RuntimeError, match="index values must be greater or equal to zero"):
            ctxt.multiple_steps_local(100, np.array([-1], dtype=np.int32), radius=radius)

        with pytest.raises(RuntimeError, match="atom indices must be unique"):
            ctxt.multiple_steps_local(100, np.array([1, 1], dtype=np.int32), radius=radius)

        with pytest.raises(RuntimeError, match="burn in steps must be greater or equal to zero"):
            ctxt.multiple_steps_local(100, np.array([1], dtype=np.int32), radius=radius, burn_in=-5)

        with pytest.raises(RuntimeError, match="radius must be greater or equal to zero"):
            ctxt.multiple_steps_local(100, np.array([1], dtype=np.int32), radius=-0.1)

        with pytest.raises(RuntimeError, match="k must be greater than zero"):
            ctxt.multiple_steps_local(100, np.array([1], dtype=np.int32), k=0.0)

        with pytest.raises(RuntimeError, match="k must be less than than 1000000.0"):
            ctxt.multiple_steps_local(100, np.array([1], dtype=np.int32), k=1e7)

    def test_multiple_steps_local_burn_in(self):
        """Verify that burn in steps are identical to regular steps"""
        seed = 2022
        np.random.seed(seed)

        N = 8
        D = 3

        coords = np.random.rand(N, D).astype(dtype=np.float64) * 2
        box = np.eye(3) * 3.0
        masses = np.random.rand(N)

        E = 2

        params, potential = prepare_nb_system(
            coords,
            E,
            p_scale=3.0,
            cutoff=1.0,
        )
        nb_pot = potential.to_gpu()

        temperature = 300
        dt = 1.5e-3
        friction = 0.0
        radius = 1.2

        # Select a single particle to use as the reference, will be frozen
        local_idxs = np.array([len(coords) - 1], dtype=np.int32)

        v0 = np.zeros_like(coords)
        bps = [nb_pot.bind(params).bound_impl(np.float32)]

        reference_values = []
        for bp in bps:
            reference_values.append(bp.execute(coords, box))

        intg = LangevinIntegrator(temperature, dt, friction, masses, seed)

        intg_impl = intg.impl()

        steps = 100
        burn_in = 100

        ctxt = custom_ops.Context(coords, v0, box, intg_impl, bps)
        ref_xs, ref_boxes = ctxt.multiple_steps_local(steps, local_idxs, radius=radius, burn_in=burn_in)

        intg_impl = intg.impl()

        ctxt = custom_ops.Context(coords, v0, box, intg_impl, bps)
        comp_xs, comp_boxes = ctxt.multiple_steps_local(steps + burn_in, local_idxs, radius=radius, burn_in=0)

        # Final frame should be identical
        np.testing.assert_array_equal(ref_xs, comp_xs)
        np.testing.assert_array_equal(ref_boxes, comp_boxes)

    def test_multiple_steps_local_consistency(self):
        """Verify that running multiple_steps_local is consistent.

        - Assert that particles nearby the local idxs move, particles far away do not
        - Assert that potentials used to run local md do return bit wise identical results.
          As multiple_steps_local modifies the potentials
        - Assert that running with a Barostat returns identical frames
        - Assert that wrapping potentials within a SummedPotential returns identical frames"""
        mol, _ = get_biphenyl()
        ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

        temperature = 300
        dt = 1.5e-3
        friction = 0.0
        seed = 2022
        radius = 1.2
        num_steps = 500
        x_interval = 100

        unbound_potentials, sys_params, masses, coords, box = get_solvent_phase_system(
            mol, ff, 0.0, minimize_energy=False
        )
        v0 = np.zeros_like(coords)
        bps = []
        for p, bp in zip(sys_params, unbound_potentials):
            bps.append(bp.bind(p).bound_impl(np.float32))

        reference_values = []
        for bp in bps:
            reference_values.append(bp.execute(coords, box))

        # Select the molecule as the local idxs
        local_idxs = np.arange(len(coords) - mol.GetNumAtoms(), len(coords), dtype=np.int32)

        intg = LangevinIntegrator(temperature, dt, friction, masses, seed)

        intg_impl = intg.impl()

        ctxt = custom_ops.Context(coords, v0, box, intg_impl, bps)
        # Run steps of local MD
        xs, boxes = ctxt.multiple_steps_local(num_steps, local_idxs, store_x_interval=x_interval, radius=radius)

        assert xs.shape[0] == num_steps // x_interval
        assert boxes.shape[0] == num_steps // x_interval

        # Indices in local idxs should have moved, except for the one selected as frozen
        moved_coords = np.sum(coords[local_idxs] != xs[-1][local_idxs])
        assert (moved_coords // 3) == len(local_idxs) - 1

        # Get the particles within a certain distance of local idxs
        nblist = custom_ops.Neighborlist_f32(len(coords))
        nblist.set_row_idxs(local_idxs.astype(np.uint32))
        # Add padding to the radius to account for probablistic selection
        ixn_list = nblist.get_nblist(coords, box, radius + 0.2)
        potential_selected_particles = np.concatenate(ixn_list)

        moving_idxs = np.concatenate([local_idxs, potential_selected_particles.reshape(-1)])
        assert np.any(coords[moving_idxs] != xs[-1][moving_idxs])

        frozen_idxs = np.delete(np.arange(0, len(coords)), moving_idxs)
        assert len(frozen_idxs) > 0
        assert np.all(coords[frozen_idxs] == xs[-1][frozen_idxs])

        # Verify that the bound potentials haven't been changed, as local md modifies potentials
        for ref_val, bp in zip(reference_values, bps):
            ref_du_dx, ref_u = ref_val
            test_du_dx, test_u = bp.execute(coords, box)
            np.testing.assert_array_equal(ref_du_dx, test_du_dx)
            np.testing.assert_equal(ref_u, test_u)

        # Verify that running with a barostat doesn't change the results
        group_idxs = get_group_indices(get_bond_list(unbound_potentials[0]))

        pressure = 1.0

        barostat = MonteCarloBarostat(coords.shape[0], pressure, temperature, group_idxs, 1, seed)
        barostat_impl = barostat.impl(bps)

        intg_impl = intg.impl()

        ctxt = custom_ops.Context(coords, v0, box, intg_impl, bps, barostat=barostat_impl)
        baro_xs, baro_boxes = ctxt.multiple_steps_local(
            num_steps, local_idxs, store_x_interval=x_interval, radius=radius
        )

        assert baro_xs.shape == xs.shape
        assert baro_boxes.shape == boxes.shape

        # Results using a barostat should be identical.
        np.testing.assert_array_equal(baro_xs, xs)
        np.testing.assert_array_equal(baro_boxes, boxes)

        # Verify that wrapping up the potentials in a summed potential is identical
        summed_potential = SummedPotential(unbound_potentials, sys_params)
        # Flatten the arrays so we can concatenate them.
        summed_potential = summed_potential.bind(np.concatenate([p.reshape(-1) for p in sys_params]))
        bp = summed_potential.bound_impl(precision=np.float32)

        intg_impl = intg.impl()

        # Rerun with the summed potential
        ctxt = custom_ops.Context(coords, v0, box, intg_impl, [bp])
        summed_pot_xs, summed_pot_boxes = ctxt.multiple_steps_local(
            num_steps, local_idxs, store_x_interval=x_interval, radius=radius
        )

        assert summed_pot_xs.shape == xs.shape
        assert summed_pot_boxes.shape == boxes.shape

        # Results using a summed potential should be identical.
        np.testing.assert_array_equal(summed_pot_xs, xs)
        np.testing.assert_array_equal(summed_pot_boxes, boxes)

    def test_multiple_steps_local_entire_system(self):
        """Verify that running multiple_steps_local is valid even when consuming the entire system, IE radius ~= inf.

        - Only a single particle, the reference, should not move
        """
        mol, _ = get_biphenyl()
        ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

        temperature = 300
        dt = 1.5e-3
        friction = 1.0
        seed = 2022
        radius = np.inf
        num_steps = 5

        unbound_potentials, sys_params, masses, coords, box = get_solvent_phase_system(
            mol, ff, 0.0, minimize_energy=False
        )
        v0 = np.zeros_like(coords)
        bps = []
        for p, bp in zip(sys_params, unbound_potentials):
            bps.append(bp.bind(p).bound_impl(np.float32))

        # Select the molecule as the local idxs
        local_idxs = np.arange(len(coords) - mol.GetNumAtoms(), len(coords), dtype=np.int32)

        intg = LangevinIntegrator(temperature, dt, friction, masses, seed)

        intg_impl = intg.impl()

        ctxt = custom_ops.Context(coords, v0, box, intg_impl, bps)

        xs, boxes = ctxt.multiple_steps_local(num_steps, local_idxs, radius=radius)

        identical_positions = (xs == coords).sum(axis=1)
        assert len(identical_positions) == 1, "Expected only a single atom to be stationary"

    def test_multiple_steps_local_no_free_particles(self):
        """Verify that running multiple_steps_local raises an exception if no free particles selected.

        - For tiny radius and large k, should select nothing
        """
        mol, _ = get_biphenyl()
        ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

        temperature = 300
        dt = 1.5e-3
        friction = 1.0
        seed = 2022
        num_steps = 5

        # tiny radius and large k can produce no free particles, which is a pointless move
        radius = np.finfo(float).eps
        k = 1.0e7

        unbound_potentials, sys_params, masses, coords, box = get_solvent_phase_system(
            mol, ff, 0.0, minimize_energy=False
        )
        v0 = np.zeros_like(coords)
        bps = []
        for p, bp in zip(sys_params, unbound_potentials):
            bps.append(bp.bind(p).bound_impl(np.float32))

        # Select the molecule as the local idxs
        local_idxs = np.arange(len(coords) - mol.GetNumAtoms(), len(coords), dtype=np.int32)

        intg = LangevinIntegrator(temperature, dt, friction, masses, seed)

        intg_impl = intg.impl()

        ctxt = custom_ops.Context(coords, v0, box, intg_impl, bps)

        with pytest.raises(RuntimeError, match="no free particles"):
            xs, boxes = ctxt.multiple_steps_local(num_steps, local_idxs, radius=radius, k=k)

    def test_setup_context_with_references(self):
        mol, _ = get_biphenyl()
        ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

        temperature = 300
        dt = 1.5e-3
        friction = 0.0
        seed = 2022
        pressure = constants.DEFAULT_PRESSURE

        unbound_potentials, sys_params, masses, coords, box = get_solvent_phase_system(
            mol, ff, 0.0, minimize_energy=False
        )
        v0 = np.zeros_like(coords)

        def build_context(barostat_interval):
            """The context returned will segfault if any of the objects get cleaned up"""

            weak_refs = []
            bps = []
            for p, bp in zip(sys_params, unbound_potentials):
                bound_impl = bp.bind(p).bound_impl(np.float32)
                bps.append(bound_impl)
                weak_refs.append(weakref.ref(bound_impl))

            intg = LangevinIntegrator(temperature, dt, friction, masses, seed)

            barostat_impl = None
            if barostat_interval > 0:
                group_idxs = get_group_indices(get_bond_list(unbound_potentials[0]))

                barostat = MonteCarloBarostat(coords.shape[0], pressure, temperature, group_idxs, 1, seed)
                barostat_impl = barostat.impl(bps)
                weak_refs.append(weakref.ref(barostat_impl))

            intg_impl = intg.impl()
            weak_refs.append(weakref.ref(intg_impl))

            return custom_ops.Context(coords, v0, box, intg_impl, bps, barostat=barostat_impl), weak_refs

        # Without barostat
        ctxt, reffed_objs = build_context(0)
        xs, boxes = ctxt.multiple_steps(100)
        assert np.all(np.isfinite(xs))
        assert np.all(np.isfinite(boxes))
        assert np.all(xs[-1] != coords)
        assert np.all(boxes[-1] == box)

        del ctxt
        gc.collect()
        for ref in reffed_objs:
            assert ref() is None

        # With Barostat
        ctxt, reffed_objs = build_context(10)
        xs, boxes = ctxt.multiple_steps(100)
        assert np.all(np.isfinite(xs))
        assert np.all(np.isfinite(boxes))
        assert np.all(xs[-1] != coords)
        # Barostat should change box size
        assert np.all(np.diagonal(boxes[-1]) != np.diagonal(box))

        del ctxt
        gc.collect()
        for ref in reffed_objs:
            assert ref() is None


if __name__ == "__main__":
    unittest.main()
