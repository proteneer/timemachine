# (ytz): check test and run benchmark with pytest:
# pytest -xsv tests/test_nonbonded.py::TestNonbonded::test_dhfr && nvprof pytest -xsv tests/test_nonbonded.py::TestNonbonded::test_benchmark
import copy
import itertools
import unittest

import numpy as np
import pytest
from common import GradientTest, gen_nonbonded_params_with_4d_offsets, prepare_system_params, prepare_water_system

from timemachine.constants import DEFAULT_FF
from timemachine.fe.utils import to_md_units
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.lib import potentials
from timemachine.md import builders
from timemachine.potentials import generic
from timemachine.testsystems.dhfr import setup_dhfr

np.set_printoptions(linewidth=500)

pytestmark = [pytest.mark.memcheck]


class TestNonbondedDHFR(GradientTest):
    def setUp(self):
        # This test checks hilbert curve re-ordering gives identical results
        host_fns, _, host_coords, self.box = setup_dhfr()

        for f in host_fns:
            if isinstance(f, potentials.Nonbonded):
                self.nonbonded_fn = f

        host_conf = []
        for x, y, z in host_coords:
            host_conf.append([to_md_units(x), to_md_units(y), to_md_units(z)])
        self.host_conf = np.array(host_conf)

        self.beta = 2.0
        self.cutoff = 1.1

    def test_nblist_hilbert(self):
        """
        This test makes sure that hilbert ordering has no impact on numerical results. The
        computed forces, energies, etc. should be bitwise identical.
        """

        np.random.seed(2021)

        N = self.host_conf.shape[0]

        for precision in [np.float32, np.float64]:

            ref_nonbonded_impl = copy.copy(self.nonbonded_fn).unbound_impl(precision)
            ref_nonbonded_impl.disable_hilbert_sort()

            test_nonbonded_impl = copy.copy(self.nonbonded_fn).unbound_impl(precision)

            padding = 0.1
            deltas = np.random.rand(N, 3) - 0.5  # [-0.5, +0.5]
            divisor = 0.5 * (2 * np.sqrt(3)) / padding
            # if deltas are kept under +- p/(2*sqrt(3)) then no rebuild gets triggered
            deltas = deltas / divisor  # exactly within bounds, and should not trigger a rebuild

            for d in deltas:
                assert np.linalg.norm(d) < padding / 2

            xs = [self.host_conf, self.host_conf + deltas]

            np.set_printoptions(precision=16)
            # under pure fixed point accumulation the results should be identical.
            for x_idx, x in enumerate(xs):

                ref_du_dx, ref_du_dp, ref_u = ref_nonbonded_impl.execute(x, self.nonbonded_fn.params, self.box)
                test_du_dx, test_du_dp, test_u = test_nonbonded_impl.execute(x, self.nonbonded_fn.params, self.box)

                np.testing.assert_array_equal(ref_du_dx, test_du_dx)
                np.testing.assert_array_equal(ref_du_dp, test_du_dp)
                np.testing.assert_array_equal(ref_u, test_u)

                ref_du_dx = ref_nonbonded_impl.execute_du_dx(x, self.nonbonded_fn.params, self.box)
                test_du_dx = test_nonbonded_impl.execute_du_dx(x, self.nonbonded_fn.params, self.box)

                for idx, (a, b) in enumerate(zip(ref_du_dx, test_du_dx)):
                    if np.linalg.norm(a - b) != 0:
                        print(idx, a, b)
                        assert 0
                    np.testing.assert_array_equal(a, b)

                np.testing.assert_array_equal(ref_du_dx, test_du_dx)

    def test_nblist_rebuild(self):
        """
        This test makes sure that periodically rebuilding the neighborlist no impact on numerical results. The
        computed forces, energies, etc. should be bitwise identical.
        """

        N = self.host_conf.shape[0]

        np.random.seed(2021)

        ref_nonbonded_impl = copy.copy(self.nonbonded_fn).unbound_impl(np.float64)
        ref_nonbonded_impl.set_nblist_padding(0.0)  # rebuild with every call

        padding = 0.1

        test_nonbonded_impl = copy.copy(self.nonbonded_fn).unbound_impl(np.float64)
        test_nonbonded_impl.set_nblist_padding(
            padding
        )  # rebuild only when deltas have moved more than padding/2 angstroms

        deltas = np.random.rand(N, 3) - 0.5  # [-0.5, +0.5]
        divisor = 0.5 * (2 * np.sqrt(3)) / padding
        # if deltas are kept under +- p/(2*sqrt(3)) then no rebuild gets triggered
        deltas = deltas / divisor  # exactly within bounds, and should not trigger a rebuild

        for d in deltas:
            assert np.linalg.norm(d) < padding / 2

        xs = [self.host_conf, self.host_conf + deltas]

        # under pure fixed point accumulation the results should be identical.
        for x_idx, x in enumerate(xs):
            ref_du_dx, ref_du_dp, ref_u = ref_nonbonded_impl.execute(x, self.nonbonded_fn.params, self.box)
            test_du_dx, test_du_dp, test_u = test_nonbonded_impl.execute(x, self.nonbonded_fn.params, self.box)

            np.testing.assert_array_equal(ref_du_dx, test_du_dx)
            np.testing.assert_array_equal(ref_du_dp, test_du_dp)
            np.testing.assert_array_equal(ref_u, test_u)

            ref_du_dx = ref_nonbonded_impl.execute_du_dx(x, self.nonbonded_fn.params, self.box)
            test_du_dx = test_nonbonded_impl.execute_du_dx(x, self.nonbonded_fn.params, self.box)

            for idx, (a, b) in enumerate(zip(ref_du_dx, test_du_dx)):
                if np.linalg.norm(a - b) != 0:
                    print(idx, a, b)
                    print(a)
                    print(b)
                    assert 0
                np.testing.assert_array_equal(a, b)

            np.testing.assert_array_equal(ref_du_dx, test_du_dx)

    def test_correctness(self):
        """
        Test against the reference jax platform for correctness.
        """
        # we can't go bigger than this due to memory limitations in the the reference platform.
        for N in [33, 65, 231, 1050, 4080]:

            np.random.seed(2022)

            test_conf = self.host_conf[:N]

            # strip out parts of the system
            test_exclusions = []
            test_scales = []
            for (i, j), (sa, sb) in zip(self.nonbonded_fn.get_exclusion_idxs(), self.nonbonded_fn.get_scale_factors()):
                if i < N and j < N:
                    test_exclusions.append((i, j))
                    test_scales.append((sa, sb))
            test_exclusions = np.array(test_exclusions, dtype=np.int32)
            test_scales = np.array(test_scales, dtype=np.float64)
            test_params = self.nonbonded_fn.params[:N, :]

            potential = generic.Nonbonded(N, test_exclusions, test_scales, self.beta, self.cutoff)

            for precision, rtol, atol in [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)]:

                self.compare_forces_gpu_vs_reference(
                    test_conf,
                    [test_params],
                    self.box,
                    potential,
                    rtol=rtol,
                    atol=atol,
                    precision=precision,
                )

    @unittest.skip("benchmark-only")
    def test_benchmark(self):
        """
        This is mainly for benchmarking nonbonded computations on the initial state.
        """

        precision = np.float32

        nb_fn = copy.deepcopy(self.nonbonded_fn)

        impl = nb_fn.unbound_impl(precision)

        for combo in itertools.product([False, True], repeat=4):

            compute_du_dx, compute_du_dp, compute_u = combo

            for trip in range(50):

                test_du_dx, test_du_dp, test_u = impl.execute_selective(
                    self.host_conf,
                    [self.nonbonded_fn.params],
                    self.box,
                    compute_du_dx,
                    compute_du_dp,
                    compute_u,
                )


class TestNonbondedWater(GradientTest):
    def test_nblist_box_resize(self):
        # test that running the coordinates under two different boxes produces correct results
        # since we should be rebuilding the nblist when the box sizes change.
        ff = Forcefield.load_from_file(DEFAULT_FF)

        host_system, host_coords, box, _ = builders.build_water_system(3.0, ff.water_ff)

        host_fns, host_masses = openmm_deserializer.deserialize_system(host_system, cutoff=1.0)

        test_nonbonded_fn = None
        for f in host_fns:
            if isinstance(f, potentials.Nonbonded):
                test_nonbonded_fn = f
        assert test_nonbonded_fn is not None
        assert test_nonbonded_fn.params is not None

        host_conf = []
        for x, y, z in host_coords:
            host_conf.append([to_md_units(x), to_md_units(y), to_md_units(z)])
        host_conf = np.array(host_conf)

        potential = generic.Nonbonded.from_gpu(test_nonbonded_fn)

        big_box = box + np.eye(3) * 1000

        # (ytz): note the ordering should be from large box to small box. though in the current code
        # the rebuild is triggered as long as the box *changes*.
        for test_box in [big_box, box]:

            for precision, rtol, atol in [(np.float64, 1e-8, 1e-10), (np.float32, 1e-4, 3e-5)]:

                self.compare_forces_gpu_vs_reference(
                    host_conf,
                    [test_nonbonded_fn.params],
                    test_box,
                    potential,
                    rtol=rtol,
                    atol=atol,
                    precision=precision,
                )


class TestNonbonded(GradientTest):
    def test_exclusion(self):

        # This test verifies behavior when two particles are arbitrarily
        # close but are marked as excluded to ensure proper cancellation
        # of exclusions occur in the fixed point math.

        np.random.seed(2020)

        water_coords = self.get_water_coords(3, sort=False)
        test_system = water_coords[:126]  # multiple of 3
        box = np.eye(3) * 3

        N = test_system.shape[0]

        EA = 10

        atom_idxs = np.arange(test_system.shape[0])

        # pick a set of atoms that will be mutually excluded from each other.
        # we will need to set their exclusions manually
        exclusion_atoms = np.random.choice(atom_idxs, size=EA, replace=False)
        exclusion_idxs = []

        for idx, i in enumerate(exclusion_atoms):
            for jdx, j in enumerate(exclusion_atoms):
                if jdx > idx:
                    exclusion_idxs.append((i, j))

        E = len(exclusion_idxs)
        exclusion_idxs = np.array(exclusion_idxs, dtype=np.int32)
        scales = np.ones((E, 2), dtype=np.float64)
        # perturb the system
        for idx in exclusion_atoms:
            test_system[idx] = np.zeros(3) + np.random.rand() / 1000 + 2

        beta = 2.0
        cutoff = 1.0

        potential = generic.Nonbonded(N, exclusion_idxs, scales, beta, cutoff)

        for precision, rtol in [(np.float64, 1e-8), (np.float32, 1e-4)]:

            params = prepare_system_params(test_system, cutoff)

            self.compare_forces_gpu_vs_reference(test_system, [params], box, potential, rtol, precision=precision)

    def test_nonbonded(self):

        np.random.seed(4321)
        ff = Forcefield.load_from_file(DEFAULT_FF)

        _, all_coords, box, _ = builders.build_water_system(3.0, ff.water_ff)
        all_coords = all_coords / all_coords.unit
        for size in [33, 231, 1050]:

            coords = all_coords[:size]

            for cutoff in [1.0]:
                # E = 0 # DEBUG!
                charge_params, potential = prepare_water_system(coords, p_scale=5.0, cutoff=cutoff)
                for precision, rtol, atol in [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)]:

                    self.compare_forces_gpu_vs_reference(
                        coords,
                        gen_nonbonded_params_with_4d_offsets(np.random.default_rng(2022), charge_params, cutoff),
                        box,
                        potential,
                        rtol=rtol,
                        atol=atol,
                        precision=precision,
                    )

    def test_nonbonded_with_box_smaller_than_cutoff(self):

        np.random.seed(4321)

        precision = np.float32
        cutoff = 1
        size = 33
        padding = 0.1
        ff = Forcefield.load_from_file(DEFAULT_FF)

        _, coords, box, _ = builders.build_water_system(3.0, ff.water_ff)
        coords = coords / coords.unit
        coords = coords[:size]

        # Down shift box size to be only a portion of the cutoff
        charge_params, potential = prepare_water_system(coords, p_scale=1.0, cutoff=cutoff)

        def run_nonbonded(potential, x, box, params, steps=100):

            x = (x.astype(np.float32)).astype(np.float64)
            params = (params.astype(np.float32)).astype(np.float64)

            assert x.ndim == 2
            assert x.dtype == np.float64
            assert params.dtype == np.float64

            for _ in range(steps):
                _ = potential.execute_selective(x, params, box, True, True, True)

        test_impl = potential.to_gpu().unbound_impl(precision)

        # With the default box, all is well
        run_nonbonded(test_impl, coords, box, charge_params, steps=2)

        db_cutoff = (cutoff + padding) * 2

        # Make box with diagonals right at the limit
        box = np.eye(3) * db_cutoff
        run_nonbonded(test_impl, coords, box, charge_params)

        # Only populate the diag with values that are too low
        box = np.eye(3) * (db_cutoff * 0.3)
        with self.assertRaises(RuntimeError) as raised:
            run_nonbonded(test_impl, coords, box, charge_params)
        assert "more than half" in str(raised.exception)


if __name__ == "__main__":
    unittest.main()
