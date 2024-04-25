import unittest
from dataclasses import replace
from typing import cast

import numpy as np
import pytest
from common import GradientTest, gen_nonbonded_params_with_4d_offsets, prepare_system_params, prepare_water_system

from timemachine import potentials
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.md import builders
from timemachine.testsystems.dhfr import setup_dhfr

np.set_printoptions(linewidth=500)

pytestmark = [pytest.mark.memcheck]


class TestNonbondedDHFR(GradientTest):
    def setUp(self):
        # This test checks hilbert curve re-ordering gives identical results
        host_fns, _, host_coords, self.box = setup_dhfr()

        nonbonded_bp = next(bp for bp in host_fns if isinstance(bp.potential, potentials.Nonbonded))
        self.nonbonded_fn = cast(potentials.Nonbonded, nonbonded_bp.potential)
        self.nonbonded_params = nonbonded_bp.params
        self.host_conf = host_coords
        self.beta = self.nonbonded_fn.beta
        self.nonbonded_fn.cutoff = 1.2
        self.cutoff = self.nonbonded_fn.cutoff

    def test_nblist_hilbert(self):
        """
        This test makes sure that hilbert ordering has no impact on numerical results. The
        computed forces, energies, etc. should be bitwise identical.
        """

        np.random.seed(2021)

        N = self.host_conf.shape[0]

        for precision in [np.float32, np.float64]:
            ref_nonbonded_impl = replace(self.nonbonded_fn, disable_hilbert_sort=True).to_gpu(precision).unbound_impl
            test_nonbonded_impl = self.nonbonded_fn.to_gpu(precision).unbound_impl

            padding = self.nonbonded_fn.nblist_padding
            deltas = np.random.rand(N, 3) - 0.5  # [-0.5, +0.5]
            divisor = 0.5 * (2 * np.sqrt(3)) / padding
            # if deltas are kept under +- p/(2*sqrt(3)) then no rebuild gets triggered
            deltas = deltas / divisor  # exactly within bounds, and should not trigger a rebuild

            for d in deltas:
                assert np.linalg.norm(d) < padding / 2

            xs = [self.host_conf, self.host_conf + deltas]

            np.set_printoptions(precision=16)
            # under pure fixed point accumulation the results should be identical.
            for x in xs:
                ref_du_dx, ref_du_dp, ref_u = ref_nonbonded_impl.execute(x, self.nonbonded_params, self.box)
                test_du_dx, test_du_dp, test_u = test_nonbonded_impl.execute(x, self.nonbonded_params, self.box)

                np.testing.assert_array_equal(ref_du_dx, test_du_dx)
                np.testing.assert_array_equal(ref_du_dp, test_du_dp)
                np.testing.assert_array_equal(ref_u, test_u)

                ref_du_dx = ref_nonbonded_impl.execute_du_dx(x, self.nonbonded_params, self.box)
                test_du_dx = test_nonbonded_impl.execute_du_dx(x, self.nonbonded_params, self.box)

                for idx, (a, b) in enumerate(zip(ref_du_dx, test_du_dx)):
                    if np.linalg.norm(a - b) != 0:
                        print(idx, a, b)
                        assert 0
                    np.testing.assert_array_equal(a, b)

                np.testing.assert_array_equal(ref_du_dx, test_du_dx)

    def test_nblist_rebuild(self):
        """
        This test makes sure that periodically rebuilding the neighborlist has no impact on numerical results. The
        computed forces, energies, etc. should be bitwise identical.
        """

        N = self.host_conf.shape[0]

        np.random.seed(2021)

        ref_nonbonded_impl = replace(self.nonbonded_fn, nblist_padding=0.0).to_gpu(np.float64).unbound_impl

        padding = 0.1

        # rebuild only when deltas have moved more than padding/2 angstroms
        test_nonbonded_impl = replace(self.nonbonded_fn, nblist_padding=padding).to_gpu(np.float64).unbound_impl

        deltas = np.random.rand(N, 3) - 0.5  # [-0.5, +0.5]
        divisor = 0.5 * (2 * np.sqrt(3)) / padding
        # if deltas are kept under +- p/(2*sqrt(3)) then no rebuild gets triggered
        deltas = deltas / divisor  # exactly within bounds, and should not trigger a rebuild

        for d in deltas:
            assert np.linalg.norm(d) < padding / 2

        xs = [self.host_conf, self.host_conf + deltas]

        # under pure fixed point accumulation the results should be identical.
        for x in xs:
            ref_du_dx, ref_du_dp, ref_u = ref_nonbonded_impl.execute(x, self.nonbonded_params, self.box)
            test_du_dx, test_du_dp, test_u = test_nonbonded_impl.execute(x, self.nonbonded_params, self.box)

            np.testing.assert_array_equal(ref_du_dx, test_du_dx)
            np.testing.assert_array_equal(ref_du_dp, test_du_dp)
            np.testing.assert_array_equal(ref_u, test_u)

            ref_du_dx = ref_nonbonded_impl.execute_du_dx(x, self.nonbonded_params, self.box)
            test_du_dx = test_nonbonded_impl.execute_du_dx(x, self.nonbonded_params, self.box)

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
        for N in [33, 65, 231, 1050, 3080]:
            rng = np.random.default_rng(2022)

            test_conf = self.host_conf[:N]

            # strip out parts of the system
            test_exclusions = []
            test_scales = []
            for (i, j), (sa, sb) in zip(self.nonbonded_fn.exclusion_idxs, self.nonbonded_fn.scale_factors):
                if i < N and j < N:
                    test_exclusions.append((i, j))
                    test_scales.append((sa, sb))

            test_exclusions = np.array(test_exclusions, dtype=np.int32)
            test_scales = np.array(test_scales, dtype=np.float64)
            test_params = self.nonbonded_params[:N, :]

            for atom_idxs in [None, np.array(rng.choice(N, N // 2, replace=False), dtype=np.int32)]:
                potential = potentials.Nonbonded(
                    N, test_exclusions, test_scales, self.beta, self.cutoff, atom_idxs=atom_idxs
                )

                for precision, rtol, atol in [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)]:
                    self.compare_forces(
                        test_conf, test_params, self.box, potential, potential.to_gpu(precision), rtol=rtol, atol=atol
                    )


class TestNonbondedWater(GradientTest):
    def test_nblist_box_resize(self):
        # test that running the coordinates under two different boxes produces correct results
        # since we should be rebuilding the nblist when the box sizes change.
        ff = Forcefield.load_default()

        host_system, host_conf, box, _ = builders.build_water_system(3.0, ff.water_ff)

        host_fns, host_masses = openmm_deserializer.deserialize_system(host_system, cutoff=1.2)

        test_bp = next(bp for bp in host_fns if isinstance(bp.potential, potentials.Nonbonded))
        assert test_bp.params is not None

        big_box = box + np.eye(3) * 1000

        # (ytz): note the ordering should be from large box to small box. though in the current code
        # the rebuild is triggered as long as the box *changes*.
        for test_box in [big_box, box]:
            for precision, rtol, atol, du_dp_rtol, du_dp_atol in [
                (np.float64, 1e-8, 1e-10, 1e-6, 1e-9),
                (np.float32, 1e-4, 3e-5, 1e-4, 3e-5),
            ]:
                self.compare_forces(
                    host_conf,
                    test_bp.params,
                    test_box,
                    test_bp.potential,
                    test_bp.potential.to_gpu(precision),
                    rtol=rtol,
                    atol=atol,
                    du_dp_rtol=du_dp_rtol,
                    du_dp_atol=du_dp_atol,
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
        cutoff = 1.2

        potential = potentials.Nonbonded(N, exclusion_idxs, scales, beta, cutoff)

        for precision, rtol in [(np.float64, 1e-8), (np.float32, 1e-4)]:
            params = prepare_system_params(test_system, cutoff)

            self.compare_forces(test_system, params, box, potential, potential.to_gpu(precision), rtol)

    def test_nonbonded(self):
        np.random.seed(4321)
        ff = Forcefield.load_default()

        _, all_coords, box, _ = builders.build_water_system(3.0, ff.water_ff)
        for size in [33, 231, 1050]:
            coords = all_coords[:size]

            for cutoff in [1.2]:
                # E = 0 # DEBUG!
                charge_params, potential = prepare_water_system(coords, p_scale=5.0, cutoff=cutoff)
                for precision, rtol, atol in [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)]:
                    test_impl = potential.to_gpu(precision)
                    for params in gen_nonbonded_params_with_4d_offsets(
                        np.random.default_rng(2022), charge_params, cutoff
                    ):
                        self.compare_forces(coords, params, box, potential, test_impl, rtol=rtol, atol=atol)


if __name__ == "__main__":
    unittest.main()
