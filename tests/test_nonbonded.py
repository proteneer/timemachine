# (ytz): check test and run benchmark with pytest:
# pytest -xsv tests/test_nonbonded.py::TestNonbonded::test_dhfr && nvprof pytest -xsv tests/test_nonbonded.py::TestNonbonded::test_benchmark

import copy
import gzip

import pickle
import functools
import unittest
import scipy.linalg
from jax.config import config; config.update("jax_enable_x64", True)

import numpy as np
import jax
import jax.numpy as jnp

import functools

from common import GradientTest
from common import prepare_nb_system, prepare_water_system, prepare_reference_nonbonded

from timemachine.potentials import bonded, nonbonded, gbsa
from timemachine.lib import potentials
from md import builders

from hilbertcurve.hilbertcurve import HilbertCurve
from fe.utils import to_md_units

from ff.handlers import openmm_deserializer
from simtk.openmm import app

np.set_printoptions(linewidth=500)


def hilbert_sort(conf, D):
    hc = HilbertCurve(64, D)
    int_confs = (conf*1000).astype(np.int64)
    dists = []
    for xyz in int_confs.tolist():
        dist = hc.distance_from_coordinates(xyz)
        dists.append(dist)
    perm = np.argsort(dists)
    return perm

class TestNonbondedDHFR(GradientTest):

    def setUp(self):
        # This test checks hilbert curve re-ordering gives identical results
        pdb_path = 'tests/data/5dfr_solv_equil.pdb'
        host_pdb = app.PDBFile(pdb_path)
        protein_ff = app.ForceField('amber99sbildn.xml', 'tip3p.xml')

        host_system = protein_ff.createSystem(
            host_pdb.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=None,
            rigidWater=False
        )

        host_coords = host_pdb.positions
        box = host_pdb.topology.getPeriodicBoxVectors()
        self.box = np.asarray(box/box.unit)

        host_fns, host_masses = openmm_deserializer.deserialize_system(
            host_system,
            cutoff=1.0
        )

        for f in host_fns:
            if isinstance(f, potentials.Nonbonded):
                self.nonbonded_fn = f

        host_conf = []
        for x,y,z in host_coords:
            host_conf.append([to_md_units(x),to_md_units(y),to_md_units(z)])
        self.host_conf = np.array(host_conf)

        self.beta = 2.0
        self.cutoff = 1.1
        self.lamb = 0.1

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
            deltas = np.random.rand(N,3)-0.5 # [-0.5, +0.5]
            divisor = 0.5*(2*np.sqrt(3))/padding
            # if deltas are kept under +- p/(2*sqrt(3)) then no rebuild gets triggered
            deltas = deltas/divisor # exactly within bounds, and should not trigger a rebuild

            for d in deltas:
                assert np.linalg.norm(d) < padding/2

            xs = [self.host_conf, self.host_conf + deltas]

            np.set_printoptions(precision=16)
            # under pure fixed point accumulation the results should be identical.
            for x_idx, x in enumerate(xs):

                ref_du_dx, ref_du_dp, ref_du_dl, ref_u = ref_nonbonded_impl.execute(x, self.nonbonded_fn.params, self.box, 0.0)
                test_du_dx, test_du_dp, test_du_dl, test_u = test_nonbonded_impl.execute(x, self.nonbonded_fn.params, self.box, 0.0)

                np.testing.assert_array_equal(ref_du_dx, test_du_dx)
                np.testing.assert_array_equal(ref_du_dp, test_du_dp)
                np.testing.assert_array_equal(ref_du_dl, test_du_dl)
                np.testing.assert_array_equal(ref_u, test_u)

                ref_du_dx = ref_nonbonded_impl.execute_du_dx(x, self.nonbonded_fn.params, self.box, 0.0)
                test_du_dx = test_nonbonded_impl.execute_du_dx(x, self.nonbonded_fn.params, self.box, 0.0)

                for idx, (a, b) in enumerate(zip(ref_du_dx, test_du_dx)):
                    if np.linalg.norm(a-b) != 0:
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
        ref_nonbonded_impl.set_nblist_padding(0.0) # rebuild with every call

        padding = 0.1

        test_nonbonded_impl = copy.copy(self.nonbonded_fn).unbound_impl(np.float64)
        test_nonbonded_impl.set_nblist_padding(padding) # rebuild only when deltas have moved more than padding/2 angstroms

        deltas = np.random.rand(N,3)-0.5 # [-0.5, +0.5]
        divisor = 0.5*(2*np.sqrt(3))/padding
        # if deltas are kept under +- p/(2*sqrt(3)) then no rebuild gets triggered
        deltas = deltas/divisor # exactly within bounds, and should not trigger a rebuild

        for d in deltas:
            assert np.linalg.norm(d) < padding/2

        xs = [self.host_conf, self.host_conf + deltas]

        # under pure fixed point accumulation the results should be identical.
        for x_idx, x in enumerate(xs):
            ref_du_dx, ref_du_dp, ref_du_dl, ref_u = ref_nonbonded_impl.execute(x, self.nonbonded_fn.params, self.box, 0.0)
            test_du_dx, test_du_dp, test_du_dl, test_u = test_nonbonded_impl.execute(x, self.nonbonded_fn.params, self.box, 0.0)

            np.testing.assert_array_equal(ref_du_dx, test_du_dx)
            np.testing.assert_array_equal(ref_du_dp, test_du_dp)
            np.testing.assert_array_equal(ref_du_dl, test_du_dl)
            np.testing.assert_array_equal(ref_u, test_u)

            ref_du_dx = ref_nonbonded_impl.execute_du_dx(x, self.nonbonded_fn.params, self.box, 0.0)
            test_du_dx = test_nonbonded_impl.execute_du_dx(x, self.nonbonded_fn.params, self.box, 0.0)

            for idx, (a, b) in enumerate(zip(ref_du_dx, test_du_dx)):
                if np.linalg.norm(a-b) != 0:
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
        max_N = self.host_conf.shape[0]

        # we can't go bigger than this due to memory limitations in the the reference platform.
        for N in [33, 65, 231, 1050, 4080]:

            test_conf = self.host_conf[:N]

            # strip out parts of the system
            test_exclusions = []
            test_scales = []
            for (i, j), (sa, sb) in zip(self.nonbonded_fn.get_exclusion_idxs(), self.nonbonded_fn.get_scale_factors()):
                if i < N and j < N:
                    test_exclusions.append((i,j))
                    test_scales.append((sa, sb))
            test_exclusions = np.array(test_exclusions, dtype=np.int32)
            test_scales = np.array(test_scales, dtype=np.float64)
            test_params = self.nonbonded_fn.params[:N, :]

            test_lambda_plane_idxs = np.random.randint(low=-2, high=2, size=N, dtype=np.int32)
            test_lambda_offset_idxs = np.random.randint(low=-2, high=2, size=N, dtype=np.int32)

            test_nonbonded_fn = potentials.Nonbonded(
                test_exclusions,
                test_scales,
                test_lambda_plane_idxs,
                test_lambda_offset_idxs,
                self.beta,
                self.cutoff
            )

            ref_nonbonded_fn = prepare_reference_nonbonded(
                test_params,
                test_exclusions,
                test_scales,
                test_lambda_plane_idxs,
                test_lambda_offset_idxs,
                self.beta,
                self.cutoff
            )

            for precision, rtol in [(np.float64, 1e-8), (np.float32, 1e-4)]:

                self.compare_forces(
                    test_conf,
                    test_params,
                    self.box,
                    self.lamb,
                    ref_nonbonded_fn,
                    test_nonbonded_fn,
                    rtol,
                    precision=precision
                )

    @unittest.skip("benchmark-only")
    def test_benchmark(self):
        """
        This is mainly for benchmarking nonbonded computations on the initial state.
        """

        N = self.host_conf.shape[0]

        host_conf = self.host_conf[:N]

        precision = np.float32

        nb_fn = copy.deepcopy(self.nonbonded_fn)

        test_lambda_plane_idxs = np.random.randint(low=-2, high=2, size=N, dtype=np.int32)
        test_lambda_offset_idxs = np.random.randint(low=-2, high=2, size=N, dtype=np.int32)

        nb_fn.set_lambda_plane_idxs(test_lambda_plane_idxs)
        nb_fn.set_lambda_offset_idxs(test_lambda_offset_idxs)

        impl = nb_fn.unbound_impl(np.float32)

        for combo in itertools.product([False, True], repeat=4):

            (compute_du_dx, compute_du_dp, compute_du_dl, compute_u) = combo

            for trip in range(50):

                test_du_dx, test_du_dp, test_du_dl, test_u = impl.execute_selective(
                    self.host_conf,
                    self.nonbonded_fn.params,
                    self.box,
                    self.lamb,
                    compute_du_dx,
                    compute_du_dp,
                    compute_du_dl,
                    compute_u
                )


class TestNonbondedWater(GradientTest):

    def test_nblist_box_resize(self):
        # test that running the coordinates under two different boxes produces correct results
        # since we should be rebuilding the nblist when the box sizes change.

        host_system, host_coords, box, _ = builders.build_water_system(2.0)

        host_fns, host_masses = openmm_deserializer.deserialize_system(
            host_system,
            cutoff=1.0
        )

        for f in host_fns:
            if isinstance(f, potentials.Nonbonded):
                test_nonbonded_fn = f

        host_conf = []
        for x,y,z in host_coords:
            host_conf.append([to_md_units(x),to_md_units(y),to_md_units(z)])
        host_conf = np.array(host_conf)

        lamb = 0.1

        N = host_conf.shape[0]

        ref_nonbonded_fn = prepare_reference_nonbonded(
            test_nonbonded_fn.params,
            test_nonbonded_fn.get_exclusion_idxs(),
            test_nonbonded_fn.get_scale_factors(),
            test_nonbonded_fn.get_lambda_plane_idxs(),
            test_nonbonded_fn.get_lambda_offset_idxs(),
            test_nonbonded_fn.get_beta(),
            test_nonbonded_fn.get_cutoff()
        )

        big_box = box + np.eye(3)*1000

        # print(big_box, small_box)
        # (ytz): note the ordering should be from large box to small box. though in the current code
        # the rebuild is triggered as long as the box *changes*.
        for test_box in [big_box, box]:

            for precision, rtol in [(np.float64, 1e-8), (np.float32, 1e-4)]:

                self.compare_forces(
                    host_conf,
                    test_nonbonded_fn.params,
                    test_box,
                    lamb,
                    ref_nonbonded_fn,
                    test_nonbonded_fn,
                    rtol,
                    precision=precision
                )



class TestNonbonded(GradientTest):

    def test_fma_compiler_bug(self):

        # this test case deals with a rather annoying fma compiler bug in CUDA.
        # see https://github.com/proteneer/timemachine/issues/386
        fp=gzip.open('tests/repro.pkl.gz','rb') # This assumes that primes.data is already packed with gzip
        x_t, box, lamb, nb_bp = pickle.load(fp)

        for precision in [np.float32, np.float64]:

            impl = nb_bp.unbound_impl(precision)
            du_dx, du_dp, du_dl, u = impl.execute(x_t, nb_bp.params, box, lamb)

            uimpl2 = nb_bp.unbound_impl(precision)

            uimpl2.disable_hilbert_sort()
            du_dx2, du_dp2, du_dl2, u2 = uimpl2.execute(x_t, nb_bp.params, box, lamb)

            np.testing.assert_array_equal(u2, u)
            np.testing.assert_array_equal(du_dx2, du_dx)
            np.testing.assert_array_equal(du_dp2, du_dp)
            np.testing.assert_array_equal(du_dl2, du_dl) # this one fails without the patch.

    def test_exclusion(self):

        # This test verifies behavior when two particles are arbitrarily
        # close but are marked as excluded to ensure proper cancellation
        # of exclusions occur in the fixed point math.

        np.random.seed(2020)

        # water_coords = self.get_water_coords(3, sort=False)
        water_coords = self.get_water_coords(3, sort=False)
        test_system = water_coords[:126] # multiple of 3
        padding = 0.2
        diag = np.amax(test_system, axis=0) - np.amin(test_system, axis=0) + padding
        box = np.eye(3)
        np.fill_diagonal(box, diag)

        N = test_system.shape[0]

        EA = 10

        atom_idxs = np.arange(test_system.shape[0])

        # pick a set of atoms that will be mutually excluded from each other.
        # we will need to set their exclusions manually
        exclusion_atoms = np.random.choice(atom_idxs, size=(EA), replace=False)
        exclusion_idxs = []

        for idx, i in enumerate(exclusion_atoms):
            for jdx, j in enumerate(exclusion_atoms):
                if jdx > idx:
                    exclusion_idxs.append((i,j))

        E = len(exclusion_idxs)
        exclusion_idxs = np.array(exclusion_idxs, dtype=np.int32)
        scales = np.ones((E, 2), dtype=np.float64)
        # perturb the system
        for idx in exclusion_atoms:
            test_system[idx] = np.zeros(3) + np.random.rand()/1000+2

        beta = 2.0

        lambda_plane_idxs = np.random.randint(low=0, high=2, size=N, dtype=np.int32)
        lambda_offset_idxs = np.random.randint(low=0, high=2, size=N, dtype=np.int32)

        cutoff = 1.0

        for precision, rtol in [(np.float64, 1e-8), (np.float32, 1e-4)]:

            test_u = potentials.Nonbonded(
                exclusion_idxs,
                scales,
                lambda_plane_idxs,
                lambda_offset_idxs,
                beta,
                cutoff
            )

            charge_rescale_mask = np.ones((N, N))
            for (i,j), exc in zip(exclusion_idxs, scales[:, 0]):
                charge_rescale_mask[i][j] = 1 - exc
                charge_rescale_mask[j][i] = 1 - exc

            lj_rescale_mask = np.ones((N, N))
            for (i,j), exc in zip(exclusion_idxs, scales[:, 1]):
                lj_rescale_mask[i][j] = 1 - exc
                lj_rescale_mask[j][i] = 1 - exc

            ref_u = functools.partial(
                nonbonded.nonbonded_v3,
                charge_rescale_mask=charge_rescale_mask,
                lj_rescale_mask=lj_rescale_mask,
                scales=scales,
                beta=beta,
                cutoff=cutoff,
                lambda_plane_idxs=lambda_plane_idxs,
                lambda_offset_idxs=lambda_offset_idxs
            )

            lamb = 0.0

            params = np.stack([
                (np.random.rand(N).astype(np.float64) - 0.5)*np.sqrt(138.935456), # q
                np.random.rand(N).astype(np.float64)/10.0, # sig
                np.random.rand(N).astype(np.float64) # eps
            ], axis=1)

            self.compare_forces(
                test_system,
                params,
                box,
                lamb,
                ref_u,
                test_u,
                rtol,
                precision=precision
            )

    def test_nonbonded(self):

        np.random.seed(4321)
        D = 3

        for size in [33, 231, 1050]:

            _, coords, box, _ = builders.build_water_system(6.2)
            coords = coords/coords.unit
            coords = coords[:size]

            N = coords.shape[0]

            lambda_plane_idxs = np.random.randint(low=-2, high=2, size=N, dtype=np.int32)
            lambda_offset_idxs = np.random.randint(low=-2, high=2, size=N, dtype=np.int32)

            for precision, rtol in [(np.float64, 1e-8), (np.float32, 1e-4)]:

                for cutoff in [1.0]:
                    # E = 0 # DEBUG!
                    charge_params, ref_potential, test_potential = prepare_water_system(
                        coords,
                        lambda_plane_idxs,
                        lambda_offset_idxs,
                        p_scale=1.0,
                        cutoff=cutoff
                    )

                    for lamb in [0.0, 0.1, 0.2]:

                        print("lambda", lamb, "cutoff", cutoff, "precision", precision, "xshape", coords.shape)

                        self.compare_forces(
                            coords,
                            charge_params,
                            box,
                            lamb,
                            ref_potential,
                            test_potential,
                            rtol,
                            precision=precision
                        )


if __name__ == "__main__":
    unittest.main()
