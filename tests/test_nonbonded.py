# (ytz): check test and run benchmark with pytest:
# pytest -xsv tests/test_nonbonded.py::TestNonbonded::test_dhfr && nvprof pytest -xsv tests/test_nonbonded.py::TestNonbonded::test_benchmark

import copy

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

from training import water_box

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


class TestNonbonded(GradientTest):

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


    def test_dhfr(self):

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
        box = np.asarray(box/box.unit)

        host_fns, host_masses = openmm_deserializer.deserialize_system(
            host_system,
            cutoff=1.0
        )

        for f in host_fns:
            if isinstance(f, potentials.Nonbonded):
                nonbonded_fn = f

        host_conf = []
        for x,y,z in host_coords:
            host_conf.append([to_md_units(x),to_md_units(y),to_md_units(z)])
        host_conf = np.array(host_conf)

        beta = 2.0
        cutoff = 1.1
        lamb = 0.1

        max_N = host_conf.shape[0]

        for N in [33, 65, 231, 1050, 4080]:

            print("N", N)

            test_conf = host_conf[:N]

           # process exclusions
            test_exclusions = []
            test_scales = []
            for (i, j), (sa, sb) in zip(nonbonded_fn.get_exclusion_idxs(), nonbonded_fn.get_scale_factors()):
                if i < N and j < N:
                    test_exclusions.append((i,j))
                    test_scales.append((sa, sb))
            test_exclusions = np.array(test_exclusions, dtype=np.int32)
            test_scales = np.array(test_scales, dtype=np.float64)
            test_params = nonbonded_fn.params[:N, :]

            test_lambda_plane_idxs = np.random.randint(low=-2, high=2, size=N, dtype=np.int32)
            test_lambda_offset_idxs = np.random.randint(low=-2, high=2, size=N, dtype=np.int32)

            test_nonbonded_fn = potentials.Nonbonded(
                test_exclusions,
                test_scales,
                test_lambda_plane_idxs,
                test_lambda_offset_idxs,
                beta,
                cutoff
            )

            ref_nonbonded_fn = prepare_reference_nonbonded(
                test_params,
                test_exclusions,
                test_scales,
                test_lambda_plane_idxs,
                test_lambda_offset_idxs,
                beta,
                cutoff
            )

            for precision, rtol in [(np.float64, 1e-8), (np.float32, 1e-4)]:

                self.compare_forces(
                    test_conf,
                    test_params,
                    box,
                    lamb,
                    ref_nonbonded_fn,
                    test_nonbonded_fn,
                    rtol,
                    precision=precision
                )

    # @unittest.skip("slow")
    def test_benchmark(self):

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
        box = np.asarray(box/box.unit)

        host_fns, host_masses = openmm_deserializer.deserialize_system(
            host_system,
            cutoff=1.0
        )

        for f in host_fns:
            if isinstance(f, potentials.Nonbonded):
                nonbonded_fn = f

        host_conf = []
        for x,y,z in host_coords:
            host_conf.append([to_md_units(x),to_md_units(y),to_md_units(z)])
        host_conf = np.array(host_conf)

        beta = 2.0
        cutoff = 1.1
        lamb = 0.0

        N = host_conf.shape[0]

        test_conf = host_conf[:N]

       # process exclusions
        test_exclusions = []
        test_scales = []
        for (i, j), (sa, sb) in zip(nonbonded_fn.get_exclusion_idxs(), nonbonded_fn.get_scale_factors()):
            if i < N and j < N:
                test_exclusions.append((i,j))
                test_scales.append((sa, sb))
        test_exclusions = np.array(test_exclusions, dtype=np.int32)
        test_scales = np.array(test_scales, dtype=np.float64)
        test_params = nonbonded_fn.params[:N, :]

        test_lambda_plane_idxs = np.zeros(N, dtype=np.int32)
        test_lambda_offset_idxs = np.zeros(N, dtype=np.int32)

        test_nonbonded_fn = potentials.Nonbonded(
            test_exclusions,
            test_scales,
            test_lambda_plane_idxs,
            test_lambda_offset_idxs,
            beta,
            cutoff
        )

        precision = np.float32

        impl = test_nonbonded_fn.unbound_impl(precision)

        for _ in range(100):

            impl.execute_du_dx(test_conf, test_params, box, lamb)

    def test_nonbonded(self):

        np.random.seed(4321)
        D = 3

        benchmark = False

        # test numerical accuracy on a box of water

        for size in [33, 231, 1050]:

            if not benchmark:
                water_coords = self.get_water_coords(D, sort=False)
                test_system = water_coords[:size]
                padding = 0.2
                diag = np.amax(test_system, axis=0) - np.amin(test_system, axis=0) + padding
                box = np.eye(3)
                np.fill_diagonal(box, diag)
            else:
                # _, test_system, box, _ = water_box.prep_system(8.1) # 8.1 is 50k atoms, roughly DHFR
                _, test_system, box, _ = water_box.prep_system(6.2) # 6.2 is 23k atoms, roughly DHFR
                test_system = test_system/test_system.unit

            for coords in [test_system]:

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
                                precision=precision,
                                benchmark=benchmark
                            )


if __name__ == "__main__":
    unittest.main()
