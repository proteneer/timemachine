import numpy as np
import jax
import jax.numpy as jnp
from jax.config import config; config.update("jax_enable_x64", True)
import functools

from common import GradientTest
from timemachine.lib import potentials
from timemachine.potentials import bonded


class TestBonded(GradientTest):

    def test_centroid_restraint(self):

        N = 10

        for precision, rtol in [(np.float64, 1e-9), (np.float32, 2e-5)]:

            x_primal = self.get_random_coords(N, 3)

            gai = np.random.randint(0, N, 4, dtype=np.int32)
            gbi = np.random.randint(0, N, 3, dtype=np.int32)

            kb = 5.4
            b0 = 2.3

            masses = np.random.rand(N)

            ref_nrg = jax.partial(
                bonded.centroid_restraint,
                masses=masses,
                group_a_idxs=gai,
                group_b_idxs=gbi,
                kb=kb,
                b0=b0
            )

            box = np.eye(3)*100

            # we need to clear the du_dp buffer each time, so we need
            # to instantiate test_nrg inside here
            test_nrg = potentials.CentroidRestraint(
                gai,
                gbi,
                masses,
                kb,
                b0
            )

            params = np.array([], dtype=np.float64)
            lamb = 0.3 # doesn't matter

            self.compare_forces(
                x_primal,
                params,
                box,
                lamb,
                ref_nrg,
                test_nrg,
                rtol,
                precision=precision
            )


    def test_harmonic_bond(self):
        np.random.seed(125)

        N = 64
        B = 35
        D = 3

        x = self.get_random_coords(N, D)

        atom_idxs = np.arange(N)
        params = np.random.rand(B, 2).astype(np.float64)
        bond_idxs = []
        for _ in range(B):
            bond_idxs.append(np.random.choice(atom_idxs, size=2, replace=False))
        bond_idxs = np.array(bond_idxs, dtype=np.int32)

        lamb = 0.0

        for precision, rtol in [(np.float32, 4e-5), (np.float64, 1e-9)]:
            test_potential = potentials.HarmonicBond(bond_idxs)
            ref_potential = functools.partial(
                bonded.harmonic_bond,
                bond_idxs=bond_idxs
            )

            box = np.eye(3)*100

            self.compare_forces(
                x,
                params,
                box,
                lamb,
                ref_potential,
                test_potential,
                rtol,
                precision=precision
            )

    def test_harmonic_angle(self):
        np.random.seed(125)

        N = 64
        A = 25
        D = 3

        x = self.get_random_coords(N, D)

        atom_idxs = np.arange(N)
        params = np.random.rand(A, 2).astype(np.float64)
        angle_idxs = []
        for _ in range(A):
            angle_idxs.append(np.random.choice(atom_idxs, size=3, replace=False))
        angle_idxs = np.array(angle_idxs, dtype=np.int32)

        lamb = 0.0

        for precision, rtol in [(np.float64, 1e-9), (np.float32, 2e-5)]:
            # print(precision, rtol)
            test_potential = potentials.HarmonicAngle(angle_idxs)
            ref_potential = functools.partial(bonded.harmonic_angle, angle_idxs=angle_idxs)

            box = np.eye(3)*100

            self.compare_forces(
                x,
                params,
                box,
                lamb,
                ref_potential,
                test_potential,
                rtol,
                precision=precision
            )


    def test_periodic_torsion(self):
        np.random.seed(125)

        N = 64
        T = 25
        D = 3

        x = self.get_random_coords(N, D)

        atom_idxs = np.arange(N)
        params = np.random.rand(T, 3).astype(np.float64)
        torsion_idxs = []
        for _ in range(T):
            torsion_idxs.append(np.random.choice(atom_idxs, size=4, replace=False))

        torsion_idxs = np.array(torsion_idxs, dtype=np.int32)

        lamb = 0.0

        for precision, rtol in [(np.float32, 2e-5), (np.float64, 1e-9)]:

            test_potential = potentials.PeriodicTorsion(torsion_idxs)

            # test the parameter derivatives for correctness.
            ref_potential = functools.partial(bonded.periodic_torsion, torsion_idxs=torsion_idxs)

            box = np.eye(3)*100

            self.compare_forces(
                x,
                params,
                box,
                lamb,
                ref_potential,
                test_potential,
                rtol,
                precision=precision
            )
