import numpy as np
import jax
from jax.config import config

config.update("jax_enable_x64", True)
import functools

from common import GradientTest
from timemachine.lib import potentials
from timemachine.potentials import bonded


class TestBonded(GradientTest):

    def test_centroid_restraint(self, n_particles=10, n_A=4, n_B=3, kb=5.4, b0=2.3):
        """Randomly define subsets A and B of a larger collection of particles,
        generate a centroid restraint between A and B, and then validate the resulting CentroidRestraint force"""
        box = np.eye(3) * 100

        # specific to centroid restraint force
        relative_tolerance_at_precision = {np.float32: 2e-5, np.float64: 1e-9}

        for precision, rtol in relative_tolerance_at_precision.items():
            x_primal = self.get_random_coords(n_particles, 3)

            gai = np.random.randint(0, n_particles, n_A, dtype=np.int32)
            gbi = np.random.randint(0, n_particles, n_B, dtype=np.int32)

            masses = np.random.rand(n_particles)

            ref_nrg = jax.partial(
                bonded.centroid_restraint,
                # masses=masses,
                group_a_idxs=gai,
                group_b_idxs=gbi,
                kb=kb,
                b0=b0
            )

            # we need to clear the du_dp buffer each time, so we need
            # to instantiate test_nrg inside here
            test_nrg = potentials.CentroidRestraint(
                gai,
                gbi,
                # masses,
                kb,
                b0
            )

            params = np.array([], dtype=np.float64)
            lamb = 0.3  # doesn't matter

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

    def test_centroid_restraint_singularity(self):
        # test singularity is stable when dij=0 and b0 = 0
        box = np.eye(3) * 100

        # specific to centroid restraint force
        relative_tolerance_at_precision = {np.float32: 2e-5, np.float64: 1e-9}

        n_particles = 5

        for precision, rtol in relative_tolerance_at_precision.items():
            x0 = self.get_random_coords(n_particles, 3)
            coords = np.concatenate([x0, x0])

            gai = np.arange(5).astype(np.int32)
            gbi = (np.arange(5) + 5).astype(np.int32)

            masses = np.random.rand(n_particles)

            kb = 10.0
            b0 = 0.0

            ref_nrg = jax.partial(
                bonded.centroid_restraint,
                group_a_idxs=gai,
                group_b_idxs=gbi,
                kb=kb,
                b0=b0
            )

            # we need to clear the du_dp buffer each time, so we need
            # to instantiate test_nrg inside here
            test_nrg = potentials.CentroidRestraint(
                gai,
                gbi,
                kb,
                b0
            )

            params = np.array([], dtype=np.float64)
            lamb = 0.3  # doesn't matter

            self.compare_forces(
                coords,
                params,
                box,
                lamb,
                ref_nrg,
                test_nrg,
                rtol,
                precision=precision
            )


    def test_harmonic_bond(self, n_particles=64, n_bonds=35, dim=3):
        """Randomly connect pairs of particles, then validate the resulting HarmonicBond force"""
        np.random.seed(125)  # TODO: where should this seed be set?

        x = self.get_random_coords(n_particles, dim)

        atom_idxs = np.arange(n_particles)
        params = np.random.rand(n_bonds, 2).astype(np.float64)

        bond_idxs = []
        for _ in range(n_bonds):
            bond_idxs.append(np.random.choice(atom_idxs, size=2, replace=False))
        bond_idxs = np.array(bond_idxs, dtype=np.int32)

        lamb = 0.0
        box = np.eye(3) * 100

        # specific to harmonic bond force
        relative_tolerance_at_precision = {np.float32: 2e-5, np.float64: 1e-9}

        for precision, rtol in relative_tolerance_at_precision.items():
            test_potential = potentials.HarmonicBond(bond_idxs)
            ref_potential = functools.partial(
                bonded.harmonic_bond,
                bond_idxs=bond_idxs
            )

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

        lamb_mult = np.random.randint(-5, 5, size=n_bonds, dtype=np.int32)
        lamb_offset = np.random.randint(-5, 5, size=n_bonds, dtype=np.int32)
        lamb = 0.35

        for precision, rtol in relative_tolerance_at_precision.items():
            test_potential = potentials.HarmonicBond(bond_idxs, lamb_mult, lamb_offset)
            ref_potential = functools.partial(
                bonded.harmonic_bond,
                bond_idxs=bond_idxs,
                lamb_mult=lamb_mult,
                lamb_offset=lamb_offset
            )

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


    def test_harmonic_bond_singularity(self):
        """ Test that two particles sitting directly on top of each other should generate a proper force. """
        x = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]], dtype=np.float64)

        params = np.array([[2.0, 0.0]], dtype=np.float64)
        bond_idxs = np.array([[0, 1]], dtype=np.int32)

        lamb = 0.0
        box = np.eye(3) * 100

        # specific to harmonic bond force
        relative_tolerance_at_precision = {np.float32: 2e-5, np.float64: 1e-9}

        for precision, rtol in relative_tolerance_at_precision.items():
            test_potential = potentials.HarmonicBond(bond_idxs)
            ref_potential = functools.partial(
                bonded.harmonic_bond,
                bond_idxs=bond_idxs
            )

            # we assert finite-ness of the forces.
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

        # test with both zero and non zero terms
        x = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]], dtype=np.float64)

        params = np.array([[2.0, 0.0], [2.0, 1.0]], dtype=np.float64)
        bond_idxs = np.array([[0, 1], [0, 2]], dtype=np.int32)

        # specific to harmonic bond force
        relative_tolerance_at_precision = {np.float32: 2e-5, np.float64: 1e-9}

        for precision, rtol in relative_tolerance_at_precision.items():
            test_potential = potentials.HarmonicBond(bond_idxs)
            ref_potential = functools.partial(
                bonded.harmonic_bond,
                bond_idxs=bond_idxs
            )

            # we assert finite-ness of the forces.
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

    def test_harmonic_angle(self, n_particles=64, n_angles=25, dim=3):
        """Randomly connect triples of particles, then validate the resulting HarmonicAngle force"""
        np.random.seed(125)

        x = self.get_random_coords(n_particles, dim)

        atom_idxs = np.arange(n_particles)
        params = np.random.rand(n_angles, 2).astype(np.float64)
        angle_idxs = []
        for _ in range(n_angles):
            angle_idxs.append(np.random.choice(atom_idxs, size=3, replace=False))
        angle_idxs = np.array(angle_idxs, dtype=np.int32)

        lamb = 0.0
        box = np.eye(3) * 100

        # specific to harmonic angle force
        relative_tolerance_at_precision = {np.float32: 2e-5, np.float64: 1e-9}

        for precision, rtol in relative_tolerance_at_precision.items():
            test_potential = potentials.HarmonicAngle(angle_idxs)
            ref_potential = functools.partial(bonded.harmonic_angle, angle_idxs=angle_idxs)

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


        lamb_mult = np.random.randint(-5, 5, size=n_angles, dtype=np.int32)
        lamb_offset = np.random.randint(-5, 5, size=n_angles, dtype=np.int32)
        lamb = 0.35

        for precision, rtol in relative_tolerance_at_precision.items():
            test_potential = potentials.HarmonicAngle(angle_idxs, lamb_mult, lamb_offset)
            ref_potential = functools.partial(
                bonded.harmonic_angle,
                angle_idxs=angle_idxs,
                lamb_mult=lamb_mult,
                lamb_offset=lamb_offset
            )

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


    def test_periodic_torsion(self, n_particles=64, n_torsions=25, dim=3):
        """Randomly connect quadruples of particles, then validate the resulting PeriodicTorsion force"""
        np.random.seed(125)

        x = self.get_random_coords(n_particles, dim)

        atom_idxs = np.arange(n_particles)
        params = np.random.rand(n_torsions, 3).astype(np.float64)
        torsion_idxs = []
        for _ in range(n_torsions):
            torsion_idxs.append(np.random.choice(atom_idxs, size=4, replace=False))

        torsion_idxs = np.array(torsion_idxs, dtype=np.int32)

        lamb = 0.0
        box = np.eye(3) * 100

        # specific to periodic torsion force
        relative_tolerance_at_precision = {np.float32: 2e-5, np.float64: 1e-9}

        for precision, rtol in relative_tolerance_at_precision.items():
            test_potential = potentials.PeriodicTorsion(torsion_idxs)
            ref_potential = functools.partial(bonded.periodic_torsion, torsion_idxs=torsion_idxs)

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

        lamb_mult = np.random.randint(-5, 5, size=n_torsions, dtype=np.int32)
        lamb_offset = np.random.randint(-5, 5, size=n_torsions, dtype=np.int32)
        lamb = 0.35

        for precision, rtol in relative_tolerance_at_precision.items():
            test_potential = potentials.PeriodicTorsion(torsion_idxs, lamb_mult, lamb_offset)
            ref_potential = functools.partial(
                bonded.periodic_torsion,
                torsion_idxs=torsion_idxs,
                lamb_mult=lamb_mult,
                lamb_offset=lamb_offset)

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