import jax.numpy as jnp
import numpy as np
import pytest
from common import GradientTest

from timemachine.potentials import (
    CentroidRestraint,
    ChiralAtomRestraint,
    ChiralBondRestraint,
    FlatBottomBond,
    HarmonicAngle,
    HarmonicBond,
    LogFlatBottomBond,
    PeriodicTorsion,
)

pytestmark = [pytest.mark.memcheck]

from timemachine.potentials.bonded import harmonic_angle


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

            # masses = np.random.rand(n_particles)

            # we need to clear the du_dp buffer each time, so we need
            # to instantiate test_nrg inside here
            potential = CentroidRestraint(
                gai,
                gbi,
                # masses,
                kb,
                b0,
            )

            params = np.array([], dtype=np.float64)

            test_impl = potential.to_gpu(precision)
            self.compare_forces(x_primal, params, box, potential, test_impl, rtol)
            self.assert_differentiable_interface_consistency(x_primal, params, box, test_impl)

    def test_centroid_restraint_singularity(self):
        # test singularity is stable when dij=0 and b0 = 0
        box = np.eye(3) * 100

        # specific to centroid restraint force
        relative_tolerance_at_precision = {np.float32: 2e-5, np.float64: 1e-9}

        n_particles = 5

        for precision, rtol in relative_tolerance_at_precision.items():
            x0 = self.get_random_coords(n_particles, 3)
            coords_0 = np.concatenate([x0, x0])
            coords_1 = self.get_random_coords(n_particles * 2, 3)

            for coords in [coords_0, coords_1]:
                gai = np.arange(5).astype(np.int32)
                gbi = (np.arange(5) + 5).astype(np.int32)

                kb = 10.0
                b0 = 0.0

                # we need to clear the du_dp buffer each time, so we need
                # to instantiate test_nrg inside here
                potential = CentroidRestraint(gai, gbi, kb, b0)

                params = np.array([], dtype=np.float64)

                self.compare_forces(coords, params, box, potential, potential.to_gpu(precision), rtol)

    def test_harmonic_bond(self, n_particles=64, n_bonds=35, dim=3):
        """Randomly connect pairs of particles, then validate the resulting HarmonicBond force"""
        np.random.seed(125)  # TODO: where should this seed be set?

        x = self.get_random_coords(n_particles, dim)

        atom_idxs = np.arange(n_particles)
        params = np.random.rand(n_bonds, 2).astype(np.float64)

        bond_idxs = []
        for _ in range(n_bonds):
            bond_idxs.append(np.random.choice(atom_idxs, size=2, replace=False))
        bond_idxs = np.array(bond_idxs, dtype=np.int32) if n_bonds else np.zeros((0, 2), dtype=np.int32)

        box = np.eye(3) * 100

        # specific to harmonic bond force
        relative_tolerance_at_precision = {np.float64: 1e-7, np.float32: 2e-5}

        potential = HarmonicBond(bond_idxs)
        for precision, rtol in relative_tolerance_at_precision.items():
            test_impl = potential.to_gpu(precision)
            self.compare_forces(x, params, box, potential, test_impl, rtol)
            self.assert_differentiable_interface_consistency(x, params, box, test_impl)

            # test bitwise commutativity
            test_potential_rev = HarmonicBond(bond_idxs[:, ::-1])

            test_potential_rev_impl = test_potential_rev.to_gpu(precision).unbound_impl

            test_du_dx, test_du_dp, test_u = test_impl.unbound_impl.execute(x, params, box, 1, 1, 1)

            test_du_dx_rev, test_du_dp_rev, test_u_rev = test_potential_rev_impl.execute(x, params, box, 1, 1, 1)

            np.testing.assert_array_equal(test_u, test_u_rev)
            np.testing.assert_array_equal(test_du_dx, test_du_dx_rev)
            np.testing.assert_array_equal(test_du_dp, test_du_dp_rev)

    def test_flat_bottom_bond(self, n_particles=64, n_bonds=35, dim=3):
        """Randomly connect pairs of particles, then validate the resulting FlatBottomBond force"""
        np.random.seed(2022)

        # TODO(deboggle) : reduce code duplication between HarmonicBond and FlatBottomBond
        box = np.eye(3) * 100
        x = self.get_random_coords(n_particles, dim)

        atom_idxs = np.arange(n_particles)

        k = np.random.rand(n_bonds) * 1000  # k large
        r_min = np.random.rand(n_bonds)  # r_min non-negative
        r_max = r_min + np.random.rand(n_bonds)  # r_max >= r_min
        params = np.array([k, r_min, r_max]).astype(np.float64).T
        assert params.shape == (n_bonds, 3)

        bond_idxs = []
        for _ in range(n_bonds):
            bond_idxs.append(np.random.choice(atom_idxs, size=2, replace=False))
        bond_idxs = np.array(bond_idxs, dtype=np.int32) if n_bonds else np.zeros((0, 2), dtype=np.int32)

        # Shift half of the bond indices by a single box dimension to ensure testing PBCs
        x[bond_idxs[:, 1][: n_bonds // 2]] += np.diagonal(box)

        relative_tolerance_at_precision = {np.float64: 1e-7, np.float32: 2e-5}

        potential = FlatBottomBond(bond_idxs)
        for precision, rtol in relative_tolerance_at_precision.items():
            test_impl = potential.to_gpu(precision)
            self.compare_forces(x, params, box, potential, test_impl, rtol)
            self.assert_differentiable_interface_consistency(x, params, box, test_impl)

            # test bitwise commutativity
            test_potential = FlatBottomBond(bond_idxs)
            test_potential_rev = FlatBottomBond(bond_idxs[:, ::-1])

            test_potential_impl = test_potential.to_gpu(precision).unbound_impl
            test_potential_rev_impl = test_potential_rev.to_gpu(precision).unbound_impl

            test_du_dx, test_du_dp, test_u = test_potential_impl.execute(x, params, box, 1, 1, 1)

            test_du_dx_rev, test_du_dp_rev, test_u_rev = test_potential_rev_impl.execute(x, params, box, 1, 1, 1)

            np.testing.assert_array_equal(test_u, test_u_rev)
            np.testing.assert_array_equal(test_du_dx, test_du_dx_rev)
            np.testing.assert_array_equal(test_du_dp, test_du_dp_rev)

    def test_log_flat_bottom_bond(self, n_particles=64, n_bonds=35, dim=3):
        """Randomly connect pairs of particles, then validate the resulting LogFlatBottomBond force"""
        np.random.seed(2022)

        # TODO(deboggle) : reduce code duplication between HarmonicBond, FlatBottomBond, and LogFlatBottomBond
        box = np.eye(3) * 100
        x = self.get_random_coords(n_particles, dim)

        atom_idxs = np.arange(n_particles)

        k = np.random.rand(n_bonds) * 1000  # k not too large to avoid exp(-inf)
        r_min = np.zeros(n_bonds)
        r_max = np.random.rand(n_bonds) * 0.1  # equivalent for the special case of local MD
        params = np.array([k, r_min, r_max]).astype(np.float64).T
        assert params.shape == (n_bonds, 3)

        bond_idxs = []
        for _ in range(n_bonds):
            bond_idxs.append(np.random.choice(atom_idxs, size=2, replace=False))

        bond_idxs = np.array(bond_idxs, dtype=np.int32) if n_bonds else np.zeros((0, 2), dtype=np.int32)

        # Shift half of the bond indices by a single box dimension to ensure testing PBCs
        x[bond_idxs[:, 1][: n_bonds // 2]] += np.diagonal(box)

        relative_tolerance_at_precision = {np.float64: 1e-7, np.float32: 2e-5}

        beta = np.random.rand()

        potential = LogFlatBottomBond(bond_idxs, beta)
        for precision, rtol in relative_tolerance_at_precision.items():
            test_impl = potential.to_gpu(precision)
            self.compare_forces(x, params, box, potential, test_impl, rtol)
            self.assert_differentiable_interface_consistency(x, params, box, test_impl)

            # test bitwise commutativity
            test_potential = LogFlatBottomBond(bond_idxs, beta)
            test_potential_rev = LogFlatBottomBond(bond_idxs[:, ::-1], beta)

            test_potential_impl = test_potential.to_gpu(precision).unbound_impl
            test_potential_rev_impl = test_potential_rev.to_gpu(precision).unbound_impl

            test_du_dx, test_du_dp, test_u = test_potential_impl.execute(x, params, box, 1, 1, 1)

            test_du_dx_rev, test_du_dp_rev, test_u_rev = test_potential_rev_impl.execute(x, params, box, 1, 1, 1)

            np.testing.assert_array_equal(test_u, test_u_rev)
            np.testing.assert_array_equal(test_du_dx, test_du_dx_rev)
            np.testing.assert_array_equal(test_du_dp, test_du_dp_rev)

    def test_harmonic_bond_singularity(self):
        """Test that two particles sitting directly on top of each other should generate a proper force."""
        x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)

        params = np.array([[2.0, 0.0]], dtype=np.float64)
        bond_idxs = np.array([[0, 1]], dtype=np.int32)

        box = np.eye(3) * 100

        # specific to harmonic bond force
        relative_tolerance_at_precision = {np.float32: 2e-5, np.float64: 1e-9}

        potential = HarmonicBond(bond_idxs)
        for precision, rtol in relative_tolerance_at_precision.items():
            # we assert finite-ness of the forces.
            self.compare_forces(x, params, box, potential, potential.to_gpu(precision), rtol)

        # test with both zero and non zero terms
        x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)

        params = np.array([[2.0, 0.0], [2.0, 1.0]], dtype=np.float64)
        bond_idxs = np.array([[0, 1], [0, 2]], dtype=np.int32)

        # specific to harmonic bond force
        relative_tolerance_at_precision = {np.float32: 2e-5, np.float64: 1e-9}

        potential = HarmonicBond(bond_idxs)
        for precision, rtol in relative_tolerance_at_precision.items():
            # we assert finite-ness of the forces.
            self.compare_forces(x, params, box, potential, potential.to_gpu(precision), rtol)

    def test_harmonic_angle_jax_impl(self, n_particles=64, n_angles=25, dim=3):
        # test that implementation of the harmonic angle using kahan's trick
        # is consistent with the simple, original implementation

        def original_harmonic_angle(conf, params, box, angle_idxs):
            ci = conf[angle_idxs[:, 0]]
            cj = conf[angle_idxs[:, 1]]
            ck = conf[angle_idxs[:, 2]]
            kas = params[:, 0]
            a0s = params[:, 1]
            rji = ci - cj
            rjk = ck - cj
            top = jnp.sum(jnp.multiply(rji, rjk), -1)
            bot = jnp.linalg.norm(rji, axis=-1) * jnp.linalg.norm(rjk, axis=-1)
            tb = top / bot
            angle = jnp.arccos(tb)
            energies = kas / 2 * jnp.power(angle - a0s, 2)
            return jnp.sum(energies, -1)  # reduce over all angles

        for _ in range(25):
            x = self.get_random_coords(n_particles, dim)
            atom_idxs = np.arange(n_particles)
            params = np.random.rand(n_angles, 3).astype(np.float64)
            params[:, -1] = 0
            angle_idxs = []
            for _ in range(n_angles):
                angle_idxs.append(np.random.choice(atom_idxs, size=3, replace=False))
            angle_idxs = np.array(angle_idxs, dtype=np.int32) if n_angles else np.zeros((0, 3), dtype=np.int32)
            box = np.eye(3) * 100
            test_u = harmonic_angle(x, params, box, angle_idxs)
            ref_u = original_harmonic_angle(x, params[:, :2], box, angle_idxs)
            np.testing.assert_almost_equal(test_u, ref_u)

    def test_harmonic_angle(self, n_particles=64, n_angles=25, dim=3):
        """Randomly connect triples of particles, then validate the resulting HarmonicAngle force"""
        np.random.seed(125)

        x = self.get_random_coords(n_particles, dim)

        atom_idxs = np.arange(n_particles)
        params = np.random.rand(n_angles, 3).astype(np.float64)
        params[:, 2] = 0  # zero out epsilons
        angle_idxs = []
        for _ in range(n_angles):
            angle_idxs.append(np.random.choice(atom_idxs, size=3, replace=False))
        angle_idxs = np.array(angle_idxs, dtype=np.int32) if n_angles else np.zeros((0, 3), dtype=np.int32)

        box = np.eye(3) * 100

        # specific to harmonic angle force
        relative_tolerance_at_precision = {np.float32: 2e-5, np.float64: 1e-9}

        potential = HarmonicAngle(angle_idxs)
        for precision, rtol in relative_tolerance_at_precision.items():
            self.compare_forces(x, params, box, potential, potential.to_gpu(precision), rtol)

            # (ytz): leave these tests here, we lost bitwise identical energies and forces
            # when the ordering of atoms [ijk] is swapped to [jki] during the refactor to use the Kahan trick.
            # even the angle computation itself is not guaranteed to be identical (let alone energies and forces).
            # this isn't a deal breaker, but was just a nice to have.

            test_potential = HarmonicAngle(angle_idxs)
            test_potential_rev = HarmonicAngle(angle_idxs[:, ::-1])

            test_potential_impl = test_potential.to_gpu(precision).unbound_impl
            test_potential_rev_impl = test_potential_rev.to_gpu(precision).unbound_impl

            test_du_dx, test_du_dp, test_u = test_potential_impl.execute(x, params, box, 1, 1, 1)

            test_du_dx_rev, test_du_dp_rev, test_u_rev = test_potential_rev_impl.execute(x, params, box, 1, 1, 1)

            np.testing.assert_array_equal(test_u, test_u_rev)
            np.testing.assert_array_equal(test_du_dp, test_du_dp_rev)
            np.testing.assert_array_equal(test_du_dx, test_du_dx_rev)

    def test_periodic_torsion(self, n_particles=64, n_torsions=25, dim=3):
        """Randomly connect quadruples of particles, then validate the resulting PeriodicTorsion force"""
        np.random.seed(125)

        x = self.get_random_coords(n_particles, dim)

        atom_idxs = np.arange(n_particles)
        params = np.random.rand(n_torsions, 3).astype(np.float64)
        torsion_idxs = []
        for _ in range(n_torsions):
            torsion_idxs.append(np.random.choice(atom_idxs, size=4, replace=False))

        torsion_idxs = np.array(torsion_idxs, dtype=np.int32) if n_torsions else np.zeros((0, 4), dtype=np.int32)

        box = np.eye(3) * 100

        # specific to periodic torsion force
        relative_tolerance_at_precision = {np.float32: 2e-5, np.float64: 1e-9}

        potential = PeriodicTorsion(torsion_idxs)
        for precision, rtol in relative_tolerance_at_precision.items():
            self.compare_forces(x, params, box, potential, potential.to_gpu(precision), rtol)

        potential = PeriodicTorsion(torsion_idxs)
        for precision, rtol in relative_tolerance_at_precision.items():
            self.compare_forces(x, params, box, potential, potential.to_gpu(precision), rtol)

            # test bitwise commutativity
            test_potential = PeriodicTorsion(torsion_idxs)
            test_potential_rev = PeriodicTorsion(torsion_idxs[:, ::-1])

            test_potential_impl = test_potential.to_gpu(precision).unbound_impl
            test_potential_rev_impl = test_potential_rev.to_gpu(precision).unbound_impl

            test_du_dx, test_du_dp, test_u = test_potential_impl.execute(x, params, box, 1, 1, 1)

            test_du_dx_rev, test_du_dp_rev, test_u_rev = test_potential_rev_impl.execute(x, params, box, 1, 1, 1)

            np.testing.assert_array_equal(test_u, test_u_rev)
            np.testing.assert_array_equal(test_du_dx, test_du_dx_rev)
            np.testing.assert_array_equal(test_du_dp, test_du_dp_rev)

    def test_empty_potentials(self):
        # Check that no error is given if the terms are empty
        self.test_periodic_torsion(n_torsions=0)
        self.test_harmonic_angle(n_angles=0)
        self.test_harmonic_bond(n_bonds=0)

    def test_chiral_atom_restraint(self, n_particles=64, n_restraints=35, dim=3):
        """Randomly connect 4 particles, then validate the resulting forces"""
        np.random.seed(125)  # TODO: where should this seed be set?

        x = self.get_random_coords(n_particles, dim)
        params = np.random.rand(n_restraints).astype(np.float64)

        restr_idxs = []
        for _ in range(n_restraints):
            restr_idxs.append(np.random.choice(n_particles, size=4, replace=False))

        restr_idxs = np.array(restr_idxs, dtype=np.int32) if n_restraints else np.zeros((0, 4), dtype=np.int32)

        box = np.eye(3) * 100

        relative_tolerance_at_precision = {np.float64: 1e-9, np.float32: 1e-6}

        potential = ChiralAtomRestraint(restr_idxs)
        for precision, rtol in relative_tolerance_at_precision.items():
            self.compare_forces(x, params, box, potential, potential.to_gpu(precision), rtol)

            with self.assertRaises(Exception):
                # wrong length
                bad_idxs = np.array([[0, 1, 2, 3, 4], [4, 4, 3, 2, 1]])
                bad_potential = ChiralAtomRestraint(bad_idxs)
                bad_potential.to_gpu(precision)

        # test some bad potentials

    def test_chiral_bond_restraint(self, n_particles=64, n_restraints=35, dim=3):
        """Randomly connect 4 particles, then validate the resulting forces. Also test
        that scrambling the sign is sufficient"""
        np.random.seed(125)  # TODO: where should this seed be set?

        x = self.get_random_coords(n_particles, dim)

        params = np.random.rand(n_restraints).astype(np.float64)

        restr_idxs = []
        signs = []
        for _ in range(n_restraints):
            restr_idxs.append(np.random.choice(n_particles, size=4, replace=False))
            signs.append(np.random.choice(a=[-1, 1]))
        restr_idxs = np.array(restr_idxs, dtype=np.int32) if n_restraints else np.zeros((0, 4), dtype=np.int32)
        signs = np.array(signs, dtype=np.int32)

        box = np.eye(3) * 100

        relative_tolerance_at_precision = {np.float64: 1e-9, np.float32: 1e-6}

        potential = ChiralBondRestraint(restr_idxs, signs)
        for precision, rtol in relative_tolerance_at_precision.items():
            self.compare_forces(x, params, box, potential, potential.to_gpu(precision), rtol)

            with self.assertRaises(RuntimeError):
                # wrong length idxs
                bad_idxs = np.array([[0, 1, 2, 3, 4], [4, 4, 3, 2, 1]], dtype=np.int32)
                bad_signs = np.array([1, -1], dtype=np.int32)
                bad_potential = ChiralBondRestraint(bad_idxs, bad_signs)
                bad_potential.to_gpu(precision).unbound_impl

            with self.assertRaises(RuntimeError):
                # inconsistent lengths between idxs and signs
                bad_idxs = np.array([[0, 1, 2, 3], [4, 5, 3, 2]], dtype=np.int32)
                bad_signs = np.array([1, -1, 1], dtype=np.int32)
                bad_potential = ChiralBondRestraint(bad_idxs, bad_signs)
                bad_potential.to_gpu(precision).unbound_impl
