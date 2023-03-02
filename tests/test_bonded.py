import functools

import numpy as np
import pytest
from common import GradientTest
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import AllChem

from timemachine.constants import DEFAULT_TEMP
from timemachine.ff import Forcefield
from timemachine.lib import potentials
from timemachine.md.enhanced import generate_ligand_samples
from timemachine.potentials import generic, rmsd


def analyze_angles(mol: Chem.rdchem.Mol, angle_idxs: NDArray, frames: NDArray):
    angles = []
    for ifr, frame in enumerate(frames):
        ci = frame[angle_idxs[:, 0]]
        cj = frame[angle_idxs[:, 1]]
        ck = frame[angle_idxs[:, 2]]

        vij = ci - cj
        vjk = ck - cj

        top = np.sum(np.multiply(vij, vjk), -1)
        bot = np.linalg.norm(vij, axis=-1) * np.linalg.norm(vjk, axis=-1)

        tb = top / bot
        # clip for numerical stability
        tb = np.clip(tb, a_min=-1, a_max=1)
        angle = np.rad2deg(np.arccos(tb))
        angles.append(angle)

    angles = np.array(angles).T
    return angles


def test_refit_angle_k():
    # propyne
    mol = Chem.AddHs(Chem.MolFromSmiles("[H]C#CC([H])([H])[H]"))
    AllChem.EmbedMolecule(mol)
    n_samples = 10000
    seed = 2023

    # Don't use DEFAULT_FF here since we'll change it
    orig_ff_name = "smirnoff_2_0_0_ccc.py"
    new_ff_name = "smirnoff_2_0_0_ccc_cos_angle.py"
    ff = Forcefield.load_from_file(new_ff_name)
    coords = generate_ligand_samples(n_samples, mol, ff, DEFAULT_TEMP, seed)[0][:, 0]
    ff_default = Forcefield.load_from_file(orig_ff_name)
    coords_default = generate_ligand_samples(n_samples, mol, ff_default, DEFAULT_TEMP, seed)[0][:, 0]

    params, angle_idxs = ff.ha_handle.parameterize(mol)
    angles = analyze_angles(mol, angle_idxs, coords)
    angles_default = analyze_angles(mol, angle_idxs, coords_default)

    for i, angle in enumerate(angles):
        _, a0 = params[i]
        a0 = np.rad2deg(a0)
        angle_default = angles_default[i]

        # 5 degree cut-off comes from comparing to cos_angle=False run
        assert np.std(angle) < 5

        # The values shouldn't change much except for a0 = π
        if a0 < 175.0:
            assert np.mean(angle) == pytest.approx(np.mean(angle_default), abs=1.0)
            assert np.std(angle) == pytest.approx(np.std(angle_default), abs=2.0)
        else:
            # 10 degree cut-off comes from comparing to cos_angle=False run
            assert np.abs(np.mean(angle) - a0) < 10.0
            assert np.mean(angle) > np.mean(angles_default)


@pytest.mark.memcheck
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
            potential = generic.CentroidRestraint(
                gai,
                gbi,
                # masses,
                kb,
                b0,
            )

            params = np.array([], dtype=np.float64)

            self.compare_forces_gpu_vs_reference(x_primal, [params], box, potential, rtol, precision=precision)

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
                potential = generic.CentroidRestraint(gai, gbi, kb, b0)

                params = np.array([], dtype=np.float64)

                self.compare_forces_gpu_vs_reference(coords, [params], box, potential, rtol, precision=precision)

    @pytest.mark.skip("Currently not needed")
    def test_rmsd_restraint(self):
        # test the RMSD force by generating random coordinates.

        np.random.seed(2021)

        box = np.eye(3) * 100

        n_particles_a = 25
        n_particles_b = 7

        relative_tolerance_at_precision = {np.float32: 2e-5, np.float64: 1e-9}

        for precision, rtol in relative_tolerance_at_precision.items():

            for _ in range(100):
                coords = self.get_random_coords(n_particles_a + n_particles_b, 3)

                n = coords.shape[0]
                n_mapped_atoms = 5

                atom_map = np.stack(
                    [
                        np.random.randint(0, n_particles_a, n_mapped_atoms, dtype=np.int32),
                        np.random.randint(0, n_particles_b, n_mapped_atoms, dtype=np.int32) + n_particles_a,
                    ],
                    axis=1,
                )

                atom_map = atom_map.astype(np.int32)

                params = np.array([], dtype=np.float64)
                k = 1.35

                for precision, rtol, atol in [(np.float64, 1e-6, 1e-6), (np.float32, 1e-4, 1e-6)]:

                    ref_u = functools.partial(
                        rmsd.rmsd_restraint, group_a_idxs=atom_map[:, 0], group_b_idxs=atom_map[:, 1], k=k
                    )

                    test_u = potentials.RMSDRestraint(atom_map, n, k)

                    # note the slightly higher than usual atol (1e-6 vs 1e-8)
                    # this is due to fixed point accumulation of energy wipes out
                    # the low magnitude energies as some of test cases have
                    # an infinitesmally small absolute error (on the order of 1e-12)
                    self.compare_forces(coords, [params], box, ref_u, test_u, rtol, atol=atol, precision=precision)

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

        potential = generic.HarmonicBond(bond_idxs)
        for precision, rtol in relative_tolerance_at_precision.items():
            self.compare_forces_gpu_vs_reference(x, [params], box, potential, rtol, precision=precision)

        potential = generic.HarmonicBond(bond_idxs)

        for precision, rtol in relative_tolerance_at_precision.items():
            self.compare_forces_gpu_vs_reference(x, [params], box, potential, rtol, precision=precision)

            # test bitwise commutativity
            test_potential = potentials.HarmonicBond(bond_idxs)
            test_potential_rev = potentials.HarmonicBond(bond_idxs[:, ::-1])

            test_potential_impl = test_potential.unbound_impl(precision)
            test_potential_rev_impl = test_potential_rev.unbound_impl(precision)

            test_du_dx, test_du_dp, test_u = test_potential_impl.execute_selective(x, params, box, 1, 1, 1)

            test_du_dx_rev, test_du_dp_rev, test_u_rev = test_potential_rev_impl.execute_selective(
                x, params, box, 1, 1, 1
            )

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

        potential = generic.FlatBottomBond(bond_idxs)
        for precision, rtol in relative_tolerance_at_precision.items():
            self.compare_forces_gpu_vs_reference(x, [params], box, potential, rtol, precision=precision)

            # test bitwise commutativity
            test_potential = potentials.FlatBottomBond(bond_idxs)
            test_potential_rev = potentials.FlatBottomBond(bond_idxs[:, ::-1])

            test_potential_impl = test_potential.unbound_impl(precision)
            test_potential_rev_impl = test_potential_rev.unbound_impl(precision)

            test_du_dx, test_du_dp, test_u = test_potential_impl.execute_selective(x, params, box, 1, 1, 1)

            test_du_dx_rev, test_du_dp_rev, test_u_rev = test_potential_rev_impl.execute_selective(
                x, params, box, 1, 1, 1
            )

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

        potential = generic.HarmonicBond(bond_idxs)
        for precision, rtol in relative_tolerance_at_precision.items():
            # we assert finite-ness of the forces.
            self.compare_forces_gpu_vs_reference(x, [params], box, potential, rtol, precision=precision)

        # test with both zero and non zero terms
        x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)

        params = np.array([[2.0, 0.0], [2.0, 1.0]], dtype=np.float64)
        bond_idxs = np.array([[0, 1], [0, 2]], dtype=np.int32)

        # specific to harmonic bond force
        relative_tolerance_at_precision = {np.float32: 2e-5, np.float64: 1e-9}

        potential = generic.HarmonicBond(bond_idxs)
        for precision, rtol in relative_tolerance_at_precision.items():
            # we assert finite-ness of the forces.
            self.compare_forces_gpu_vs_reference(x, [params], box, potential, rtol, precision=precision)

    def test_harmonic_angle(self, n_particles=64, n_angles=25, dim=3):
        """Randomly connect triples of particles, then validate the resulting HarmonicAngle force"""
        np.random.seed(125)

        x = self.get_random_coords(n_particles, dim)

        atom_idxs = np.arange(n_particles)
        params = np.random.rand(n_angles, 2).astype(np.float64)
        angle_idxs = []
        for _ in range(n_angles):
            angle_idxs.append(np.random.choice(atom_idxs, size=3, replace=False))
        angle_idxs = np.array(angle_idxs, dtype=np.int32) if n_angles else np.zeros((0, 3), dtype=np.int32)

        box = np.eye(3) * 100

        # specific to harmonic angle force
        relative_tolerance_at_precision = {np.float32: 2e-5, np.float64: 1e-9}

        potential = generic.HarmonicAngle(angle_idxs)
        for precision, rtol in relative_tolerance_at_precision.items():
            self.compare_forces_gpu_vs_reference(x, [params], box, potential, rtol, precision=precision)

            # test bitwise commutativity
            test_potential = potentials.HarmonicAngle(angle_idxs)
            test_potential_rev = potentials.HarmonicAngle(angle_idxs[:, ::-1])

            test_potential_impl = test_potential.unbound_impl(precision)
            test_potential_rev_impl = test_potential_rev.unbound_impl(precision)

            test_du_dx, test_du_dp, test_u = test_potential_impl.execute_selective(x, params, box, 1, 1, 1)

            test_du_dx_rev, test_du_dp_rev, test_u_rev = test_potential_rev_impl.execute_selective(
                x, params, box, 1, 1, 1
            )

            np.testing.assert_array_equal(test_u, test_u_rev)
            np.testing.assert_array_equal(test_du_dx, test_du_dx_rev)
            np.testing.assert_array_equal(test_du_dp, test_du_dp_rev)

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

        potential = generic.PeriodicTorsion(torsion_idxs)
        for precision, rtol in relative_tolerance_at_precision.items():
            self.compare_forces_gpu_vs_reference(x, [params], box, potential, rtol, precision=precision)

        potential = generic.PeriodicTorsion(torsion_idxs)
        for precision, rtol in relative_tolerance_at_precision.items():
            self.compare_forces_gpu_vs_reference(x, [params], box, potential, rtol, precision=precision)

            # test bitwise commutativity
            test_potential = potentials.PeriodicTorsion(torsion_idxs)
            test_potential_rev = potentials.PeriodicTorsion(torsion_idxs[:, ::-1])

            test_potential_impl = test_potential.unbound_impl(precision)
            test_potential_rev_impl = test_potential_rev.unbound_impl(precision)

            test_du_dx, test_du_dp, test_u = test_potential_impl.execute_selective(x, params, box, 1, 1, 1)

            test_du_dx_rev, test_du_dp_rev, test_u_rev = test_potential_rev_impl.execute_selective(
                x, params, box, 1, 1, 1
            )

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

        potential = generic.ChiralAtomRestraint(restr_idxs)
        for precision, rtol in relative_tolerance_at_precision.items():
            self.compare_forces_gpu_vs_reference(x, [params], box, potential, rtol, precision=precision)

            with self.assertRaises(Exception):
                # wrong length
                bad_idxs = np.array([[0, 1, 2, 3, 4], [4, 4, 3, 2, 1]])
                bad_potential = potentials.ChiralAtomRestraint(bad_idxs)
                bad_potential.unbound_impl()

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

        potential = generic.ChiralBondRestraint(restr_idxs, signs)
        for precision, rtol in relative_tolerance_at_precision.items():
            self.compare_forces_gpu_vs_reference(x, [params], box, potential, rtol, precision=precision)

            with self.assertRaises(RuntimeError):
                # wrong length idxs
                bad_idxs = np.array([[0, 1, 2, 3, 4], [4, 4, 3, 2, 1]], dtype=np.int32)
                bad_signs = np.array([1, -1], dtype=np.int32)
                bad_potential = potentials.ChiralBondRestraint(bad_idxs, bad_signs)
                bad_potential.unbound_impl(precision)

            with self.assertRaises(RuntimeError):
                # inconsistent lengths between idxs and signs
                bad_idxs = np.array([[0, 1, 2, 3], [4, 5, 3, 2]], dtype=np.int32)
                bad_signs = np.array([1, -1, 1], dtype=np.int32)
                bad_potential = potentials.ChiralBondRestraint(bad_idxs, bad_signs)
                bad_potential.unbound_impl(precision)
