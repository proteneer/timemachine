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
