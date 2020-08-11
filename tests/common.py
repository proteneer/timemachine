import unittest
import numpy as np
import jax
import jax.numpy as jnp
import functools


from timemachine.potentials import bonded, nonbonded, gbsa
from timemachine.lib import ops, custom_ops

from hilbertcurve.hilbertcurve import HilbertCurve


def prepare_gbsa_system(
    x,
    # P_charges,
    # P_radii,
    # P_scale_factors,
    alpha,
    beta,
    gamma,
    dielectric_offset,
    surface_tension,
    solute_dielectric,
    solvent_dielectric,
    probe_radius,
    cutoff_radii,
    cutoff_force,
    lambda_plane_idxs,
    lambda_offset_idxs,
    params=None,
    precision=np.float64):


    assert cutoff_radii == cutoff_force

    N = x.shape[0]
    D = x.shape[1]

    if params is None:
        params = np.array([], dtype=np.float64)

    # charges
    charge_params = (np.random.rand(N).astype(np.float64)-0.5)*np.sqrt(138.935456)
    # charge_param_idxs = np.random.randint(low=0, high=P_charges, size=(N), dtype=np.int32) + len(params)
    # params = np.concatenate([params, charge_params])

    # gb radiis
    radii_params = 1.5*np.random.rand(N).astype(np.float64) + 1.0 # 1.0 to 2.5
    radii_params = radii_params/10 # convert to nm form
    # radii_param_idxs = np.random.randint(low=0, high=P_radii, size=(N), dtype=np.int32) + len(params)
    # params = np.concatenate([params, radii_params])

    # scale factors
    scale_params = np.random.rand(N).astype(np.float64)/3 + 0.75
    # scale_param_idxs = np.random.randint(low=0, high=P_scale_factors, size=(N), dtype=np.int32) + len(params)
    # params = np.concatenate([params, scale_params])

    gb_params = np.stack([radii_params, scale_params], axis=1)
    # lambda_plane_idxs = np.random.randint(
    #     low=0,
    #     high=2,
    #     size=(N),
    #     dtype=np.int32
    # )

    # lambda_offset_idxs = np.random.randint(
    #     low=0,
    #     high=2,
    #     size=(N),
    #     dtype=np.int32
    # )

    custom_gb_ctor = functools.partial(ops.GBSA,
        charge_params,
        gb_params,
        # charge_param_idxs,
        # radii_param_idxs,
        # scale_param_idxs,
        lambda_plane_idxs,
        lambda_offset_idxs,
        alpha,
        beta,
        gamma,
        dielectric_offset,
        surface_tension,
        solute_dielectric,
        solvent_dielectric,
        probe_radius,
        cutoff_radii,
        cutoff_force,
        precision=precision
    )

    # ideally cutoff is the max(cutoff_radii, cutoff_force)
    # box = np.array([
        # [10000.0, 0.0, 0.0, 0.0],
        # [0.0, 10000.0, 0.0, 0.0],
        # [0.0, 0.0, 10000.0, 0.0],
        # [0.0, 0.0, 0.0, 2*cutoff_radii],
    # ])

    box = None

    gbsa_obc_fn = functools.partial(
        gbsa.gbsa_obc,
        # box=box,
        # charge_idxs=charge_param_idxs,
        # radii_idxs=radii_param_idxs,
        # scale_idxs=scale_param_idxs,
        # charge_params=charge_params,
        # gb_params=gb_params,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        dielectric_offset=dielectric_offset,
        surface_tension=surface_tension,
        solute_dielectric=solute_dielectric,
        solvent_dielectric=solvent_dielectric,
        probe_radius=probe_radius,
        cutoff_radii=cutoff_radii,
        cutoff_force=cutoff_force,
        lambda_plane_idxs=lambda_plane_idxs,
        lambda_offset_idxs=lambda_offset_idxs
    )

    return (charge_params, gb_params), gbsa_obc_fn, custom_gb_ctor


def prepare_lj_system(
    x,
    E, # number of exclusions
    lambda_plane_idxs,
    lambda_offset_idxs,
    lambda_group_idxs,
    p_scale,
    cutoff=100.0,
    precision=np.float64):

    N = x.shape[0]
    D = x.shape[1]

    # charge_params = (np.random.rand(N).astype(np.float64) - 0.5)*np.sqrt(138.935456)
    sig_params = np.random.rand(N) / p_scale
    eps_params = np.random.rand(N)
    lj_params = np.stack([sig_params, eps_params], axis=1)

    atom_idxs = np.arange(N)
    exclusion_idxs = np.random.choice(atom_idxs, size=(E, 2), replace=False)
    exclusion_idxs = np.array(exclusion_idxs, dtype=np.int32).reshape(-1, 2)

    # charge_scales = np.random.rand(E)
    lj_scales = np.random.rand(E)

    custom_nonbonded_ctor = functools.partial(ops.LennardJones,
        # charge_params,
        lj_params,
        exclusion_idxs,
        # charge_scales,
        lj_scales,
        lambda_plane_idxs,
        lambda_offset_idxs,
        lambda_group_idxs,
        cutoff,
        precision=precision
    )

    # disable PBCs
    # make sure this is big enough!
    # box = np.array([
    #     [100.0, 0.0, 0.0, 0.0],
    #     [0.0, 100.0, 0.0, 0.0],
    #     [0.0, 0.0, 100.0, 0.0],
    #     [0.0, 0.0, 0.0, 2*cutoff],
    # ])

    ref_total_energy = functools.partial(
        nonbonded.group_lennard_jones,
        exclusion_idxs=exclusion_idxs,
        # charge_scales=charge_scales,
        lj_scales=lj_scales,
        cutoff=cutoff,
        lambda_plane_idxs=lambda_plane_idxs,
        lambda_offset_idxs=lambda_offset_idxs,
        lambda_group_idxs=lambda_group_idxs
    )

    return lj_params, ref_total_energy, custom_nonbonded_ctor



def prepare_es_system(
    x,
    E, # number of exclusions
    lambda_plane_idxs,
    lambda_offset_idxs,
    p_scale,
    cutoff=100.0,
    precision=np.float64):

    N = x.shape[0]
    D = x.shape[1]

    charge_params = (np.random.rand(N).astype(np.float64) - 0.5)*np.sqrt(138.935456)

    atom_idxs = np.arange(N)
    exclusion_idxs = np.random.choice(atom_idxs, size=(E, 2), replace=False)
    exclusion_idxs = np.array(exclusion_idxs, dtype=np.int32).reshape(-1, 2)

    charge_scales = np.random.rand(E)

    custom_nonbonded_ctor = functools.partial(ops.Electrostatics,
        charge_params,
        exclusion_idxs,
        charge_scales,
        lambda_plane_idxs,
        lambda_offset_idxs,
        cutoff,
        precision=precision
    )

    # disable PBCs
    # make sure this is big enough!
    # box = np.array([
    #     [100.0, 0.0, 0.0, 0.0],
    #     [0.0, 100.0, 0.0, 0.0],
    #     [0.0, 0.0, 100.0, 0.0],
    #     [0.0, 0.0, 0.0, 2*cutoff],
    # ])

    ref_total_energy = functools.partial(
        nonbonded.nongroup_electrostatics,
        exclusion_idxs=exclusion_idxs,
        charge_scales=charge_scales,
        cutoff=cutoff,
        lambda_plane_idxs=lambda_plane_idxs,
        lambda_offset_idxs=lambda_offset_idxs
    )

    return charge_params, ref_total_energy, custom_nonbonded_ctor



def prepare_nonbonded_system(
    x,
    E, # number of exclusions
    lambda_plane_idxs,
    lambda_offset_idxs,
    p_scale,
    cutoff=100.0,
    precision=np.float64):

    N = x.shape[0]
    D = x.shape[1]

    charge_params = (np.random.rand(N).astype(np.float64) - 0.5)*np.sqrt(138.935456)
    sig_params = np.random.rand(N) / p_scale
    eps_params = np.random.rand(N)
    lj_params = np.stack([sig_params, eps_params], axis=1)

    atom_idxs = np.arange(N)
    exclusion_idxs = np.random.choice(atom_idxs, size=(E, 2), replace=False)
    exclusion_idxs = np.array(exclusion_idxs, dtype=np.int32).reshape(-1, 2)

    charge_scales = np.random.rand(E)
    lj_scales = np.random.rand(E)

    custom_nonbonded_ctor = functools.partial(ops.Nonbonded,
        charge_params,
        lj_params,
        exclusion_idxs,
        charge_scales,
        lj_scales,
        lambda_plane_idxs,
        lambda_offset_idxs,
        cutoff,
        precision=precision
    )

    # disable PBCs
    # make sure this is big enough!
    # box = np.array([
    #     [100.0, 0.0, 0.0, 0.0],
    #     [0.0, 100.0, 0.0, 0.0],
    #     [0.0, 0.0, 100.0, 0.0],
    #     [0.0, 0.0, 0.0, 2*cutoff],
    # ])

    ref_total_energy = functools.partial(
        nonbonded.nonbonded,
        exclusion_idxs=exclusion_idxs,
        charge_scales=charge_scales,
        lj_scales=lj_scales,
        cutoff=cutoff,
        lambda_plane_idxs=lambda_plane_idxs,
        lambda_offset_idxs=lambda_offset_idxs
    )

    return (charge_params, lj_params), ref_total_energy, custom_nonbonded_ctor

def prepare_restraints(
    x,
    B,
    precision):

    N = x.shape[0]
    D = x.shape[1]

    atom_idxs = np.arange(N)

    params = np.random.randn(B, 3).astype(np.float64)

    bond_params = np.random.rand(B, 2).astype(np.float64)
    bond_idxs = []
    for _ in range(B):
        bond_idxs.append(np.random.choice(atom_idxs, size=2, replace=False))
    bond_idxs = np.array(bond_idxs, dtype=np.int32)

    lambda_flags = np.random.randint(0, 2, size=(B,)).astype(np.int32)

    custom_restraint = ops.Restraint(bond_idxs, params, lambda_flags, precision=precision)
    restraint_fn = functools.partial(bonded.restraint, box=None, lamb_flags=lambda_flags, bond_idxs=bond_idxs)

    return (params, restraint_fn), custom_restraint

def prepare_bonded_system(
    x,
    B,
    A,
    T,
    precision):

    N = x.shape[0]
    D = x.shape[1]

    atom_idxs = np.arange(N)

    bond_params = np.random.rand(B, 2).astype(np.float64)
    bond_idxs = []
    for _ in range(B):
        bond_idxs.append(np.random.choice(atom_idxs, size=2, replace=False))
    bond_idxs = np.array(bond_idxs, dtype=np.int32)
    # params = np.concatenate([params, bond_params])

    # angle_params = np.random.rand(P_angles).astype(np.float64)
    # angle_param_idxs = np.random.randint(low=0, high=P_angles, size=(A,2), dtype=np.int32) + len(params)
    # angle_idxs = []
    # for _ in range(A):
    #     angle_idxs.append(np.random.choice(atom_idxs, size=3, replace=False))
    # angle_idxs = np.array(angle_idxs, dtype=np.int32)
    # params = np.concatenate([params, angle_params])

    # torsion_params = np.random.rand(P_torsions).astype(np.float64)
    # torsion_param_idxs = np.random.randint(low=0, high=P_torsions, size=(T,3), dtype=np.int32) + len(params)
    # torsion_idxs = []
    # for _ in range(T):
    #     torsion_idxs.append(np.random.choice(atom_idxs, size=4, replace=False))
    # torsion_idxs = np.array(torsion_idxs, dtype=np.int32)
    # params = np.concatenate([params, torsion_params])


    print("precision", precision)
    custom_bonded = ops.HarmonicBond(bond_idxs, bond_params, precision=precision)
    harmonic_bond_fn = functools.partial(bonded.harmonic_bond, box=None, bond_idxs=bond_idxs)

    # custom_angles = ops.HarmonicAngle(angle_idxs, angle_param_idxs, precision=precision)
    # harmonic_angle_fn = functools.partial(bonded.harmonic_angle, box=None, angle_idxs=angle_idxs, param_idxs=angle_param_idxs)

    # custom_torsions = ops.PeriodicTorsion(torsion_idxs, torsion_param_idxs, precision=precision)
    # periodic_torsion_fn = functools.partial(bonded.periodic_torsion, box=None, torsion_idxs=torsion_idxs, param_idxs=torsion_param_idxs)

    return (bond_params, harmonic_bond_fn), custom_bonded
    # return params, [harmonic_bond_fn, harmonic_angle_fn, periodic_torsion_fn], [custom_bonded, custom_angles, custom_torsions]

def hilbert_sort(conf, D):
    hc = HilbertCurve(64, D)
    int_confs = (conf*1000).astype(np.int64)
    dists = []
    for xyz in int_confs.tolist():
        dist = hc.distance_from_coordinates(xyz)
        dists.append(dist)
    perm = np.argsort(dists)
    # np.random.shuffle(perm)
    return perm


class GradientTest(unittest.TestCase):

    def get_random_coords(self, N, D):
        x = np.random.rand(N,D).astype(dtype=np.float64)
        return x

    def get_water_coords(self, D, sort=False):
        x = np.load("tests/data/water.npy").astype(np.float32).astype(np.float64)
        x = x[:, :D]

        # x = (x).astype(np.float64)
        # if sort:
            # perm = hilbert_sort(x, D)
            # x = x[perm]

        return x

    def get_cdk8_coords(self, D, sort=False):
        x = np.load("cdk8.npy").astype(np.float64)
        print("num_atoms", x.shape[0])
        if sort:
            perm = hilbert_sort(x, D)
            x = x[perm]

        return x


    def assert_equal_vectors(self, truth, test, rtol):
        """
        OpenMM convention - errors are compared against norm of force vectors
        """
        assert np.array(truth).shape == np.array(test).shape

        norms = np.linalg.norm(truth, axis=-1, keepdims=True)
        norms = np.where(norms < 1., 1.0, norms)
        errors = (truth-test)/norms

        # print(errors)
        max_error = np.amax(np.abs(errors))
        mean_error = np.mean(np.abs(errors).reshape(-1))
        std_error = np.std(errors.reshape(-1))
        max_error_arg = np.argmax(errors)//truth.shape[1]

        errors = np.abs(errors) > rtol

        print("max relative error", max_error, "rtol", rtol, norms[max_error_arg], "mean error", mean_error, "std error", std_error)
        if np.sum(errors) > 0:
            print("FATAL: max relative error", max_error, truth[max_error_arg], test[max_error_arg])
            assert 0

    def assert_param_derivs(self, truth, test):
        for ref, test in zip(truth, test):
            if np.abs(ref) < 1:
                np.testing.assert_almost_equal(ref, test, decimal=2)
            else:
                np.testing.assert_allclose(ref, test, rtol=5e-3)


    def compare_forces(self, x, lamb, x_tangent, lamb_tangent, ref_nrg_fn, custom_force, precision, rtol=None):
        # this is actually sort of important for 64bit, don't remove me!
        #
        # params = (params.astype(np.float32)).astype(np.float64)

        N = x.shape[0]
        D = x.shape[1]

        assert x.dtype == np.float64
        # assert params.dtype == np.float64

        ref_nrg = ref_nrg_fn(x, lamb)
        grad_fn = jax.grad(ref_nrg_fn, argnums=(0, 1))
        ref_dx, ref_dl = grad_fn(x, lamb)
        test_dx, test_dl, test_nrg = custom_force.execute_lambda(x, lamb)


        print(ref_dl, test_dl)
        print(ref_nrg, test_nrg)
        # print(ref_dx, "\n", test_dx)

        np.testing.assert_allclose(ref_nrg, test_nrg, rtol)

        self.assert_equal_vectors(
            np.array(ref_dx),
            np.array(test_dx),
            rtol,
        )

        if ref_dl == 0:
            np.testing.assert_almost_equal(ref_dl, test_dl, 1e-5)
        else:
            np.testing.assert_allclose(ref_dl, test_dl, rtol)


        print("FIRST ORDER PASSED, SKIPPING VJPs")

        return

        test_x_tangent, test_x_primal = custom_force.execute_lambda_jvp(
            x,
            lamb,
            x_tangent,
            lamb_tangent
        )

        primals = (x, lamb)
        tangents = (x_tangent, lamb_tangent)

        _, t = jax.jvp(grad_fn, primals, tangents)

        self.assert_equal_vectors(
            t[0],
            test_x_tangent,
            rtol,
        )

        return

        assert 0

        ref_p_tangent = t[1] # use t[2] after switcheroo

        # TBD compare relative to the *norm* of the group of similar derivatives.
        # for r_idx, (r, tt) in enumerate(zip(t[1], test_p_tangent)):
        #     err = abs((r - tt)/r)
        #     if err > 1e-4:
        #         print(r_idx, err, r, tt)

        # print(ref_p_tangent)
        # print(test_p_tangent)
        # print(np.abs(ref_p_tangent - test_p_tangent))

        if precision == np.float64:
            
            print(np.amax(ref_p_tangent - test_p_tangent), np.amin(ref_p_tangent - test_p_tangent))

            for a, b in zip(ref_p_tangent, test_p_tangent):
                try:
                    np.testing.assert_allclose(a, b, rtol=1e-8)
                except:
                    assert 0
            np.testing.assert_allclose(ref_p_tangent, test_p_tangent, rtol=rtol)

        else:
            self.assert_param_derivs(ref_p_tangent, test_p_tangent)



    # def compare_forces(self, x, params, lamb, ref_nrg_fn, custom_force, precision, rtol=None):
    #     x = (x.astype(np.float32)).astype(np.float64)
    #     params = (params.astype(np.float32)).astype(np.float64)

    #     N = x.shape[0]
    #     D = x.shape[1]

    #     assert x.dtype == np.float64
    #     assert params.dtype == np.float64

    #     ref_nrg = ref_nrg_fn(x, params, lamb)
    #     grad_fn = jax.grad(ref_nrg_fn, argnums=(0, 1, 2))
    #     ref_dx, ref_dp, ref_dl = grad_fn(x, params, lamb)
    #     test_dx, test_dl, test_nrg = custom_force.execute_lambda(x, params, lamb)

    #     np.testing.assert_allclose(ref_nrg, test_nrg, rtol)

    #     self.assert_equal_vectors(
    #         np.array(ref_dx),
    #         np.array(test_dx),
    #         rtol,
    #     )

    #     if ref_dl == 0:
    #         np.testing.assert_almost_equal(ref_dl, test_dl, 1e-5)
    #     else:
    #         np.testing.assert_allclose(ref_dl, test_dl, rtol)

    #     x_tangent = np.random.rand(N, D).astype(np.float64)
    #     params_tangent = np.zeros_like(params)
    #     lamb_tangent = np.random.rand()


    #     test_x_tangent, test_p_tangent, test_x_primal, test_p_primal = custom_force.execute_lambda_jvp(
    #         x,
    #         params,
    #         lamb,
    #         x_tangent,
    #         params_tangent,
    #         lamb_tangent
    #     )

    #     primals = (x, params, lamb)
    #     tangents = (x_tangent, params_tangent, lamb_tangent)

    #     _, t = jax.jvp(grad_fn, primals, tangents)

    #     ref_p_tangent = t[1]

    #     self.assert_equal_vectors(
    #         t[0],
    #         test_x_tangent,
    #         rtol,
    #     )

    #     # TBD compare relative to the *norm* of the group of similar derivatives.
    #     # for r_idx, (r, tt) in enumerate(zip(t[1], test_p_tangent)):
    #     #     err = abs((r - tt)/r)
    #     #     if err > 1e-4:
    #     #         print(r_idx, err, r, tt)

    #     # print(ref_p_tangent)
    #     # print(test_p_tangent)
    #     # print(np.abs(ref_p_tangent - test_p_tangent))

    #     if precision == np.float64:
            
    #         print(np.amax(ref_p_tangent - test_p_tangent), np.amin(ref_p_tangent - test_p_tangent))

    #         for a, b in zip(ref_p_tangent, test_p_tangent):
    #             try:
    #                 np.testing.assert_allclose(a, b, rtol=1e-8)
    #             except:
    #                 assert 0
    #         np.testing.assert_allclose(ref_p_tangent, test_p_tangent, rtol=rtol)

    #     else:
    #         self.assert_param_derivs(ref_p_tangent, test_p_tangent)

