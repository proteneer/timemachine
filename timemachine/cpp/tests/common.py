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
    P_charges,
    P_radii,
    P_scale_factors,
    alpha,
    beta,
    gamma,
    dielectric_offset,
    surface_tension,
    solute_dielectric,
    solvent_dielectric,
    probe_radius,
    params=None,
    precision=np.float64):

    N = x.shape[0]
    D = x.shape[1]

    if params is None:
        params = np.array([], dtype=np.float64)

    # charges
    charge_params = np.random.rand(P_charges).astype(np.float64)*np.sqrt(138.935456)/4
    charge_param_idxs = np.random.randint(low=0, high=P_charges, size=(N), dtype=np.int32) + len(params)
    params = np.concatenate([params, charge_params])

    # gb radiis
    radii_params = 1.5*np.random.rand(P_radii).astype(np.float64) + 1.0 # 1.0 to 2.5
    radii_params = radii_params/10 # convert to nm form
    radii_param_idxs = np.random.randint(low=0, high=P_radii, size=(N), dtype=np.int32) + len(params)
    params = np.concatenate([params, radii_params])

    # scale factors
    scale_params = np.random.rand(P_scale_factors).astype(np.float64)/3 + 0.75
    scale_param_idxs = np.random.randint(low=0, high=P_scale_factors, size=(N), dtype=np.int32) + len(params)
    params = np.concatenate([params, scale_params])

    cutoff = 100.0

    custom_gb = ops.GBSA(
        charge_param_idxs,
        radii_param_idxs,
        scale_param_idxs,
        alpha,
        beta,
        gamma,
        dielectric_offset,
        surface_tension,
        solute_dielectric,
        solvent_dielectric,
        probe_radius,
        cutoff,
        D,
        precision=precision
    )

    gbsa_obc_fn = functools.partial(
        gbsa.gbsa_obc,
        charge_idxs=charge_param_idxs,
        radii_idxs=radii_param_idxs,
        scale_idxs=scale_param_idxs,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        dielectric_offset=dielectric_offset,
        surface_tension=surface_tension,
        solute_dielectric=solute_dielectric,
        solvent_dielectric=solvent_dielectric,
        probe_radius=probe_radius
    )

    def ref_total_energy(x, p):
        return lj_fn(x, p) - lj_fn_exc(x, p) + es_fn(x, p)

    return params, [gbsa_obc_fn], [custom_gb]


def prepare_nonbonded_system(
    x,
    E, # number of exclusions
    P_charges,
    P_lj,
    P_exc,
    params=None,
    p_scale=4.0,
    e_scale=1.0,
    cutoff=100.0,
    custom_D=None,
    precision=np.float64):

    N = x.shape[0]
    D = x.shape[1]

    if params is None:
        params = np.array([], dtype=np.float64)

    # charges
    charge_params = np.random.rand(P_charges).astype(np.float64)/e_scale
    # charge_params = np.zeros_like(charge_params)
    charge_param_idxs = np.random.randint(low=0, high=P_charges, size=(N), dtype=np.int32) + len(params)
    params = np.concatenate([params, charge_params])

    # lennard jones
    lj_sig_params = np.random.rand(P_lj)/p_scale # we want these to be pretty small for numerical stability
    lj_sig_idxs = np.random.randint(low=0, high=P_lj, size=(N,), dtype=np.int32) + len(params)
    params = np.concatenate([params, lj_sig_params])

    lj_eps_params = np.random.rand(P_lj)
    lj_eps_idxs = np.random.randint(low=0, high=P_lj, size=(N,), dtype=np.int32) + len(params)
    params = np.concatenate([params, lj_eps_params])

    lj_param_idxs = np.stack([lj_sig_idxs, lj_eps_idxs], axis=-1)

    # generate exclusion parameters
    exclusion_idxs = np.random.randint(low=0, high=N, size=(E,2), dtype=np.int32)
    for e_idx, (i,j) in enumerate(exclusion_idxs):
        if i == j:
            exclusion_idxs[e_idx][0] = i
            exclusion_idxs[e_idx][1] = (j+1) % N # mod is in case we overflow

    for e_idx, (i,j) in enumerate(exclusion_idxs):
        if i == j:
            raise Exception("BAD")

    exclusion_params = np.random.rand(P_exc).astype(np.float64) # must be between 0 and 1
    exclusion_charge_idxs = np.random.randint(low=0, high=P_exc, size=(E), dtype=np.int32) + len(params)
    exclusion_lj_idxs = np.random.randint(low=0, high=P_exc, size=(E), dtype=np.int32) + len(params)
    params = np.concatenate([params, exclusion_params])

    if custom_D is None:
        custom_D = D

    custom_nonbonded = ops.Nonbonded(
        charge_param_idxs,
        lj_param_idxs,
        exclusion_idxs,
        exclusion_charge_idxs,
        exclusion_lj_idxs,
        cutoff,
        custom_D,
        precision=precision
    )

    lj_fn = functools.partial(nonbonded.lennard_jones, box=None, param_idxs=lj_param_idxs, cutoff=cutoff)
    lj_fn_exc = functools.partial(nonbonded.lennard_jones_exclusion, box=None, param_idxs=lj_param_idxs, cutoff=cutoff, exclusions=exclusion_idxs, exclusion_scale_idxs=exclusion_lj_idxs)
    es_fn = functools.partial(nonbonded.simple_energy, param_idxs=charge_param_idxs, cutoff=cutoff, exclusions=exclusion_idxs, exclusion_scale_idxs=exclusion_charge_idxs)

    def ref_total_energy(x, p):
        return lj_fn(x, p) - lj_fn_exc(x, p) + es_fn(x, p)

    return params, [ref_total_energy], [custom_nonbonded]


def prepare_bonded_system(
    x,
    P_bonds,
    P_angles,
    P_torsions,
    B,
    A,
    T,
    precision):

    N = x.shape[0]
    D = x.shape[1]

    atom_idxs = np.arange(N)

    params = np.array([], dtype=np.float64);
    bond_params = np.random.rand(P_bonds).astype(np.float64)
    bond_param_idxs = np.random.randint(low=0, high=P_bonds, size=(B,2), dtype=np.int32) + len(params)
    bond_idxs = []
    for _ in range(B):
        bond_idxs.append(np.random.choice(atom_idxs, size=2, replace=False))
    bond_idxs = np.array(bond_idxs, dtype=np.int32)
    params = np.concatenate([params, bond_params])

    params = np.array([], dtype=np.float64);
    angle_params = np.random.rand(P_angles).astype(np.float64)
    angle_param_idxs = np.random.randint(low=0, high=P_angles, size=(A,2), dtype=np.int32) + len(params)
    angle_idxs = []
    for _ in range(A):
        angle_idxs.append(np.random.choice(atom_idxs, size=3, replace=False))
    angle_idxs = np.array(angle_idxs, dtype=np.int32)
    params = np.concatenate([params, angle_params])

    params = np.array([], dtype=np.float64);
    torsion_params = np.random.rand(P_torsions).astype(np.float64)
    torsion_param_idxs = np.random.randint(low=0, high=P_torsions, size=(T,3), dtype=np.int32) + len(params)
    torsion_idxs = []
    for _ in range(T):
        torsion_idxs.append(np.random.choice(atom_idxs, size=4, replace=False))
    torsion_idxs = np.array(torsion_idxs, dtype=np.int32)

    params = np.concatenate([params, torsion_params])

    custom_bonded = ops.HarmonicBond(bond_idxs, bond_param_idxs, D, precision=precision)
    harmonic_bond_fn = functools.partial(bonded.harmonic_bond, box=None, bond_idxs=bond_idxs, param_idxs=bond_param_idxs)

    custom_angles = ops.HarmonicAngle(angle_idxs, angle_param_idxs, D, precision=precision)
    harmonic_angle_fn = functools.partial(bonded.harmonic_angle, box=None, angle_idxs=angle_idxs, param_idxs=angle_param_idxs)

    custom_torsions = ops.PeriodicTorsion(torsion_idxs, torsion_param_idxs, D, precision=precision)
    periodic_torsion_fn = functools.partial(bonded.periodic_torsion, box=None, torsion_idxs=torsion_idxs, param_idxs=torsion_param_idxs)

    return params, [harmonic_bond_fn], [custom_bonded]

def hilbert_sort(conf, D):
    hc = HilbertCurve(32, D)
    int_confs = (conf*10000).astype(np.int64)
    dists = []
    for xyz in int_confs.tolist():
        dist = hc.distance_from_coordinates(xyz)
        dists.append(dist)
    perm = np.argsort(dists)
    return perm


class GradientTest(unittest.TestCase):

    def get_random_coords(self, N, D):
        x = np.random.rand(N,D).astype(dtype=np.float64)
        return x

    def get_water_coords(self, D, sort=False):
        x = np.load("water.npy").astype(np.float64)
        x = x[:2976, :D]
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

    def compare_forces(self, x, params, ref_nrg_fn, custom_force, precision, rtol=None):
        x = (x.astype(np.float32)).astype(np.float64)
        params = (params.astype(np.float32)).astype(np.float64)

        N = x.shape[0]
        D = x.shape[1]

        assert x.dtype == np.float64
        assert params.dtype == np.float64

        # test_dx, test_dp = custom_force.execute_first_order(x, params)
        test_dx = custom_force.execute(x, params)

        grad_fn = jax.grad(ref_nrg_fn, argnums=(0, 1))
        ref_dx, ref_dp = grad_fn(x, params)

        self.assert_equal_vectors(
            np.array(ref_dx),
            np.array(test_dx),
            rtol,
        )

        x_tangent = np.random.rand(N, D).astype(np.float32).astype(np.float64)
        params_tangent = np.zeros_like(params)

        test_x_tangent, test_p_tangent = custom_force.execute_jvp(
            x,
            params,
            x_tangent,
            params_tangent
        )

        primals = (x, params)
        tangents = (x_tangent, params_tangent)

        _, t = jax.jvp(grad_fn, primals, tangents)

        ref_p_tangent = t[1]

        self.assert_equal_vectors(
            t[0],
            test_x_tangent,
            rtol,
        )

        # TBD compare relative to the *norm* of the group of similar derivatives.
        # for r_idx, (r, tt) in enumerate(zip(t[1], test_p_tangent)):
        #     err = abs((r - tt)/r)
        #     if err > 1e-4:
        #         print(r_idx, err, r, tt)

        if precision == np.float64:
            np.testing.assert_allclose(ref_p_tangent, test_p_tangent, rtol=rtol)
        else:
            self.assert_param_derivs(ref_p_tangent, test_p_tangent)

