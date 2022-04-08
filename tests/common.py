import contextlib
import functools
import itertools
import os
import unittest
from tempfile import TemporaryDirectory
from typing import List

import jax
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve
from numpy.typing import NDArray

from timemachine.constants import ONE_4PI_EPS0
from timemachine.ff import Forcefield
from timemachine.lib import potentials
from timemachine.potentials import bonded, nonbonded


@contextlib.contextmanager
def temporary_working_dir():
    init_dir = os.getcwd()
    with TemporaryDirectory() as temp:
        try:
            os.chdir(temp)
            yield temp
        finally:
            os.chdir(init_dir)


def get_110_ccc_ff():
    forcefield = Forcefield.load_from_file("smirnoff_1_1_0_ccc.py")
    return forcefield


def prepare_lj_system(
    x,
    E,  # number of exclusions
    lambda_plane_idxs,
    lambda_offset_idxs,
    p_scale,
    tip3p,
    cutoff=100.0,
    precision=np.float64,
):

    assert x.ndim == 2
    N = x.shape[0]
    # D = x.shape[1]

    sig_params = np.random.rand(N) / p_scale
    eps_params = np.random.rand(N)
    lj_params = np.stack([sig_params, eps_params], axis=1)

    if tip3p:
        mask = []
        for i in range(N):
            if i % 3 == 0:
                mask.append(1)
            else:
                mask.append(0)
        mask = np.array(mask)
        eps_params = lj_params[:, 1]
        tip_params = np.where(mask, eps_params, 0)
        lj_params[:, 1] = tip_params

    atom_idxs = np.arange(N)
    exclusion_idxs = np.random.choice(atom_idxs, size=(E, 2), replace=False)
    exclusion_idxs = np.array(exclusion_idxs, dtype=np.int32).reshape(-1, 2)

    lj_scales = np.random.rand(E)

    test_potential = potentials.LennardJones(
        exclusion_idxs, lj_scales, lambda_plane_idxs, lambda_offset_idxs, cutoff, precision=precision
    )

    ref_potential = functools.partial(
        nonbonded.lennard_jones_v2,
        exclusion_idxs=exclusion_idxs,
        lj_scales=lj_scales,
        cutoff=cutoff,
        lambda_plane_idxs=lambda_plane_idxs,
        lambda_offset_idxs=lambda_offset_idxs,
    )

    return lj_params, ref_potential, test_potential


# def prepare_es_system(
#     x,
#     E, # number of exclusions
#     lambda_offset_idxs,
#     p_scale,
#     cutoff,
#     precision=np.float64):

#     N = x.shape[0]
#     D = x.shape[1]

#     charge_params = (np.random.rand(N).astype(np.float64) - 0.5)*np.sqrt(ONE_4PI_EPS0)

#     atom_idxs = np.arange(N)
#     exclusion_idxs = np.random.choice(atom_idxs, size=(E, 2), replace=False)
#     exclusion_idxs = np.array(exclusion_idxs, dtype=np.int32).reshape(-1, 2)

#     charge_scales = np.random.rand(E)

#     # beta = np.random.rand()*2

#     beta = 2.0

#     test_potential = potentials.Electrostatics(
#         exclusion_idxs,
#         charge_scales,
#         lambda_offset_idxs,
#         beta,
#         cutoff,
#         precision=precision
#     )

#     ref_total_energy = functools.partial(
#         nonbonded.electrostatics_v2,
#         exclusion_idxs=exclusion_idxs,
#         charge_scales=charge_scales,
#         beta=beta,
#         cutoff=cutoff,
#         lambda_offset_idxs=lambda_offset_idxs
#     )

#     return charge_params, ref_total_energy, test_potential


def prepare_reference_nonbonded(params, exclusion_idxs, scales, lambda_plane_idxs, lambda_offset_idxs, beta, cutoff):

    N = params.shape[0]
    charge_rescale_mask, lj_rescale_mask = nonbonded.convert_exclusions_to_rescale_masks(exclusion_idxs, scales, N)

    ref_total_energy = functools.partial(
        nonbonded.nonbonded_v3,
        charge_rescale_mask=charge_rescale_mask,
        lj_rescale_mask=lj_rescale_mask,
        beta=beta,
        cutoff=cutoff,
        lambda_plane_idxs=lambda_plane_idxs,
        lambda_offset_idxs=lambda_offset_idxs,
    )

    return ref_total_energy


def prepare_system_params(x: NDArray, sigma_scale: float = 5.0) -> NDArray:
    """
    Prepares random parameters given a set of coordinates. The parameters are adjusted to be the correct
    order of magnitude.

    Parameters
    ----------

    x: Numpy array of Coordinates

    sigma_scale: Factor to scale down sigma values by

    Returns
    -------
    (N, 3) np.ndarray containing charges, sigmas and epsilons respectively.
    """
    assert x.ndim == 2
    N = x.shape[0]

    params = np.stack(
        [
            (np.random.rand(N).astype(np.float64) - 0.5) * np.sqrt(ONE_4PI_EPS0),  # q
            np.random.rand(N).astype(np.float64) / sigma_scale,  # sig
            np.random.rand(N).astype(np.float64),  # eps
        ],
        axis=1,
    )

    params[:, 1] = params[:, 1] / 2
    params[:, 2] = np.sqrt(params[:, 2])
    return params


def prepare_water_system(x, lambda_plane_idxs, lambda_offset_idxs, p_scale, cutoff):

    assert x.ndim == 2
    N = x.shape[0]
    # D = x.shape[1]

    assert N % 3 == 0

    params = prepare_system_params(x, sigma_scale=p_scale)

    scales = []
    exclusion_idxs = []
    for i in range(N // 3):
        O_idx = i * 3 + 0
        H1_idx = i * 3 + 1
        H2_idx = i * 3 + 2
        exclusion_idxs.append([O_idx, H1_idx])  # 1-2
        exclusion_idxs.append([O_idx, H2_idx])  # 1-2
        exclusion_idxs.append([H1_idx, H2_idx])  # 1-3

        scales.append([1.0, 1.0])
        scales.append([1.0, 1.0])
        scales.append([np.random.rand(), np.random.rand()])

    scales = np.array(scales, dtype=np.float64)
    exclusion_idxs = np.array(exclusion_idxs, dtype=np.int32)

    beta = 2.0

    test_potential = potentials.Nonbonded(exclusion_idxs, scales, lambda_plane_idxs, lambda_offset_idxs, beta, cutoff)

    charge_rescale_mask, lj_rescale_mask = nonbonded.convert_exclusions_to_rescale_masks(exclusion_idxs, scales, N)

    ref_total_energy = functools.partial(
        nonbonded.nonbonded_v3,
        charge_rescale_mask=charge_rescale_mask,
        lj_rescale_mask=lj_rescale_mask,
        beta=beta,
        cutoff=cutoff,
        lambda_plane_idxs=lambda_plane_idxs,
        lambda_offset_idxs=lambda_offset_idxs,
        runtime_validate=False,
    )

    return params, ref_total_energy, test_potential


def prepare_nb_system(x, E, lambda_plane_idxs, lambda_offset_idxs, p_scale, cutoff):  # number of exclusions

    assert x.ndim == 2
    N = x.shape[0]
    # D = x.shape[1]

    params = prepare_system_params(x, sigma_scale=p_scale)

    atom_idxs = np.arange(N)

    exclusion_idxs = np.random.choice(atom_idxs, size=(E, 2), replace=False)
    exclusion_idxs = np.array(exclusion_idxs, dtype=np.int32).reshape(-1, 2)

    scales = np.stack([np.random.rand(E), np.random.rand(E)], axis=1)

    beta = 2.0

    test_potential = potentials.Nonbonded(exclusion_idxs, scales, lambda_plane_idxs, lambda_offset_idxs, beta, cutoff)

    charge_rescale_mask, lj_rescale_mask = nonbonded.convert_exclusions_to_rescale_masks(exclusion_idxs, scales, N)

    ref_total_energy = functools.partial(
        nonbonded.nonbonded_v3,
        charge_rescale_mask=charge_rescale_mask,
        lj_rescale_mask=lj_rescale_mask,
        beta=beta,
        cutoff=cutoff,
        lambda_plane_idxs=lambda_plane_idxs,
        lambda_offset_idxs=lambda_offset_idxs,
        runtime_validate=False,
    )

    return params, ref_total_energy, test_potential


def prepare_restraints(x, B, precision):

    assert x.ndim == 2
    N = x.shape[0]
    # D = x.shape[1]

    atom_idxs = np.arange(N)

    params = np.random.randn(B, 3).astype(np.float64)

    bond_idxs = []
    for _ in range(B):
        bond_idxs.append(np.random.choice(atom_idxs, size=2, replace=False))
    bond_idxs = np.array(bond_idxs, dtype=np.int32)

    lambda_flags = np.random.randint(0, 2, size=(B,)).astype(np.int32)

    custom_restraint = potentials.Restraint(bond_idxs, params, lambda_flags, precision=precision)
    restraint_fn = functools.partial(bonded.restraint, box=None, lamb_flags=lambda_flags, bond_idxs=bond_idxs)

    return (params, restraint_fn), custom_restraint


def prepare_bonded_system(x, B, A, T, precision):

    assert x.ndim == 2
    N = x.shape[0]
    # D = x.shape[1]

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
    custom_bonded = potentials.HarmonicBond(bond_idxs, bond_params, precision=precision)
    harmonic_bond_fn = functools.partial(bonded.harmonic_bond, box=None, bond_idxs=bond_idxs)

    # custom_angles = potentials.HarmonicAngle(angle_idxs, angle_param_idxs, precision=precision)
    # harmonic_angle_fn = functools.partial(bonded.harmonic_angle, box=None, angle_idxs=angle_idxs, param_idxs=angle_param_idxs)

    # custom_torsions = potentials.PeriodicTorsion(torsion_idxs, torsion_param_idxs, precision=precision)
    # periodic_torsion_fn = functools.partial(bonded.periodic_torsion, box=None, torsion_idxs=torsion_idxs, param_idxs=torsion_param_idxs)

    return (bond_params, harmonic_bond_fn), custom_bonded
    # return params, [harmonic_bond_fn, harmonic_angle_fn, periodic_torsion_fn], [custom_bonded, custom_angles, custom_torsions]


def hilbert_sort(conf, D):
    hc = HilbertCurve(64, D)
    int_confs = (conf * 1000).astype(np.int64)
    dists = []
    for xyz in int_confs.tolist():
        dist = hc.distance_from_coordinates(xyz)
        dists.append(dist)
    perm = np.argsort(dists)
    # np.random.shuffle(perm)
    return perm


class GradientTest(unittest.TestCase):
    def get_random_coords(self, N, D):
        x = np.random.rand(N, D).astype(dtype=np.float64)
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
        assert np.all(np.isfinite(truth))
        assert np.all(np.isfinite(test))
        assert np.array(truth).shape == np.array(test).shape

        norms = np.linalg.norm(truth, axis=-1, keepdims=True)
        norms = np.where(norms < 1.0, 1.0, norms)
        errors = (truth - test) / norms

        # print(errors)
        max_error = np.amax(np.abs(errors))
        max_error_arg = np.argmax(errors) // truth.shape[1]

        errors = np.abs(errors) > rtol

        # mean_error = np.mean(np.abs(errors).reshape(-1))
        # std_error = np.std(errors.reshape(-1))
        # print("max relative error", max_error, "rtol", rtol, norms[max_error_arg], "mean error", mean_error, "std error", std_error)
        if np.sum(errors) > 0:
            print("FATAL: max relative error", max_error, truth[max_error_arg], test[max_error_arg])
            assert 0

    def compare_forces(
        self,
        x: NDArray,
        params: NDArray,
        box: NDArray,
        lambdas: List[float],
        ref_potential,
        test_potential,
        rtol: float,
        precision,
        atol: float = 1e-8,
        benchmark: bool = False,
    ):
        """
        Compares the forces between a reference and a test potential.


        Note
        ----
        Preferable to pass a list of lambdas to this function than run this function
        repeatedly, as this function constructs an unbound impl for the test_potential
        which can be expensive relative to the time it takes to compute the forces/energies/etc.

        """
        test_impl = test_potential.unbound_impl(precision)

        x = (x.astype(np.float32)).astype(np.float64)
        params = (params.astype(np.float32)).astype(np.float64)

        assert x.ndim == 2
        # N = x.shape[0]
        # D = x.shape[1]

        assert x.dtype == np.float64
        assert params.dtype == np.float64

        for lamb in lambdas:
            ref_u = ref_potential(x, params, box, lamb)
            grad_fn = jax.grad(ref_potential, argnums=(0, 1, 3))
            ref_du_dx, ref_du_dp, ref_du_dl = grad_fn(x, params, box, lamb)
            for combo in itertools.product([False, True], repeat=4):

                (compute_du_dx, compute_du_dp, compute_du_dl, compute_u) = combo

                # do each computation twice to check determinism
                test_du_dx, test_du_dp, test_du_dl, test_u = test_impl.execute_selective(
                    x, params, box, lamb, compute_du_dx, compute_du_dp, compute_du_dl, compute_u
                )
                if compute_u:
                    np.testing.assert_allclose(ref_u, test_u, rtol=rtol, atol=atol)
                if compute_du_dx:
                    self.assert_equal_vectors(np.array(ref_du_dx), np.array(test_du_dx), rtol)
                if compute_du_dl:
                    np.testing.assert_allclose(ref_du_dl, test_du_dl, rtol=rtol)
                if compute_du_dp:
                    np.testing.assert_allclose(ref_du_dp, test_du_dp, rtol=rtol, atol=atol)

                test_du_dx_2, test_du_dp_2, test_du_dl_2, test_u_2 = test_impl.execute_selective(
                    x, params, box, lamb, compute_du_dx, compute_du_dp, compute_du_dl, compute_u
                )

            np.testing.assert_array_equal(test_du_dx, test_du_dx_2)
            np.testing.assert_array_equal(test_du_dl, test_du_dl_2)
            np.testing.assert_array_equal(test_u, test_u_2)

            if isinstance(test_potential, potentials.Nonbonded):
                np.testing.assert_array_equal(test_du_dp, test_du_dp_2)
