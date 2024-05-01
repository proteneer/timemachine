import contextlib
import itertools
import os
import unittest
from collections.abc import Iterator
from dataclasses import dataclass
from importlib import resources
from tempfile import TemporaryDirectory
from typing import Optional

import jax
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve
from numpy.typing import NDArray

from timemachine.constants import DEFAULT_TEMP, ONE_4PI_EPS0
from timemachine.fe import rbfe
from timemachine.fe.free_energy import HostConfig
from timemachine.fe.single_topology import SingleTopology
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.lib import custom_ops
from timemachine.md.builders import build_protein_system
from timemachine.potentials import Nonbonded
from timemachine.potentials.potential import GpuImplWrapper
from timemachine.potentials.types import PotentialFxn

HILBERT_GRID_DIM = 128


@contextlib.contextmanager
def temporary_working_dir():
    init_dir = os.getcwd()
    with TemporaryDirectory() as temp:
        try:
            os.chdir(temp)
            yield temp
        finally:
            os.chdir(init_dir)


def convert_quaternion_for_scipy(quat: NDArray) -> NDArray:
    """Scipy has the convention of (x, y, z, w) which is different than the wikipedia definition, swap ordering to verify using scipy.

    References
    ----------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html
    """
    return np.append(quat[1:], [quat[0]])


def fixed_overflowed(a):
    """Refer to timemachine/cpp/src/kernels/k_fixed_point.cuh::FLOAT_TO_FIXED_ENERGY for documentation on how we handle energies and overflows"""
    converted_a = np.int64(np.uint64(a))
    assert converted_a != np.iinfo(np.int64).min, "Unexpected value for fixed point energy"
    return converted_a == np.iinfo(np.int64).max


def prepare_single_topology_initial_state(st: SingleTopology, host_config: Optional[HostConfig], lamb: float = 0.1):
    temperature = DEFAULT_TEMP
    host = None
    if host_config is not None:
        host = rbfe.setup_optimized_host(st, host_config)
    initial_state = rbfe.setup_initial_states(st, host, temperature, [lamb], seed=2022)[0]
    return initial_state


def prepare_system_params(x: NDArray, cutoff: float, sigma_scale: float = 5.0) -> NDArray:
    """
    Prepares random parameters given a set of coordinates. The parameters are adjusted to be the correct
    order of magnitude.

    Parameters
    ----------

    x: Numpy array of Coordinates

    sigma_scale: Factor to scale down sigma values by

    Returns
    -------
    (N, 4) np.ndarray containing charges, sigmas, epsilons and w coordinates respectively.
    """
    assert x.ndim == 2
    N = x.shape[0]

    params = np.stack(
        [
            (np.random.rand(N).astype(np.float64) - 0.5) * np.sqrt(ONE_4PI_EPS0),  # q
            np.random.rand(N).astype(np.float64) / sigma_scale,  # sig
            np.random.rand(N).astype(np.float64),  # eps
            (2 * np.random.rand(N).astype(np.float64) - 1) * cutoff,  # w
        ],
        axis=1,
    )

    params[:, 1] = params[:, 1] / 2
    params[:, 2] = np.sqrt(params[:, 2])
    return params


def prepare_water_system(x, p_scale, cutoff):
    assert x.ndim == 2
    N = x.shape[0]
    # D = x.shape[1]

    assert N % 3 == 0

    params = prepare_system_params(x, cutoff, sigma_scale=p_scale)

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

    potential = Nonbonded(N, exclusion_idxs, scales, beta, cutoff)

    return params, potential


def assert_energy_arrays_match(
    reference_energies: NDArray, test_energies: NDArray, threshold: float = 1e8, rtol: float = 1e-8, atol: float = 1e-8
):
    """When comparing the reference platform (jax) to the cuda platform we can get Nans beyond a certain
    value and these NaNs will cause issues when comparing energies (or log weights). This method compares all of the values that aren't
    nans and makes sure that all the cases where the values are Nan are very large in the reference energies.

    Handles any values computed in fixed point where overflows can happen. Typically energies, but can also be log weights

    Parameters
    ----------

    reference_energies: np.ndarray
        Energies/log weights from the reference platform, method is not trustworthy otherwise

    test_energies: np.ndarray
        Energies/log weights from the C++ platform

    threshold: float
        Threshold to use, defaults to 1e8 which is empirically selected

    rtol: float
        Relative tolerance, defaults to 1e-8

    atol: float
        Absolute tolerance, defaults to 1e-8

    Raises
    ------
    AssertionError - Don't match
    """
    reference_energies = reference_energies.copy()
    test_energies = test_energies.copy()
    assert reference_energies.shape == test_energies.shape
    reference_energies = reference_energies.reshape(-1)
    test_energies = test_energies.reshape(-1)
    comparable_energies = np.argwhere(np.abs(reference_energies) < threshold)
    large_energy_indices = np.argwhere(np.abs(reference_energies) >= threshold)
    np.testing.assert_allclose(
        reference_energies[comparable_energies], test_energies[comparable_energies], rtol=rtol, atol=atol
    )
    # Pull out nans, as they are effectively greater than the threshold
    non_nan_idx = np.isfinite(test_energies[large_energy_indices])
    # Large energies are not reliable, so beyond the threshold we simply verify that both the reference and test both exceed the threshold
    assert np.all(np.abs(test_energies[large_energy_indices][non_nan_idx]) >= threshold)


def prepare_nb_system(
    x,
    E,  # number of exclusions
    p_scale,
    cutoff,
):
    assert x.ndim == 2
    N = x.shape[0]
    # D = x.shape[1]

    params = prepare_system_params(x, cutoff, sigma_scale=p_scale)

    atom_idxs = np.arange(N)

    exclusion_idxs = np.random.choice(atom_idxs, size=(E, 2), replace=False)
    exclusion_idxs = np.array(exclusion_idxs, dtype=np.int32).reshape(-1, 2)

    scales = np.stack([np.random.rand(E), np.random.rand(E)], axis=1)

    beta = 2.0

    potential = Nonbonded(N, exclusion_idxs, scales, beta, cutoff)

    return params, potential


def hilbert_sort(conf, box):
    hc = HilbertCurve(HILBERT_GRID_DIM, conf.shape[1])

    box_diag = np.diagonal(box)
    # hc assumes non-negative coordinates, re-image coordinates into home box
    conf = conf - box_diag * np.floor(conf / box_diag)
    assert (conf >= 0.0).all()

    int_confs = (conf * 1000).astype(np.int64)
    dists = []
    for xyz in int_confs.tolist():
        dist = hc.distance_from_coordinates(xyz)
        dists.append(dist)
    perm = np.argsort(dists, kind="stable")
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
        ref_potential: PotentialFxn,
        test_potential: GpuImplWrapper,
        rtol: float,
        atol: float = 1e-8,
        du_dp_rtol: Optional[float] = None,
        du_dp_atol: Optional[float] = None,
    ):
        """Compares the forces between a reference and a test potential."""
        x = (x.astype(np.float32)).astype(np.float64)
        if du_dp_rtol is None:
            du_dp_rtol = rtol * 10
        if du_dp_atol is None:
            du_dp_atol = atol * 10

        assert x.ndim == 2
        # N = x.shape[0]
        # D = x.shape[1]

        assert x.dtype == np.float64

        params = (params.astype(np.float32)).astype(np.float64)
        assert params.dtype == np.float64
        ref_u = ref_potential(x, params, box)
        assert (
            np.abs(ref_u) < np.iinfo(np.int64).max / custom_ops.FIXED_EXPONENT
        ), "System is invalid, GPU platform unable to represent energies"
        grad_fn = jax.grad(ref_potential, argnums=(0, 1))
        ref_du_dx, ref_du_dp = grad_fn(x, params, box)

        for combo in itertools.product([False, True], repeat=3):
            compute_du_dx, compute_du_dp, compute_u = combo

            # do each computation twice to check determinism
            test_du_dx, test_du_dp, test_u = test_potential.unbound_impl.execute(
                x, params, box, compute_du_dx, compute_du_dp, compute_u
            )
            if compute_u:
                np.testing.assert_allclose(ref_u, test_u, rtol=rtol, atol=atol)
            else:
                assert test_u is None
            if compute_du_dx:
                self.assert_equal_vectors(np.array(ref_du_dx), np.array(test_du_dx), rtol)
            else:
                assert test_du_dx is None
            if compute_du_dp:
                np.testing.assert_allclose(ref_du_dp, test_du_dp, rtol=du_dp_rtol, atol=du_dp_atol)
            else:
                assert test_du_dp is None

            test_du_dx_2, test_du_dp_2, test_u_2 = test_potential.unbound_impl.execute(
                x, params, box, compute_du_dx, compute_du_dp, compute_u
            )
            np.testing.assert_array_equal(test_du_dx, test_du_dx_2)
            np.testing.assert_array_equal(test_u, test_u_2)
            np.testing.assert_array_equal(test_du_dp, test_du_dp_2)

    def assert_differentiable_interface_consistency(
        self, x: NDArray, params: NDArray, box: NDArray, gpu_impl: GpuImplWrapper
    ):
        """Check that energy and derivatives computed using the JAX differentiable interface are consistent with values
        returned by execute"""
        ref_du_dx, ref_du_dp, ref_u = gpu_impl.unbound_impl.execute(x, params, box, True, True, True)
        test_u, (test_du_dx, test_du_dp) = jax.value_and_grad(gpu_impl, (0, 1))(x, params, box)
        assert ref_u == test_u
        np.testing.assert_array_equal(test_du_dx, ref_du_dx)
        np.testing.assert_array_equal(test_du_dp, ref_du_dp)


def gen_nonbonded_params_with_4d_offsets(
    rng: np.random.Generator, params, w_max: float, w_min: Optional[float] = None
) -> Iterator[NDArray]:
    if w_min is None:
        w_min = -w_max

    num_atoms, _ = params.shape

    def params_with_w_coords(w_coords):
        params_ = np.array(params)
        params_[:, 3] = w_coords
        return params

    # all zero
    yield params_with_w_coords(0.0)

    # all w_max
    yield params_with_w_coords(w_max)

    # half zero, half w_max
    w_coords = np.zeros(num_atoms)
    w_coords[-num_atoms // 2 :] = w_max
    yield params_with_w_coords(w_coords)

    # random uniform in [w_min, w_max]
    w_coords = rng.uniform(w_min, w_max, (num_atoms,))
    yield params_with_w_coords(w_coords)

    # sparse with half zero
    zero_idxs = rng.choice(num_atoms, (num_atoms // 2,), replace=False)
    w_coords[zero_idxs] = 0.0
    yield params_with_w_coords(w_coords)


@dataclass
class SplitForcefield:
    ref: Forcefield  # ref ff
    intra: Forcefield  # intermolecular charge terms/LJ terms scaled
    solv: Forcefield  # water-ligand charge terms/LJ terms scaled
    prot: Forcefield  # protein-ligand charge terms/LJ terms scaled
    scaled: Forcefield  # all NB terms scaled


def load_split_forcefields() -> SplitForcefield:
    """
    Returns:
        SplitForcefield which contains the ff with various
        terms scaled.
    """
    SIG_IDX, EPS_IDX = 0, 1
    Q_SCALE = 10
    SIG_SCALE = 0.5  # smaller change to prevent overflow
    EPS_SCALE = 2.0

    ff_ref = Forcefield.load_default()

    ff_intra = Forcefield.load_default()
    assert ff_intra.q_handle_intra is not None
    assert ff_intra.lj_handle_intra is not None
    ff_intra.q_handle_intra.params *= Q_SCALE
    ff_intra.lj_handle_intra.params[:, SIG_IDX] *= SIG_SCALE
    ff_intra.lj_handle_intra.params[:, EPS_IDX] *= EPS_SCALE

    ff_solv = Forcefield.load_default()
    assert ff_solv.q_handle_solv is not None
    assert ff_solv.lj_handle_solv is not None
    ff_solv.q_handle_solv.params *= Q_SCALE
    ff_solv.lj_handle_solv.params[:, SIG_IDX] *= SIG_SCALE
    ff_solv.lj_handle_solv.params[:, EPS_IDX] *= EPS_SCALE

    ff_prot = Forcefield.load_default()
    assert ff_prot.q_handle is not None
    assert ff_prot.lj_handle is not None
    ff_prot.q_handle.params *= Q_SCALE
    ff_prot.lj_handle.params[:, SIG_IDX] *= SIG_SCALE
    ff_prot.lj_handle.params[:, EPS_IDX] *= EPS_SCALE

    ff_scaled = Forcefield.load_default()
    assert ff_scaled.q_handle is not None
    assert ff_scaled.q_handle_intra is not None
    assert ff_scaled.q_handle_solv is not None
    assert ff_scaled.lj_handle is not None
    assert ff_scaled.lj_handle_intra is not None
    assert ff_scaled.lj_handle_solv is not None
    ff_scaled.q_handle.params *= Q_SCALE
    ff_scaled.q_handle_intra.params *= Q_SCALE
    ff_scaled.q_handle_solv.params *= Q_SCALE
    ff_scaled.lj_handle.params[:, SIG_IDX] *= SIG_SCALE
    ff_scaled.lj_handle.params[:, EPS_IDX] *= EPS_SCALE
    ff_scaled.lj_handle_intra.params[:, SIG_IDX] *= SIG_SCALE
    ff_scaled.lj_handle_intra.params[:, EPS_IDX] *= EPS_SCALE
    ff_scaled.lj_handle_solv.params[:, SIG_IDX] *= SIG_SCALE
    ff_scaled.lj_handle_solv.params[:, EPS_IDX] *= EPS_SCALE
    return SplitForcefield(ff_ref, ff_intra, ff_solv, ff_prot, ff_scaled)


def check_split_ixns(
    ligand_conf,
    ligand_idxs,
    precision,
    rtol,
    atol,
    compute_ref_grad_u,
    compute_new_grad_u,
    compute_intra_grad_u,
    compute_ixn_grad_u,
):
    ffs = load_split_forcefields()

    with resources.path("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as path_to_pdb:
        complex_system, host_conf, box, _, num_water_atoms = build_protein_system(
            str(path_to_pdb), ffs.ref.protein_ff, ffs.ref.water_ff
        )
        box += np.diag([0.1, 0.1, 0.1])

    coords0 = np.concatenate([host_conf, ligand_conf])
    num_protein_atoms = host_conf.shape[0] - num_water_atoms
    protein_idxs = np.arange(num_protein_atoms, dtype=np.int32)
    water_idxs = np.arange(num_water_atoms, dtype=np.int32) + num_protein_atoms
    num_host_atoms = host_conf.shape[0]
    host_bps, host_masses = openmm_deserializer.deserialize_system(complex_system, cutoff=1.2)
    ligand_idxs += num_host_atoms  # shift for the host

    n_lambdas = 3
    for lamb in np.linspace(0, 1, n_lambdas):
        """
        Note: Notation here is interaction type _ scaled term
        interaction type:
            LL - ligand-ligand intramolecular interactions
            PL - protein-ligand interactions
            WL - water-ligand interactions
            sum - full NB potential

        scaled term:
            ref - ref ff
            intra - ligand-ligand intramolecular parameters are scaled
            prot - protein-ligand interaction parameters are scaled
            solv - water-ligand interaction parameters are scaled
        """

        # Compute the grads, potential with the ref ff
        LL_grad_ref, LL_u_ref = compute_intra_grad_u(
            ffs.ref, precision, ligand_conf, box, lamb, num_water_atoms, num_host_atoms
        )
        sum_grad_ref, sum_u_ref = compute_ref_grad_u(ffs.ref, precision, coords0, box, lamb, num_water_atoms, host_bps)
        PL_grad_ref, PL_u_ref = compute_ixn_grad_u(
            ffs.ref,
            precision,
            coords0,
            box,
            lamb,
            num_water_atoms,
            host_bps,
            water_idxs,
            ligand_idxs,
            protein_idxs,
            is_solvent=False,
        )
        WL_grad_ref, WL_u_ref = compute_ixn_grad_u(
            ffs.ref,
            precision,
            coords0,
            box,
            lamb,
            num_water_atoms,
            host_bps,
            water_idxs,
            ligand_idxs,
            protein_idxs,
            is_solvent=True,
        )

        # Should be the same as the new code with the orig ff
        sum_grad_new, sum_u_new = compute_new_grad_u(ffs.ref, precision, coords0, box, lamb, num_water_atoms, host_bps)

        np.testing.assert_allclose(sum_u_ref, sum_u_new, rtol=rtol, atol=atol)
        np.testing.assert_allclose(sum_grad_ref, sum_grad_new, rtol=rtol, atol=atol)

        # Compute the grads, potential with the intramolecular terms scaled
        sum_grad_intra, sum_u_intra = compute_new_grad_u(
            ffs.intra, precision, coords0, box, lamb, num_water_atoms, host_bps
        )
        LL_grad_intra, LL_u_intra = compute_intra_grad_u(
            ffs.intra, precision, ligand_conf, box, lamb, num_water_atoms, num_host_atoms
        )

        # U_intra = U_sum_ref - LL_ref + LL_intra
        expected_u = sum_u_ref - LL_u_ref + LL_u_intra
        expected_grad = sum_grad_ref - LL_grad_ref + LL_grad_intra

        np.testing.assert_allclose(expected_u, sum_u_intra, rtol=rtol, atol=atol)
        np.testing.assert_allclose(expected_grad, sum_grad_intra, rtol=rtol, atol=atol)

        # Compute the grads, potential with the ligand-water terms scaled
        sum_grad_solv, sum_u_solv = compute_new_grad_u(
            ffs.solv, precision, coords0, box, lamb, num_water_atoms, host_bps
        )
        WL_grad_solv, WL_u_solv = compute_ixn_grad_u(
            ffs.solv,
            precision,
            coords0,
            box,
            lamb,
            num_water_atoms,
            host_bps,
            water_idxs,
            ligand_idxs,
            protein_idxs,
            is_solvent=True,
        )

        # U_solv = U_sum_ref - WL_ref + WL_solv
        expected_u = sum_u_ref - WL_u_ref + WL_u_solv
        expected_grad = sum_grad_ref - WL_grad_ref + WL_grad_solv

        np.testing.assert_allclose(expected_u, sum_u_solv, rtol=rtol, atol=atol)
        np.testing.assert_allclose(expected_grad, sum_grad_solv, rtol=rtol, atol=atol)

        # Compute the grads, potential with the protein-ligand terms scaled
        sum_grad_prot, sum_u_prot = compute_new_grad_u(
            ffs.prot, precision, coords0, box, lamb, num_water_atoms, host_bps
        )
        PL_grad_prot, PL_u_prot = compute_ixn_grad_u(
            ffs.prot,
            precision,
            coords0,
            box,
            lamb,
            num_water_atoms,
            host_bps,
            water_idxs,
            ligand_idxs,
            protein_idxs,
            is_solvent=False,
        )

        # U_prot = U_sum_ref - PL_ref + PL_prot
        expected_u = sum_u_ref - PL_u_ref + PL_u_prot
        expected_grad = sum_grad_ref - PL_grad_ref + PL_grad_prot

        np.testing.assert_allclose(expected_u, sum_u_prot, rtol=rtol, atol=atol)
        np.testing.assert_allclose(expected_grad, sum_grad_prot, rtol=rtol, atol=atol)
