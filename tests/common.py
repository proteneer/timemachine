import contextlib
import itertools
import os
import unittest
from collections.abc import Iterator
from dataclasses import dataclass, replace
from tempfile import TemporaryDirectory
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import AllChem

from timemachine.constants import DEFAULT_TEMP, ONE_4PI_EPS0
from timemachine.fe import interpolate, rbfe
from timemachine.fe.aligned_potential import interpolate_w_coord
from timemachine.fe.free_energy import HostConfig
from timemachine.fe.single_topology import AtomMapFlags
from timemachine.fe.utils import set_mol_name
from timemachine.ff import Forcefield
from timemachine.ff.handlers import nonbonded
from timemachine.lib import custom_ops
from timemachine.md.builders import build_protein_system
from timemachine.potentials import Nonbonded
from timemachine.potentials.potential import GpuImplWrapper
from timemachine.potentials.types import PotentialFxn
from timemachine.utils import path_to_internal_file

HILBERT_GRID_DIM = 128
# Directory to write files to that will be stored as artifacts in CI.
ARTIFACT_DIR_NAME = os.environ.get("CI_ARTIFACT_DIR", "pytest-artifacts")


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


from timemachine.fe.single_topology import SingleTopology
from timemachine.fe.utils import get_romol_conf


def prepare_single_topology_initial_state(mol_a, mol_b, core, ff, host_config: Optional[HostConfig], lamb: float = 0.1):
    temperature = DEFAULT_TEMP
    # if host_config is not None:
    # host = rbfe.setup_optimized_host(mol_a, mol_b, ff, host_config)
    is_complex_leg = host_config is not None and len(host_config.conf) != host_config.num_water_atoms

    if host_config:
        host_config = rbfe.setup_optimized_host(mol_a, mol_b, ff, host_config)
        single_topology = SingleTopology.from_mols_with_host(mol_a, mol_b, core, host_config, ff)
        num_host_atoms = len(host_config.masses)
        ligand_idxs = np.arange(0, single_topology.get_num_atoms(), dtype=np.int32) + num_host_atoms
        protein_idxs = np.arange(0, num_host_atoms, dtype=np.int32)
        x_a = np.vstack([host_config.conf, get_romol_conf(mol_a)])
        x_b = np.vstack([host_config.conf, get_romol_conf(mol_b)])
        box = host_config.box
    else:
        single_topology = SingleTopology.from_mols(mol_a, mol_b, core, ff)
        ligand_idxs = np.arange(0, single_topology.get_num_atoms(), dtype=np.int32)
        protein_idxs = np.zeros(0, dtype=np.int32)
        x_a = get_romol_conf(mol_a)
        x_b = get_romol_conf(mol_b)
        box = np.eye(3, dtype=np.float64) * 10  # make a large 10x10x10nm box

    seed = 2022
    min_cutoff = 0.7 if is_complex_leg else None

    initial_state = rbfe.setup_initial_states(
        single_topology,
        temperature,
        [lamb],
        seed,
        min_cutoff,
        x_a,
        x_b,
        box,
        ligand_idxs,
        protein_idxs,
    )[0]
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
        assert np.abs(ref_u) < np.iinfo(np.int64).max / custom_ops.FIXED_EXPONENT, (
            "System is invalid, GPU platform unable to represent energies"
        )
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
    env: Forcefield  # ligand-environment charge terms/LJ terms scaled
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

    assert ff_ref.q_handle is not None
    protein_smirks = ff_ref.q_handle.smirks
    protein_params = np.ones((len(protein_smirks),)) * 0.5

    ff_intra = Forcefield.load_default()
    assert ff_intra.q_handle_intra is not None
    assert ff_intra.lj_handle_intra is not None
    ff_intra.q_handle_intra.params *= Q_SCALE
    ff_intra.lj_handle_intra.params[:, SIG_IDX] *= SIG_SCALE
    ff_intra.lj_handle_intra.params[:, EPS_IDX] *= EPS_SCALE

    ff_env = Forcefield.load_default()
    assert ff_env.q_handle is not None
    assert ff_env.lj_handle is not None
    ff_env.q_handle.params *= Q_SCALE
    ff_env.lj_handle.params[:, SIG_IDX] *= SIG_SCALE
    ff_env.lj_handle.params[:, EPS_IDX] *= EPS_SCALE
    env_bcc_handle = nonbonded.EnvironmentBCCPartialHandler(protein_smirks, protein_params, None)
    ff_env = replace(ff_env, env_bcc_handle=env_bcc_handle)
    assert ff_env.env_bcc_handle is not None

    ff_scaled = Forcefield.load_default()
    assert ff_scaled.q_handle is not None
    assert ff_scaled.q_handle_intra is not None
    assert ff_scaled.lj_handle is not None
    assert ff_scaled.lj_handle_intra is not None
    ff_scaled.q_handle.params *= Q_SCALE
    ff_scaled.q_handle_intra.params *= Q_SCALE
    ff_scaled.lj_handle.params[:, SIG_IDX] *= SIG_SCALE
    ff_scaled.lj_handle.params[:, EPS_IDX] *= EPS_SCALE
    ff_scaled.lj_handle_intra.params[:, SIG_IDX] *= SIG_SCALE
    ff_scaled.lj_handle_intra.params[:, EPS_IDX] *= EPS_SCALE
    env_bcc_handle = nonbonded.EnvironmentBCCPartialHandler(protein_smirks, protein_params, None)
    ff_scaled = replace(ff_scaled, env_bcc_handle=env_bcc_handle)
    assert ff_scaled.env_bcc_handle is not None
    return SplitForcefield(ff_ref, ff_intra, ff_env, ff_scaled)


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

    with path_to_internal_file("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as path_to_pdb:
        host_config = build_protein_system(str(path_to_pdb), ffs.ref.protein_ff, ffs.ref.water_ff)
        host_config.box += np.diag([0.1, 0.1, 0.1])

    host_conf = host_config.conf
    box = host_config.box
    complex_top = host_config.omm_topology
    num_water_atoms = host_config.num_water_atoms
    host_system = host_config.host_system

    coords0 = np.concatenate([host_conf, ligand_conf])
    num_protein_atoms = host_conf.shape[0] - num_water_atoms
    protein_idxs = np.arange(num_protein_atoms, dtype=np.int32)
    water_idxs = np.arange(num_water_atoms, dtype=np.int32) + num_protein_atoms
    num_host_atoms = host_conf.shape[0]
    ligand_idxs += num_host_atoms  # shift for the host

    n_lambdas = 3
    for lamb in np.linspace(0, 1, n_lambdas):
        """
        Note: Notation here is interaction type _ scaled term
        interaction type:
            LL - ligand-ligand intramolecular interactions
            PL - protein-ligand interactions
            WL - water-ligand interactions
            LE - ligand-environment interactions
            sum - full NB potential

        scaled term:
            ref - ref ff
            intra - ligand-ligand intramolecular parameters are scaled
            env - ligand-environment interaction parameters are scaled
        """

        # Compute the grads, potential with the ref ff
        LL_grad_ref, LL_u_ref = compute_intra_grad_u(
            ffs.ref, precision, ligand_conf, box, lamb, num_water_atoms, num_host_atoms
        )
        sum_grad_ref, sum_u_ref = compute_ref_grad_u(
            ffs.ref, precision, coords0, box, lamb, num_water_atoms, host_system, complex_top
        )
        PL_grad_ref, PL_u_ref = compute_ixn_grad_u(
            ffs.ref,
            precision,
            coords0,
            box,
            lamb,
            num_water_atoms,
            host_system,
            water_idxs,
            ligand_idxs,
            protein_idxs,
            complex_top,
            is_solvent=False,
        )
        WL_grad_ref, WL_u_ref = compute_ixn_grad_u(
            ffs.ref,
            precision,
            coords0,
            box,
            lamb,
            num_water_atoms,
            host_system,
            water_idxs,
            ligand_idxs,
            protein_idxs,
            complex_top,
            is_solvent=True,
        )

        # Should be the same as the new code with the orig ff
        sum_grad_new, sum_u_new = compute_new_grad_u(
            ffs.ref, precision, coords0, box, lamb, num_water_atoms, host_system, complex_top
        )

        np.testing.assert_allclose(sum_u_ref, sum_u_new, rtol=rtol, atol=atol)
        np.testing.assert_allclose(sum_grad_ref, sum_grad_new, rtol=rtol, atol=atol)

        # Compute the grads, potential with the intramolecular terms scaled
        sum_grad_intra, sum_u_intra = compute_new_grad_u(
            ffs.intra, precision, coords0, box, lamb, num_water_atoms, host_system, complex_top
        )
        LL_grad_intra, LL_u_intra = compute_intra_grad_u(
            ffs.intra, precision, ligand_conf, box, lamb, num_water_atoms, num_host_atoms
        )

        # U_intra = U_sum_ref - LL_ref + LL_intra
        expected_u = sum_u_ref - LL_u_ref + LL_u_intra
        expected_grad = sum_grad_ref - LL_grad_ref + LL_grad_intra

        np.testing.assert_allclose(expected_u, sum_u_intra, rtol=rtol, atol=atol)
        np.testing.assert_allclose(expected_grad, sum_grad_intra, rtol=rtol, atol=atol)

        # Compute the grads, potential with the ligand-env terms scaled
        sum_grad_prot, sum_u_prot = compute_new_grad_u(
            ffs.env, precision, coords0, box, lamb, num_water_atoms, host_system, complex_top
        )
        PL_grad_env, PL_u_env = compute_ixn_grad_u(
            ffs.env,
            precision,
            coords0,
            box,
            lamb,
            num_water_atoms,
            host_system,
            water_idxs,
            ligand_idxs,
            protein_idxs,
            complex_top,
            is_solvent=False,
        )
        WL_grad_env, WL_u_env = compute_ixn_grad_u(
            ffs.env,
            precision,
            coords0,
            box,
            lamb,
            num_water_atoms,
            host_system,
            water_idxs,
            ligand_idxs,
            protein_idxs,
            complex_top,
            is_solvent=True,
        )

        LE_u_ref = WL_u_ref + PL_u_ref
        LE_u_env = WL_u_env + PL_u_env

        LE_grad_ref = WL_grad_ref + PL_grad_ref
        LE_grad_env = WL_grad_env + PL_grad_env

        # U_prot = U_sum_ref - LE_u_ref + LE_env
        expected_u = sum_u_ref - LE_u_ref + LE_u_env
        expected_grad = sum_grad_ref - LE_grad_ref + LE_grad_env

        np.testing.assert_allclose(expected_u, sum_u_prot, rtol=rtol, atol=atol)
        np.testing.assert_allclose(expected_grad, sum_grad_prot, rtol=rtol, atol=atol)


def ligand_from_smiles(smiles: str, seed: int = 2024) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    set_mol_name(mol, smiles)
    return mol


def get_alchemical_guest_params(
    mol_a, mol_b, atom_map_mixin, q_handle, lj_handle, lamb: float, cutoff: float
) -> jax.Array:
    """
    Return an array containing the guest_charges, guest_sigmas, guest_epsilons, guest_w_coords
    for the guest at a given lambda.
    """
    guest_charges = []
    guest_sigmas = []
    guest_epsilons = []
    guest_w_coords = []

    # generate charges and lj parameters for each guest
    guest_a_q = q_handle.parameterize(mol_a)
    guest_a_lj = lj_handle.parameterize(mol_a)

    guest_b_q = q_handle.parameterize(mol_b)
    guest_b_lj = lj_handle.parameterize(mol_b)

    for idx, membership in enumerate(atom_map_mixin.c_flags):
        if membership == AtomMapFlags.CORE:  # core atom
            a_idx = atom_map_mixin.c_to_a[idx]
            b_idx = atom_map_mixin.c_to_b[idx]

            q = interpolate.linear_interpolation(guest_a_q[a_idx], guest_b_q[b_idx], lamb)
            sig = interpolate.linear_interpolation(guest_a_lj[a_idx, 0], guest_b_lj[b_idx, 0], lamb)
            eps = interpolate.linear_interpolation(guest_a_lj[a_idx, 1], guest_b_lj[b_idx, 1], lamb)

            # interpolate charges when in common-core
            # q = (1 - lamb) * guest_a_q[a_idx] + lamb * guest_b_q[b_idx]
            # sig = (1 - lamb) * guest_a_lj[a_idx, 0] + lamb * guest_b_lj[b_idx, 0]
            # eps = (1 - lamb) * guest_a_lj[a_idx, 1] + lamb * guest_b_lj[b_idx, 1]

            # fixed at w = 0
            w = 0.0

        elif membership == AtomMapFlags.MOL_A:  # dummy_A
            a_idx = atom_map_mixin.c_to_a[idx]
            q = guest_a_q[a_idx]
            sig = guest_a_lj[a_idx, 0]
            eps = guest_a_lj[a_idx, 1]

            # Decouple dummy group A as lambda goes from 0 to 1
            w = interpolate_w_coord(0.0, cutoff, lamb)

        elif membership == AtomMapFlags.MOL_B:  # dummy_B
            b_idx = atom_map_mixin.c_to_b[idx]
            q = guest_b_q[b_idx]
            sig = guest_b_lj[b_idx, 0]
            eps = guest_b_lj[b_idx, 1]

            # Couple dummy group B as lambda goes from 0 to 1
            # NOTE: this is only for host-guest nonbonded ixns (there is no clash between A and B at lambda = 0.5)
            w = interpolate_w_coord(cutoff, 0.0, lamb)
        else:
            assert 0

        guest_charges.append(q)
        guest_sigmas.append(sig)
        guest_epsilons.append(eps)
        guest_w_coords.append(w)

    return jnp.stack(jnp.array([guest_charges, guest_sigmas, guest_epsilons, guest_w_coords]), axis=1)
