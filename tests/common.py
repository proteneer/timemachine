import contextlib
import functools
import itertools
import os
import unittest
from collections.abc import Iterator
from dataclasses import dataclass
from importlib import resources
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve
from numpy.typing import NDArray
from rdkit import Chem

from timemachine.constants import ONE_4PI_EPS0
from timemachine.fe.utils import read_sdf
from timemachine.ff import Forcefield
from timemachine.lib import potentials
from timemachine.potentials import Nonbonded, bonded
from timemachine.potentials.potential import GpuImplWrapper
from timemachine.potentials.summed import PotentialFxn


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


def get_hif2a_ligands_as_sdf_file(num_mols: int) -> NamedTemporaryFile:
    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        mols = read_sdf(path_to_ligand)
    temp_sdf = NamedTemporaryFile(suffix=".sdf")
    with Chem.SDWriter(temp_sdf.name) as writer:
        for mol in mols:
            writer.write(mol)
    return temp_sdf


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
        ref_potential: PotentialFxn,
        test_potential: GpuImplWrapper,
        rtol: float,
        atol: float = 1e-8,
    ):
        """Compares the forces between a reference and a test potential."""
        x = (x.astype(np.float32)).astype(np.float64)

        assert x.ndim == 2
        # N = x.shape[0]
        # D = x.shape[1]

        assert x.dtype == np.float64

        params = (params.astype(np.float32)).astype(np.float64)
        assert params.dtype == np.float64
        ref_u = ref_potential(x, params, box)
        grad_fn = jax.grad(ref_potential, argnums=(0, 1))
        ref_du_dx, ref_du_dp = grad_fn(x, params, box)

        for combo in itertools.product([False, True], repeat=3):

            compute_du_dx, compute_du_dp, compute_u = combo

            # do each computation twice to check determinism
            test_du_dx, test_du_dp, test_u = test_potential.unbound_impl.execute_selective(
                x, params, box, compute_du_dx, compute_du_dp, compute_u
            )
            if compute_u:
                np.testing.assert_allclose(ref_u, test_u, rtol=rtol, atol=atol)
            if compute_du_dx:
                self.assert_equal_vectors(np.array(ref_du_dx), np.array(test_du_dx), rtol)
            if compute_du_dp:
                np.testing.assert_allclose(ref_du_dp, test_du_dp, rtol=rtol, atol=atol)

            test_du_dx_2, test_du_dp_2, test_u_2 = test_potential.unbound_impl.execute_selective(
                x, params, box, compute_du_dx, compute_du_dp, compute_u
            )

            np.testing.assert_array_equal(test_du_dx, test_du_dx_2)
            np.testing.assert_array_equal(test_u, test_u_2)
            np.testing.assert_array_equal(test_du_dp, test_du_dp_2)

    def assert_differentiable_interface_consistency(
        self, x: NDArray, params: NDArray, box: NDArray, gpu_impl: GpuImplWrapper
    ):
        """Check that energy and derivatives computed using the JAX differentiable interface are consistent with values
        returned by execute_selective"""
        ref_du_dx, ref_du_dp, ref_u = gpu_impl.unbound_impl.execute_selective(x, params, box, True, True, True)
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
        return jnp.asarray(params).at[:, 3].set(w_coords)

    # all zero
    yield params_with_w_coords(0.0)

    # all w_max
    yield params_with_w_coords(w_max)

    # half zero, half w_max
    w_coords = jnp.zeros(num_atoms)
    w_coords = w_coords.at[-num_atoms // 2 :].set(w_max)
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
    scaled: Forcefield  # all charge terms scaled by 10x
    inter_scaled: Forcefield  # intermolecular charge terms scaled by 10x


def load_split_forcefields() -> SplitForcefield:
    """
    Returns:
        OpenFF 2.0.0 ff,
        OpenFF 2.0.0 ff with all charge terms scaled by 10x,
        OpenFF 2.0.0 ff with only intermolecular charge terms scaled by 10x.
    """
    ff_ref = Forcefield.load_from_file("smirnoff_2_0_0_ccc.py")

    ff_scaled = Forcefield.load_from_file("smirnoff_2_0_0_ccc.py")
    ff_scaled.q_handle.params *= 10
    ff_scaled.q_handle_intra.params *= 10

    ff_inter_scaled = Forcefield.load_from_file("smirnoff_2_0_0_ccc.py")
    ff_inter_scaled.q_handle.params *= 10
    return SplitForcefield(ff_ref, ff_scaled, ff_inter_scaled)
