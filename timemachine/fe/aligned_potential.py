from dataclasses import dataclass
from functools import partial
from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from timemachine.constants import NBParamIdx
from timemachine.fe import interpolate
from timemachine.fe.interpolate import pad
from timemachine.fe.lambda_schedule import construct_pre_optimized_relative_lambda_schedule
from timemachine.potentials import (
    ChiralAtomRestraint,
    HarmonicAngle,
    HarmonicBond,
    Nonbonded,
    NonbondedInteractionGroup,
    NonbondedPairListPrecomputed,
    PeriodicTorsion,
)


def interpolate_harmonic_bond_params(src_params, dst_params, lamb, k_min, lambda_min, lambda_max):
    """
    Interpolate harmonic bond parameters using

    1. Log-linear interpolation for force constants*
    2. Linear interpolation for equilibrium bond lengths

    * see note on special case when src_k=0 or dst_k=0 in the docstring of `interpolate_harmonic_force_constant`.

    Parameters
    ----------
    src_params : array-like, float, (2,)
        force constant and equilibrium length at lambda=0

    dst_params : array-like, float, (2,)
        force constant and equilibrium length at lambda=1

    lamb : float
        alchemical parameter

    k_min, lambda_min, lambda_max : float
        see docstring of `interpolate_harmonic_force_constant` for documentation of these parameters

    Returns
    -------
    array, float, (2,)
        interpolated (force constant, equilibrium length)
    """
    src_k, src_x = src_params
    dst_k, dst_x = dst_params

    log_linear_fn = partial(interpolate.log_linear_interpolation, min_value=k_min)
    k = pad(log_linear_fn, src_k, dst_k, lamb, lambda_min, lambda_max)
    x = pad(interpolate.linear_interpolation, src_x, dst_x, lamb, lambda_min, lambda_max)

    return [k, x]


def interpolate_chiral_volume_params(src_params, dst_params, lamb, k_min, lambda_min, lambda_max):
    src_k = src_params
    dst_k = dst_params

    log_linear_fn = partial(interpolate.log_linear_interpolation, min_value=k_min)
    k = pad(log_linear_fn, src_k, dst_k, lamb, lambda_min, lambda_max)

    return [k]


def cyclic_difference(a, b, period):
    """
    Returns the minimum difference between two points, with periodic boundaries.
    I.e. the solution of ::

        (a + x) % period = b % period

    with minimum abs(x).
    """
    d = jnp.fmod(b - a, period)

    def f(d):
        return jnp.where(d <= period / 2, d, d - period)

    return jnp.sign(d) * f(jnp.abs(d))


def interpolate_harmonic_angle_params(src_params, dst_params, lamb, k_min, lambda_min, lambda_max):
    """
    Interpolate harmonic angle parameters using

    1. Log-linear interpolation for force constants*
    2. Shortest-path linear interpolation for equilibrium angles

    * see note on special case when src_k=0 or dst_k=0 in the docstring of `interpolate_harmonic_force_constant`.

    Parameters
    ----------
    src_params : array-like, float, (2,)
        force constant and equilibrium angle at lambda=0

    dst_params : array-like, float, (2,)
        force constant and equilibrium angle at lambda=1

    lamb : float
        alchemical parameter

    k_min, lambda_min, lambda_max : float
        see docstring of `interpolate_harmonic_force_constant` for documentation of these parameters

    Returns
    -------
    array, float, (2,)
        interpolated (force constant, equilibrium phase)
    """

    src_k, src_phase, _ = src_params
    dst_k, dst_phase, _ = dst_params

    log_linear_fn = partial(interpolate.log_linear_interpolation, min_value=k_min)
    k = pad(log_linear_fn, src_k, dst_k, lamb, lambda_min, lambda_max)

    src_phase = src_phase
    dst_phase = src_phase + cyclic_difference(src_phase, dst_phase, period=2 * np.pi)
    phase = pad(interpolate.linear_interpolation, src_phase, dst_phase, lamb, lambda_min, lambda_max)

    # Use a stable functional form with small, finite `eps` for intermediate states only. The value of `eps` for
    # intermedates was chosen to be sufficiently large that no numerical instabilities were observed in testing (even
    # with bond force constants approximately zero), and sufficiently small to have negligible impact on the overlap of
    # the end states with neighboring intermediates.
    eps = jnp.where((lamb == 0.0) | (lamb == 1.0), 0.0, 1e-3)

    return [k, phase, eps]


def interpolate_periodic_torsion_params(src_params, dst_params, lamb, lambda_min, lambda_max):
    """
    Interpolate periodic torsion parameters using

    1. Linear interpolation for force constants*
    2. Linear interpolation for angles, using the shortest path
    3. No interpolation for periodicity (pinned to source value)

    * see note on special case when src_k=0 or dst_k=0 in the docstring of `interpolate_harmonic_force_constant`.

    Parameters
    ----------
    src_params : array-like, float, (2,)
        force constant, equilibrium dihedral angle, and periodicity at lambda=0

    dst_params : array-like, float, (2,)
        force constant and equilibrium dihedral angle, and periodicity at lambda=1

    lamb : float
        alchemical parameter

    lambda_min, lambda_max : float
        see docstring of `interpolate_harmonic_force_constant` for documentation of these parameters

    Returns
    -------
    array, float, (3,)
        interpolated (force constant, equilibrium phase, periodicity)
    """

    src_k, src_phase, src_period = src_params
    dst_k, dst_phase, _ = dst_params

    k = pad(interpolate.linear_interpolation, src_k, dst_k, lamb, lambda_min, lambda_max)

    src_phase = src_phase
    dst_phase = src_phase + cyclic_difference(src_phase, dst_phase, period=2 * np.pi)
    phase = pad(interpolate.linear_interpolation, src_phase, dst_phase, lamb, lambda_min, lambda_max)

    return [k, phase, src_period]


def interpolate_w_coord(w0: float | jax.Array, w1: float | jax.Array, lamb: float):
    """Interpolate 4D coordinate using schedule optimized for RBFE calculations.

    Parameters
    ----------
    w0, w1 : float
        w coordinates at lambda = 0 and 1 respectively

    lamb : float
        alchemical parameter
    """
    lambdas = construct_pre_optimized_relative_lambda_schedule(None)
    x = jnp.linspace(0.0, 1.0, len(lambdas))
    return jnp.where(
        w0 < w1,
        interpolate.linear_interpolation(w0, w1, jnp.interp(lamb, x, lambdas)),
        interpolate.linear_interpolation(w1, w0, jnp.interp(1.0 - lamb, x, lambdas)),
    )


batch_interpolate_harmonic_bond_params = jax.jit(
    jax.vmap(interpolate_harmonic_bond_params, in_axes=(0, 0, None, None, 0, 0))
)
batch_interpolate_harmonic_angle_params = jax.jit(
    jax.vmap(interpolate_harmonic_angle_params, in_axes=(0, 0, None, None, 0, 0))
)
batch_interpolate_periodic_torsion_params = jax.jit(
    jax.vmap(interpolate_periodic_torsion_params, in_axes=(0, 0, None, 0, 0))
)
batch_interpolate_chiral_atom_params = jax.jit(
    jax.vmap(interpolate_chiral_volume_params, in_axes=(0, 0, None, None, 0, 0))
)


@jax.jit
def batch_interpolate_nonbonded_pair_list_params(
    cutoff,
    src_params,
    dst_params,
    lamb: float,
):
    """
    Interpolate nonbonded pairlists between two systems. This function sets w coordinates to cutoff
    for dummy atoms, and linearly interpolates charge, sigma, and epsilon parameters.

    Parameters
    ----------
    cutoff: float
        nonbonded cutoff

    src_params: NDArray
        Array of P x 4-tuples (q,s,e,w) for the source system

    dst_params: NDArray
        Array of P x 4-tuples (q,s,e,w) for the destination system

    lamb: float
        scalar between 0 and 1

    Returns
    -------
    NDArray
        P x 4 array of interpolated pairs.

    """
    src_qlj, src_w = src_params[:, : NBParamIdx.W_IDX], src_params[:, NBParamIdx.W_IDX]
    dst_qlj, dst_w = dst_params[:, : NBParamIdx.W_IDX], dst_params[:, NBParamIdx.W_IDX]

    is_excluded_src = jnp.all(src_qlj == 0.0, axis=1, keepdims=True)
    is_excluded_dst = jnp.all(dst_qlj == 0.0, axis=1, keepdims=True)

    # parameters for pairs that do not interact in the src state
    w = interpolate_w_coord(cutoff, dst_w, lamb)
    pair_params_excluded_src = jnp.concatenate((dst_qlj, w[:, None]), axis=1)

    # parameters for pairs that do not interact in the dst state
    w = interpolate_w_coord(src_w, cutoff, lamb)
    pair_params_excluded_dst = jnp.concatenate((src_qlj, w[:, None]), axis=1)

    # parameters for pairs that interact in both src and dst states
    w = jax.vmap(interpolate.linear_interpolation, (0, 0, None))(src_w, dst_w, lamb)
    qlj = interpolate.linear_interpolation(src_qlj, dst_qlj, lamb)
    pair_params_not_excluded = jnp.concatenate((qlj, w[:, None]), axis=1)

    pair_params = jnp.where(
        is_excluded_src,
        pair_params_excluded_src,
        jnp.where(
            is_excluded_dst,
            pair_params_excluded_dst,
            pair_params_not_excluded,
        ),
    )

    return pair_params


@dataclass(kw_only=True)
class AlignedPotential:
    src_params: Union[NDArray[np.float64], jax.Array]
    dst_params: Union[NDArray[np.float64], jax.Array]
    mins: Optional[NDArray[np.float64]]
    maxes: Optional[NDArray[np.float64]]

    def interpolate(self, lamb):
        raise NotImplementedError()


@dataclass(kw_only=True)
class AlignedBond(AlignedPotential):
    idxs: NDArray[np.int32]

    def interpolate(self, lamb):
        k_min = 0.1
        params = batch_interpolate_harmonic_bond_params(
            self.src_params, self.dst_params, lamb, k_min, self.mins, self.maxes
        )
        params = jnp.array(params).T

        return HarmonicBond(self.idxs).bind(params)


@dataclass(kw_only=True)
class AlignedAngle(AlignedPotential):
    idxs: NDArray[np.int32]

    def interpolate(self, lamb):
        k_min = 0.05
        params = batch_interpolate_harmonic_angle_params(
            self.src_params, self.dst_params, lamb, k_min, self.mins, self.maxes
        )
        params = jnp.array(params).T
        return HarmonicAngle(self.idxs).bind(params)


@dataclass(kw_only=True)
class AlignedTorsion(AlignedPotential):
    idxs: NDArray[np.int32]

    def interpolate(self, lamb):
        params = batch_interpolate_periodic_torsion_params(
            self.src_params, self.dst_params, lamb, self.mins, self.maxes
        )
        params = jnp.array(params).T
        return PeriodicTorsion(self.idxs).bind(params)


@dataclass(kw_only=True)
class AlignedChiralAtom(AlignedPotential):
    idxs: NDArray[np.int32]

    def interpolate(self, lamb):
        k_min = 0.025
        params = batch_interpolate_chiral_atom_params(
            self.src_params, self.dst_params, lamb, k_min, self.mins, self.maxes
        )
        params = jnp.array(params).reshape(-1)
        return ChiralAtomRestraint(self.idxs).bind(params)


@dataclass(kw_only=True)
class AlignedNonbondedPairlist(AlignedPotential):
    idxs: NDArray[np.int32]
    cutoff: float
    beta: float

    def interpolate(self, lamb):
        # (ytz): batch_interpolate_nonbonded_pair_list_params currently fails to respect the self.mins and self.maxes
        # boundaries.
        params = batch_interpolate_nonbonded_pair_list_params(self.cutoff, self.src_params, self.dst_params, lamb)
        params = jnp.array(params)
        return NonbondedPairListPrecomputed(self.idxs, self.beta, self.cutoff).bind(params)


@dataclass(kw_only=True)
class AlignedNonbondedInteractionGroup(AlignedPotential):
    row_atom_idxs: NDArray[np.int32]
    col_atom_idxs: NDArray[np.int32]
    cutoff: float
    beta: float

    def interpolate(self, lamb):
        # (ytz): self.mins and self.max is currently ignored.
        qse_idxs = [NBParamIdx.Q_IDX, NBParamIdx.LJ_SIG_IDX, NBParamIdx.LJ_EPS_IDX]
        src_qse = self.src_params[:, qse_idxs]
        dst_qse = self.dst_params[:, qse_idxs]
        qse = interpolate.linear_interpolation(src_qse, dst_qse, lamb)
        src_w = self.src_params[:, NBParamIdx.W_IDX]
        dst_w = self.dst_params[:, NBParamIdx.W_IDX]
        w = interpolate_w_coord(src_w, dst_w, lamb).reshape(-1, 1)
        params = jnp.hstack([qse, w])
        n_atoms = len(params)
        return NonbondedInteractionGroup(n_atoms, self.row_atom_idxs, self.beta, self.cutoff, self.col_atom_idxs).bind(
            params
        )


@dataclass(kw_only=True)
class AlignedNonbondedAllPairs(AlignedPotential):
    num_atoms: int
    exclusion_idxs: NDArray[np.int32]
    atom_idxs: NDArray[np.int32]
    scale_factors: NDArray[np.float64]
    cutoff: float
    beta: float

    def interpolate(self, lamb):
        # (ytz): self.mins and self.max is currently ignored.
        qse_idxs = [NBParamIdx.Q_IDX, NBParamIdx.LJ_SIG_IDX, NBParamIdx.LJ_EPS_IDX]
        src_qse = self.src_params[:, qse_idxs]
        dst_qse = self.dst_params[:, qse_idxs]
        qse = interpolate.linear_interpolation(src_qse, dst_qse, lamb)
        src_w = self.src_params[:, NBParamIdx.W_IDX]
        dst_w = self.dst_params[:, NBParamIdx.W_IDX]
        w = interpolate_w_coord(src_w, dst_w, lamb).reshape(-1, 1)
        params = jnp.hstack([qse, w])
        return Nonbonded(
            self.num_atoms, self.exclusion_idxs, self.scale_factors, self.beta, self.cutoff, self.atom_idxs
        ).bind(params)
