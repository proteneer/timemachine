from collections.abc import Iterable
from functools import partial
from typing import Any, Callable

# import jax.numpy as jnp
import numpy as jnp


class DuplicateAlignmentKeysError(RuntimeError):
    pass


Idxs = Any
Params = Any
Key = Any


def to_hashable(x):
    return tuple(to_hashable(e) for e in x) if isinstance(x, Iterable) else x


def align_idxs_and_params(
    src_idxs,
    src_params,
    dst_idxs,
    dst_params,
    make_default: Callable[[Params], Params],
    key: Callable[[Idxs, Params], Key] = lambda idxs, _: idxs,
    get_idxs: Callable[[Key], Idxs] = lambda key: key,
    validate_idxs: Callable[[Idxs], None] = lambda _: None,
) -> set[tuple[Idxs, Params, Params]]:
    """
    Given two input parameter sets (idxs, params), aligns by the
    specified key to produce two output parameter sets, where the
    outputs have the same shape. When an (idxs, params) pair is
    present in one input set but absent in the other, the missing
    value is filled with a default computed from the value that is
    present. By default, idxs are used as the alignment key.

    Parameters
    ----------
    src_idxs, dst_idxs: array of int
        Atom indices for each potential term. E.g. for harmonic bonds,
        each would have shape (num_bonds, 2)
    src_params, dst_params: array of float
        Parameters corresponding to the specified indices
    make_default: callable
        Should return the value to fill for missing src (dst) given
        the value present for dst (src)
    key: callable
        Should return the alignment key given arguments (idxs, params)
    get_idxs: callable
        Should return the idxs given an alignment key
    validate_idxs: callable
        Called on each set of idxs in src_idxs and dst_idxs; used to
        validate input

    Returns
    -------
    set
        set of tuples (idxs, src_params, dst_params)

    Examples
    --------
        >>> align_harmonic_angle_idxs_and_params = partial(align_idxs_and_params, make_default=lambda p: (0.0, p[1]))

        For harmonic bonds, we align on idxs and fill missing
        parameters using zero for the force constant and the
        equilibrium bond length from the opposite end state.

        >>> src_idxs = [[4, 9], [3, 4]]

        >>> src_params = [[1.0 , 2.0], [3.0, 4.0]]

        >>> dst_idxs = [[3, 4], [9, 5]]

        >>> dst_params = [[5.0, 6.0], [7.0, 8.0]]

        >>> align_harmonic_angle_idxs_and_params(src_idxs, src_params, dst_idxs, dst_params)
        {((9, 5), (0.0, 8.0), (7.0, 8.0)), ((4, 9), (1.0, 2.0), (0.0, 2.0)), ((3, 4), (3.0, 4.0), (5.0, 6.0))}
    """

    for all_idxs in [src_idxs, dst_idxs]:
        for idxs in all_idxs:
            validate_idxs(idxs)

    # used to convert arrays to a hashable type for use as dict keys and in sets
    def make_kv(all_idxs, all_params):
        kvs = [(to_hashable(key(idxs, params)), params) for idxs, params in zip(all_idxs, all_params)]

        def has_duplicates(x):
            x = list(x)
            return len(set(x)) < len(x)

        if has_duplicates(k for k, _ in kvs):
            raise DuplicateAlignmentKeysError()

        return dict(kvs)

    src_kv = make_kv(src_idxs, src_params)
    dst_kv = make_kv(dst_idxs, dst_params)

    return {
        (
            get_idxs(k),
            to_hashable(src_kv[k]) if k in src_kv else make_default(dst_kv[k]),
            to_hashable(dst_kv[k]) if k in dst_kv else make_default(src_kv[k]),
        )
        for k in set(src_kv.keys()).union(dst_kv.keys())
    }


def assert_canonical_bond(bond):
    assert bond[0] < bond[-1]


align_harmonic_bond_idxs_and_params = partial(
    align_idxs_and_params,
    make_default=lambda p: (0, p[1]),  # 0 is force constant, p[1] is bond length
    validate_idxs=assert_canonical_bond,
)
align_harmonic_angle_idxs_and_params = partial(align_idxs_and_params, make_default=lambda p: (0, p[1], 0))
align_nonbonded_idxs_and_params = partial(align_idxs_and_params, make_default=lambda _: (0, 0, 0, 0))
align_chiral_atom_idxs_and_params = partial(align_idxs_and_params, make_default=lambda _: 0)
align_proper_idxs_and_params = partial(
    align_idxs_and_params,
    make_default=lambda p: (0, p[1], p[2]),  # p[1], p[2] is phase, period
    key=lambda idxs, p: (idxs, p[1], p[2]),  # align on idxs, phase, period
    get_idxs=lambda key: key[0],
)
# impropers do not require key'ing on phase, period
align_improper_idxs_and_params = partial(align_idxs_and_params, make_default=lambda p: (0, p[1], p[2]))


def align_chiral_bond_idxs_and_params(src_idxs, src_params, src_signs, dst_idxs, dst_params, dst_signs):
    return {
        (idxs, sign, p1, p2)
        for idxs, (sign, p1), (_, p2) in align_idxs_and_params(
            src_idxs,
            zip(src_signs, src_params),
            dst_idxs,
            zip(dst_signs, dst_params),
            make_default=lambda p: (p[0], 0),  # p[0] is sign, 0 is force constant
            key=lambda idxs, p: (idxs, p[0]),  # align on idxs, sign
            get_idxs=lambda key: key[0],
        )
    }


def linear_interpolation(src_params, dst_params, lamb):
    """
    Linearly interpolate between src and dst params
    """
    return (1 - lamb) * src_params + lamb * dst_params


def log_linear_interpolation(src_params, dst_params, lamb, min_value):
    """
    Linear interpolation in log space.

    Notes
    ----
    * f(0) = src_params, f(1) = dst_params only if src_params and dst_params are > min_value, respectively.
    """

    # clip to handle out-of-range end state values
    src_params = jnp.maximum(src_params, min_value)
    dst_params = jnp.maximum(dst_params, min_value)

    # tbd: handle 0s better if lamb = 0 and src_params == 0
    return jnp.exp(linear_interpolation(jnp.log(src_params), jnp.log(dst_params), lamb))


def pad(f, src_params, dst_params, lamb, lambda_min, lambda_max):
    """
    Use the specified interpolation function in the interval (lambda_min, lambda_max), otherwise pin to the end-state
    values.
    """
    return jnp.where(
        lamb <= lambda_min,
        src_params,
        jnp.where(
            lambda_max <= lamb,
            dst_params,
            f(src_params, dst_params, (lamb - lambda_min) / (lambda_max - lambda_min)),
        ),
    )
