from functools import partial
from typing import Hashable

import numpy as np


class DuplicateAlignmentKeysError(RuntimeError):
    pass


def align_idxs_and_params(
    src_idxs,
    src_params,
    dst_idxs,
    dst_params,
    make_default,
    key=lambda idxs, _: idxs,
    get_idxs=lambda key: key,
    validate_idxs=lambda _: None,
):
    """
    Aligns source and destination parameters. When a parameter is
    present in one set but absent in the other, the missing value is
    filled with a default computed from the value that is present. By
    default, idxs are used as the alignment key.

    Parameters
    ----------
    src_idxs, dst_idxs: array of int
      Atom indices for each potential term. E.g. for harmonic bonds, each would have shape (num_bonds, 2)
    src_params, dst_params: array of float
      Parameters corresponding to the specified indices
    make_default: callable
      Should return the value to fill for missing src (dst) given the value present for dst (src)
    key: callable
      Should return the alignment key given arguments (idxs, params)
    get_idxs: callable
      Should return the idxs given an alignment key
    validate_idxs: callable
      Called on each set of idxs in src_idxs and dst_idxs; used to validate input

    Returns
    -------
    set
      set of tuples (idxs, src_params, dst_params)
    """

    for all_idxs in [src_idxs, dst_idxs]:
        for idxs in all_idxs:
            validate_idxs(idxs)

    # used to convert arrays to a hashable type for use as dict keys and in sets
    def to_hashable(x):
        return x if isinstance(x, Hashable) else tuple(x)

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


align_harmonic_bond_or_angle_idxs_and_params = partial(align_idxs_and_params, make_default=lambda p: (0, p[1]))
align_harmonic_bond_idxs_and_params = partial(
    align_harmonic_bond_or_angle_idxs_and_params, validate_idxs=assert_canonical_bond
)
align_harmonic_angle_idxs_and_params = align_harmonic_bond_or_angle_idxs_and_params
align_nonbonded_idxs_and_params = partial(align_idxs_and_params, make_default=lambda _: (0, 0, 0))
align_chiral_atom_idxs_and_params = partial(align_idxs_and_params, make_default=lambda _: 0)
align_torsion_idxs_and_params = partial(
    align_idxs_and_params,
    make_default=lambda p: (0, p[1], p[2]),
    key=lambda idxs, params: (idxs, params[2]),
    get_idxs=lambda key: key[0],
)


def align_chiral_bond_idxs_and_params(src_idxs, src_params, src_signs, dst_idxs, dst_params, dst_signs):
    return {
        (idxs, s, p1, p2)
        for idxs, (s, p1), (_, p2) in align_idxs_and_params(
            src_idxs,
            zip(src_signs, src_params),
            dst_idxs,
            zip(dst_signs, dst_params),
            make_default=lambda ps: (ps[0], 0),
            key=lambda idxs, ps: (idxs, ps[0]),
            get_idxs=lambda key: key[0],
        )
    }


def linear_interpolation(src_params, dst_params, lamb):
    """
    Linearly interpolate between src and dst params
    """
    return (1 - lamb) * np.array(src_params) + lamb * np.array(dst_params)
