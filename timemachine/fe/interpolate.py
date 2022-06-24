import numpy as np


def _add_dst_to_src_bond_or_angle(src_idxs, src_params, dst_idxs, dst_params):
    """
    eg:
    src_idxs: [[4,9], [3,4]]
    src_params: [[a, b], [c, d]]

    dst_idxs: [[3,4], [9,5]]
    dst_params: [[e, f], [g, h]]

    zero_flags: [True, False]

    Generate end-state key value pairs

    src_idxs: [[4,9], [3,4], [9,5]]
    src_params: [[a,b], [c,d], [0,h]

    """
    # used for:
    # bonds, angles, and nonbonded terms. do not use this for torsions since the periods
    # are unstable. note that *missing* terms are the result of core-hopping, or chiral
    # inversions on core atoms. dummy atom interactions (bonds/angles) are fully maintained
    # at the end-state.
    assert len(set((src_idxs))) == len(src_idxs)
    assert len(set((dst_idxs))) == len(dst_idxs)

    src_kv = dict(zip(src_idxs, src_params))
    for idxs, params in zip(dst_idxs, dst_params):
        assert idxs[0] < idxs[-1]
        if idxs not in src_kv:
            # new_params = []
            _, bond_or_angle = params
            src_kv[idxs] = (0, bond_or_angle)

    return src_kv


def align_harmonic_bond_or_angle_idxs_and_params(src_idxs, src_params, dst_idxs, dst_params):

    # sanitize
    src_idxs = [tuple(p) for p in src_idxs]
    src_params = [tuple(p) for p in src_params]
    dst_idxs = [tuple(p) for p in dst_idxs]
    dst_params = [tuple(p) for p in dst_params]

    src_kv = _add_dst_to_src_bond_or_angle(src_idxs, src_params, dst_idxs, dst_params)
    dst_kv = _add_dst_to_src_bond_or_angle(dst_idxs, dst_params, src_idxs, src_params)
    assert src_kv.keys() == dst_kv.keys()

    res = set()

    for k in src_kv.keys():
        res.add((tuple(k), src_kv[k], dst_kv[k]))

    return res


align_harmonic_bond_idxs_and_params = align_harmonic_bond_or_angle_idxs_and_params
align_harmonic_angle_idxs_and_params = align_harmonic_bond_or_angle_idxs_and_params

# special case for torsions
def _add_dst_to_src_torsion(src_idxs, src_params, dst_idxs, dst_params):

    src_kv = dict()

    for idxs, params in zip(src_idxs, src_params):
        assert idxs[0] < idxs[-1]
        k, phase, period = params
        key = (idxs, period)
        src_kv[key] = (k, phase)

    for idxs, params in zip(dst_idxs, dst_params):
        assert idxs[0] < idxs[-1]
        k, phase, period = params
        key = (idxs, period)
        if key not in src_kv:
            src_kv[key] = (0, phase)

    return src_kv


def align_torsion_idxs_and_params(src_idxs, src_params, dst_idxs, dst_params):

    # sanitize
    src_idxs = [tuple(p) for p in src_idxs]
    src_params = [tuple(p) for p in src_params]
    dst_idxs = [tuple(p) for p in dst_idxs]
    dst_params = [tuple(p) for p in dst_params]

    src_kv = _add_dst_to_src_torsion(src_idxs, src_params, dst_idxs, dst_params)
    dst_kv = _add_dst_to_src_torsion(dst_idxs, dst_params, src_idxs, src_params)

    assert src_kv.keys() == dst_kv.keys()

    res = set()

    for key in src_kv.keys():
        idxs, period = key
        assert idxs[0] < idxs[-1]
        src_k, src_phase = src_kv[key]
        dst_k, dst_phase = dst_kv[key]
        res.add((idxs, (src_k, src_phase, period), (dst_k, dst_phase, period)))

    return res


def _add_dst_to_src_nonbonded(src_idxs, src_params, dst_idxs):
    assert len(set((src_idxs))) == len(src_idxs)
    assert len(set((dst_idxs))) == len(dst_idxs)

    src_kv = dict(zip(src_idxs, src_params))
    for idxs in dst_idxs:
        assert idxs[0] < idxs[-1]
        if idxs not in src_kv:
            src_kv[idxs] = (0, 0, 0)

    return src_kv


def align_nonbonded_idxs_and_params(src_idxs, src_params, dst_idxs, dst_params):

    # sanitize
    src_idxs = [tuple(p) for p in src_idxs]
    src_params = [tuple(p) for p in src_params]
    dst_idxs = [tuple(p) for p in dst_idxs]
    dst_params = [tuple(p) for p in dst_params]

    src_kv = _add_dst_to_src_nonbonded(src_idxs, src_params, dst_idxs)
    dst_kv = _add_dst_to_src_nonbonded(dst_idxs, dst_params, src_idxs)
    assert src_kv.keys() == dst_kv.keys()

    res = set()

    for k in src_kv.keys():
        res.add((tuple(k), src_kv[k], dst_kv[k]))

    return res


# special case for torsions
def _add_dst_to_src_chiral_atom(src_idxs, src_params, dst_idxs):
    src_kv = dict()

    for idxs, k in zip(src_idxs, src_params):
        src_kv[idxs] = k

    for idxs in dst_idxs:
        if idxs not in src_kv:
            src_kv[idxs] = 0

    return src_kv


def align_chiral_atom_idxs_and_params(src_idxs, src_params, dst_idxs, dst_params):

    # sanitize
    src_idxs = [tuple(p) for p in src_idxs]
    dst_idxs = [tuple(p) for p in dst_idxs]

    src_kv = _add_dst_to_src_chiral_atom(src_idxs, src_params, dst_idxs)
    dst_kv = _add_dst_to_src_chiral_atom(dst_idxs, dst_params, src_idxs)
    assert src_kv.keys() == dst_kv.keys()

    res = set()

    for k in src_kv.keys():
        res.add((tuple(k), src_kv[k], dst_kv[k]))

    return res


def _add_dst_to_src_chiral_bond(src_idxs, src_params, src_signs, dst_idxs, dst_params, dst_signs):
    src_kv = dict()

    for idxs, k, sign in zip(src_idxs, src_params, src_signs):
        assert idxs[0] < idxs[-1]
        key = (idxs, sign)
        src_kv[key] = k

    for idxs, k, sign in zip(dst_idxs, dst_params, dst_signs):
        assert idxs[0] < idxs[-1]
        key = (idxs, sign)
        if key not in src_kv:
            src_kv[key] = 0

    return src_kv


def align_chiral_bond_idxs_and_params(src_idxs, src_params, src_signs, dst_idxs, dst_params, dst_signs):

    # sanitize
    src_idxs = [tuple(p) for p in src_idxs]
    dst_idxs = [tuple(p) for p in dst_idxs]

    src_kv = _add_dst_to_src_chiral_bond(src_idxs, src_params, src_signs, dst_idxs, dst_params, dst_signs)
    dst_kv = _add_dst_to_src_chiral_bond(dst_idxs, dst_params, dst_signs, src_idxs, src_params, src_signs)

    assert src_kv.keys() == dst_kv.keys()

    res = set()

    for key in src_kv.keys():
        idxs, sign = key
        src_k = src_kv[key]
        dst_k = dst_kv[key]
        res.add((idxs, sign, src_k, dst_k))

    return res


def linear_interpolation(src_params, dst_params, lamb):
    """
    Linearly interpolate between src and dst params
    """
    return (1 - lamb) * np.array(src_params) + lamb * np.array(dst_params)
