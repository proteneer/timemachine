import jax
import jax.numpy as jnp


def pyramidal_volume(xc, x1, x2, x3):
    """
    Compute the normalized pyramidal volume given four points. This is implemented
    as a triple product with x0 at the center.

    Parameters
    ----------
    xc: np.array (3,)
        Center point

    x1: np.array: (3,)
        First point

    x2: np.array: (3,)
        Second point

    x3: np.array: (3,)
        Third point

    Returns
    -------
    float
        A number between -1.0 < x < 1.0 denoting the normalized chirality

    """
    # compute vectors
    v0 = x1 - xc
    v1 = x2 - xc
    v2 = x3 - xc

    v0 = v0 / jnp.linalg.norm(v0)
    v1 = v1 / jnp.linalg.norm(v1)
    v2 = v2 / jnp.linalg.norm(v2)

    # triple product
    return jnp.dot(jnp.cross(v0, v1), v2)


def torsion_volume(ci, cj, ck, cl):
    """
    Compute normalized torsional volume given four points. This is implemented
    as the dot product of cross products spanned by atoms (i,j,k), and atoms (j,k,l).

    Parameters
    ----------
    np.array: (3,)
        First point

    np.array: (3,)
        Second point

    np.array: (3,)
        Third point

    np.array: (3,)
        Fourth point

    Returns
    -------
    float
        A number between -1.0 < x < 1.0 denoting the normalized chirality

    """
    rij = cj - ci
    rkj = cj - ck
    rkl = cl - ck

    rij = rij / jnp.linalg.norm(rij)
    rkj = rkj / jnp.linalg.norm(rkj)
    rkl = rkl / jnp.linalg.norm(rkl)

    n1 = jnp.cross(rij, rkj)
    n2 = jnp.cross(rkj, rkl)

    return jnp.dot(n1, n2)


def U_chiral_atom(x, idxs, kc):
    """
    Flat bottom chiral restraint, penalizing positive volumes. ie.
    If chiral volume > 0, the U = kc*volume^2, else 0.
    """
    # guard against numpy/raw lists
    x = jnp.array(x)
    assert len(idxs) == 4
    x0, x1, x2, x3 = x[idxs]
    v = pyramidal_volume(x0, x1, x2, x3)
    return jnp.where(v > 0, kc * v ** 2, 0.0)


def U_chiral_bond(x, idxs, kc, s):
    """
    For torsions, the ordering of the atoms i,j,k,l is fixed (and symmetrized)
    so we can't simply swap two idxs to get the sign.

    Flat bottom chiral restraint, penalizing positive volumes. ie.
    If chiral volume > 0, the U = kc*volume^2, else 0.
    """
    x = jnp.array(x)
    assert len(idxs) == 4
    i, j, k, l = idxs
    # assert s == 1 or s == -1, can't be used during vmap/tracing
    x0, x1, x2, x3 = x[i], x[j], x[k], x[l]
    v = s * torsion_volume(x0, x1, x2, x3)
    return jnp.where(v > 0, kc * v ** 2, 0.0)


# allow batching over multiple_idxs
U_chiral_atom_batch = jax.vmap(U_chiral_atom, (None, 0, None), 0)
U_chiral_bond_batch = jax.vmap(U_chiral_bond, (None, 0, None, 0), 0)


def chiral_atom_restraint(conf, params, box, lamb, idxs, kc):
    """
    Flat-bottom chiral atom restraint
    """
    return jnp.sum(U_chiral_atom_batch(conf, idxs, kc))


def chiral_bond_restraint(conf, params, box, lamb, idxs, kc, signs):
    """
    Flat-bottom chiral bond restraint
    """
    return jnp.sum(U_chiral_bond_batch(conf, idxs, kc, signs))
