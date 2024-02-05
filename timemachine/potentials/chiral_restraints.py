import jax
import jax.numpy as jnp


def normalize(x):
    return x / jnp.linalg.norm(x)


@jax.jit
def pyramidal_volume(xc, x1, x2, x3):
    """
    Compute the normalized pyramidal volume given four points. This is implemented
    as a triple product with x0 at the center.

    Parameters
    ----------
    xc: np.ndarray (3,)
        Center point

    x1, x2, x3: np.ndarrays of shape (3,)
        Three points around center point

    Returns
    -------
    float
        A number between -1.0 < x < 1.0 denoting the normalized chirality

    """
    # compute vectors
    v0 = normalize(x1 - xc)
    v1 = normalize(x2 - xc)
    v2 = normalize(x3 - xc)

    # triple product
    return jnp.dot(jnp.cross(v0, v1), v2)


@jax.jit
def torsion_volume(ci, cj, ck, cl):
    """
    Compute normalized torsional volume given four points. This is implemented
    as the dot product of cross products spanned by atoms (i,j,k), and atoms (j,k,l).

    Parameters
    ----------
    ci, cj, ck, cl: np.ndarrays of shape (3,)
        four points

    Returns
    -------
    float
        A number between -1.0 < x < 1.0 denoting the normalized chirality
    """
    rij = normalize(cj - ci)
    rkj = normalize(cj - ck)
    rkl = normalize(cl - ck)

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
    return jnp.where(v > 0, kc * v**2, 0.0)


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
    v = torsion_volume(x0, x1, x2, x3)
    return jnp.where(v * s > 0, kc * v**2, 0.0)


# allow batching over multiple idxs and force constants
U_chiral_atom_batch = jax.vmap(U_chiral_atom, (None, 0, None), 0)
U_chiral_bond_batch = jax.vmap(U_chiral_bond, (None, 0, None, 0), 0)

# allow batching over multiple idxs and force constants
U_chiral_atom_batch_all = jax.vmap(U_chiral_atom, (None, 0, 0), 0)
U_chiral_bond_batch_all = jax.vmap(U_chiral_bond, (None, 0, 0, 0), 0)


def chiral_atom_restraint(conf, params, box, idxs):
    """
    Flat-bottom chiral atom restraint

    Notes
    -----
    * box unused
    """
    assert len(idxs) == len(params), f"len(idxs) = {len(idxs)}, len(params) = {len(params)}"
    return jnp.sum(U_chiral_atom_batch_all(conf, idxs, params)) if len(idxs) else 0.0


def chiral_bond_restraint(conf, params, box, idxs, signs):
    """
    Flat-bottom chiral bond restraint

    Notes
    -----
    * box unused
    """
    assert len(idxs) == len(params)
    assert len(idxs) == len(signs)
    return jnp.sum(U_chiral_bond_batch_all(conf, idxs, params, signs)) if len(idxs) else 0.0
