import jax.numpy as jnp

def restraint(conf, lamb, params, lamb_flags, box, bond_idxs):
    r"""
    Compute the harmonic bond energy given a collection of molecules.

    This implements a harmonic bond potential:
        V(conf; lamb) = \sum_bond kbs[bond] term[bond]^2

        where term[bond] = 1 - exp(-a0s[bond]*(dij[bond] - b0s[bond]))
        and where
            dij[bond] = distance[bond] + f_lambda^2
            and where
                f_lambda = lamb * lamb_flags

    Parameters:
    -----------
    conf: shape [num_atoms, 3] np.array
        atomic coordinates

    lamb: float
        lambda value for 4d decoupling

    params: shape [num_params,] np.array
        unique parameters

    lamb_flags: np.array
        equivalent to offset_idxs, adjusts how much we offset by

    box: shape [3, 3] np.array
        periodic boundary vectors, if not None

    bond_idxs: [num_bonds, 2] np.array
        each element (src, dst) is a unique bond in the conformation

    Notes
    -----
    * box argument is unused
    """
    f_lambda = lamb * lamb_flags

    ci = conf[bond_idxs[:, 0]]
    cj = conf[bond_idxs[:, 1]]

    dij = jnp.sqrt(jnp.sum(jnp.power(ci - cj, 2), axis=-1) + f_lambda * f_lambda)
    kbs = params[:, 0]
    b0s = params[:, 1]
    a0s = params[:, 2]

    term = 1 - jnp.exp(-a0s * (dij - b0s))

    energy = jnp.sum(kbs * term * term)

    return energy
