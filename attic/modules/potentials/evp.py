import jax
import jax.numpy as jnp
import numpy as np
from jax.config import config

config.update("jax_enable_x64", True)

from timemachine.potentials.pmi import dsyevv3


def recenter(conf):
    return conf - jnp.mean(conf, axis=0)


def inertia_tensor(conf, masses):
    com = jnp.average(conf, axis=0, weights=masses)
    conf = conf - com

    xs = conf[:, 0]
    ys = conf[:, 1]
    zs = conf[:, 2]
    xx = jnp.average(ys * ys + zs * zs, weights=masses)
    yy = jnp.average(xs * xs + zs * zs, weights=masses)
    zz = jnp.average(xs * xs + ys * ys, weights=masses)
    xy = jnp.average(-xs * ys, weights=masses)
    xz = jnp.average(-xs * zs, weights=masses)
    yz = jnp.average(-ys * zs, weights=masses)
    tensor = jnp.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])

    return com, tensor


def pmi_restraints(conf, params, box, lamb, a_idxs, b_idxs, masses, angle_force, com_force):

    a_com, a_tensor = inertia_tensor(conf[a_idxs], masses[a_idxs])
    b_com, b_tensor = inertia_tensor(conf[b_idxs], masses[b_idxs])

    a_eval, a_evec = jnp.linalg.eigh(a_tensor)  # already sorted
    b_eval, b_evec = jnp.linalg.eigh(b_tensor)  # already sorted

    r = jnp.matmul(jnp.transpose(a_evec), b_evec)
    I = jnp.eye(3)

    loss = []
    for v, e in zip(r, I):
        a_pos = jnp.arccos(jnp.sum(v * e))  # norm is always 1
        a_neg = jnp.arccos(jnp.sum(-v * e))  # norm is always 1
        a = jnp.amin([a_pos, a_neg])
        loss.append(a * a)


def pmi_u(r):
    I = jnp.eye(3)

    loss = []
    for v, e in zip(r, I):
        a_pos = jnp.arccos(jnp.sum(v * e))  # norm is always 1
        a_neg = jnp.arccos(jnp.sum(-v * e))  # norm is always 1
        a = jnp.amin([a_pos, a_neg])
        loss.append(a * a)

    return jnp.sum(loss)


# def simplified_u(r):
#     I = np.eye(3)
#     pos = np.sum(r*I, axis=-1)
#     neg = np.sum(-r*I, axis=-1)
#     acos_pos = np.arccos(pos)
#     acos_neg = np.arccos(neg)
#     # [a,b,c]
#     # [d,e,f]
#     # -------
#     # [min(a,d), min(b,e), min(c,f)]
#     a = np.amin([acos_pos, acos_neg], axis=0)
#     return np.sum(a*a)


# ported over from autodiff
def grad_eigh(w, v, wg, vg):
    """Gradient for eigenvalues and vectors of a symmetric matrix.

    Parameters
    ----------
    w: eigenvalues

    v: eigenvectors

    wg: adjoint eigenvalues

    vg: adjoint eigenvectors
    """
    vc = v  # real
    N = 3
    # wg, vg = g          # Gradient w.r.t. eigenvalues, eigenvectors.
    w_repeated = jnp.repeat(w[..., jnp.newaxis], N, axis=-1)
    # Eigenvalue part
    vjp_temp = jnp.dot(vc * wg[..., jnp.newaxis, :], v.T)

    # Add eigenvector part only if non-zero backward signal is present.
    # This can avoid NaN results for degenerate cases if the function depends
    # on the eigenvalues only.
    if jnp.any(vg):
        off_diag = jnp.ones((N, N)) - jnp.eye(N)
        F = off_diag / (w_repeated.T - w_repeated + jnp.eye(N))
        vjp_temp += jnp.dot(jnp.dot(vc, F * jnp.dot(v.T, vg)), v.T)
    else:
        assert 0

    off_diag_mask = (np.ones((3, 3)) - np.eye(3)) / 2

    return vjp_temp * jnp.eye(vjp_temp.shape[-1]) + (vjp_temp + vjp_temp.T) * off_diag_mask
    # return vjp_temp*np.eye(vjp_temp.shape[-1]) + (vjp_temp + vjp_temp.T) * tri


def simplified_u(a_tensor, b_tensor):
    a_eval, a_evec = jnp.linalg.eigh(a_tensor)
    b_eval, b_evec = jnp.linalg.eigh(b_tensor)
    r = jnp.matmul(jnp.transpose(a_evec), b_evec)
    I = jnp.eye(3)
    rI = r * I  # 3x3 -> 3x3
    pos = jnp.sum(rI, axis=-1)
    neg = jnp.sum(-rI, axis=-1)
    acos_pos = jnp.arccos(pos)
    acos_neg = jnp.arccos(neg)
    # [a,b,c]
    # [d,e,f]
    # -------
    # [min(a,d), min(b,e), min(c,f)]
    a = jnp.amin([acos_pos, acos_neg], axis=0)
    return jnp.sum(a * a)


def test_force(a_tensor, b_tensor):
    # a_eval, a_evec = np.linalg.eigh(a_tensor)
    # b_eval, b_evec = np.linalg.eigh(b_tensor)

    a_eval, a_evec = dsyevv3(a_tensor)
    b_eval, b_evec = dsyevv3(b_tensor)

    # print("ref w", a_eval)
    # print("test w", dsyevc3(a_tensor))

    # print("ref v", a_evec)
    # print("test v", dsyevv3(a_tensor))

    # assert 0

    r = jnp.matmul(jnp.transpose(a_evec), b_evec)
    I = jnp.eye(3)
    rI = r * I  # 3x3 -> 3x3
    pos = jnp.sum(rI, axis=-1)  # 3x3 -> 3
    neg = -jnp.sum(rI, axis=-1)  # 3x3 -> 3
    acos_pos = jnp.arccos(pos)  # 3 -> 3
    acos_neg = jnp.arccos(neg)  # 3 -> 3
    a = jnp.amin([acos_pos, acos_neg], axis=0)  # 2x3 -> 3

    # derivatives, start backprop
    dl_da2 = jnp.ones(3)  # 1 x 3
    da2_da = 2 * a * jnp.eye(3)  # 3 x 3
    da_darg = jnp.stack([jnp.eye(3) * (acos_pos < acos_neg), jnp.eye(3) * (acos_neg < acos_pos)])

    dl_darg = jnp.matmul(jnp.matmul(dl_da2, da2_da), da_darg)
    dpos = dl_darg[0] * (-1 / jnp.sqrt(1 - pos * pos))
    dneg = dl_darg[1] * (-1 / jnp.sqrt(1 - neg * neg))
    dneg = -dneg

    dpn_dr = jnp.array(
        [
            [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [1, 1, 1], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [1, 1, 1]],
        ]
    )

    # element wise
    dr = (jnp.matmul(dpos, dpn_dr) + jnp.matmul(dneg, dpn_dr)) * jnp.eye(3)

    dr_daevec = jnp.matmul(b_evec, dr.T)
    dr_dbevec = jnp.matmul(a_evec, dr.T)

    dl_datensor = grad_eigh(a_eval, a_evec, jnp.zeros_like(a_eval), dr_daevec)
    dl_dbtensor = grad_eigh(b_eval, b_evec, jnp.zeros_like(b_eval), dr_dbevec)

    return dl_datensor, dl_dbtensor


def test1():

    grad_fn = jax.jacobian(simplified_u, argnums=(0, 1))

    for _ in range(10):
        N = 50
        x_a = np.random.rand(N, 3)
        x_b = np.random.rand(N, 3)

        a_com, a_tensor = inertia_tensor(x_a, np.ones(N, dtype=jnp.float64))
        b_com, b_tensor = inertia_tensor(x_b, np.ones(N, dtype=jnp.float64))

        rf = np.asarray(grad_fn(a_tensor, b_tensor))
        tf = np.asarray(test_force(a_tensor, b_tensor))

        np.testing.assert_almost_equal(rf, tf, decimal=5)


def test0():

    np.random.seed(2020)

    for trip in range(10):
        print("trip", trip)
        N = 50
        x_a = np.random.rand(N, 3)

        a_com, a_tensor = inertia_tensor(x_a, np.ones(N, dtype=jnp.float64))

        onp_res = np.linalg.eigh(a_tensor)
        w = onp_res[0]
        Q = onp_res[1]
        for d in range(3):
            np.testing.assert_almost_equal(np.matmul(a_tensor, Q[:, d]), w[d] * Q[:, d])

        jnp_res = jnp.linalg.eigh(a_tensor)
        evp_res = dsyevv3(a_tensor)

        np.set_printoptions(formatter={"float": lambda x: "{0:0.16f}".format(x)})

        np.testing.assert_almost_equal(onp_res[0], jnp_res[0])
        np.testing.assert_almost_equal(onp_res[1], jnp_res[1])

        np.testing.assert_almost_equal(onp_res[0], evp_res[0])
        np.testing.assert_almost_equal(np.abs(onp_res[1]), np.abs(evp_res[1]))


# test0()
test1()
