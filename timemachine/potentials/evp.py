import numpy as onp
import jax
import jax.numpy as np

from jax.config import config; config.update("jax_enable_x64", True)

from scipy.stats import special_ortho_group


from timemachine.potentials import evp

def recenter(conf):
    return conf - np.mean(conf, axis=0)

def inertia_tensor(conf, masses):
    com = np.average(conf, axis=0, weights=masses)
    conf = conf - com
    conf_T = conf.transpose()
    
    xs = conf[:, 0]
    ys = conf[:, 1]
    zs = conf[:, 2]
    xx = np.average(ys*ys + zs*zs, weights=masses)
    yy = np.average(xs*xs + zs*zs, weights=masses)
    zz = np.average(xs*xs + ys*ys, weights=masses)
    xy = np.average(-xs*ys, weights=masses)
    xz = np.average(-xs*zs, weights=masses)
    yz = np.average(-ys*zs, weights=masses)
    tensor = np.array([
        [xx, xy, xz],
        [xy, yy, yz],
        [xz, yz, zz]
    ])

    return com, tensor

def pmi_restraints(conf, params, box, lamb, a_idxs, b_idxs, masses, angle_force, com_force):

    a_com, a_tensor = inertia_tensor(conf[a_idxs], masses[a_idxs])
    b_com, b_tensor = inertia_tensor(conf[b_idxs], masses[b_idxs])

    a_eval, a_evec = np.linalg.eigh(a_tensor) # already sorted
    b_eval, b_evec = np.linalg.eigh(b_tensor) # already sorted

    r = np.matmul(np.transpose(a_evec), b_evec)
    I = np.eye(3)

    loss = []
    for v, e in zip(r, I):
        a_pos = np.arccos(np.sum(v*e)) # norm is always 1
        a_neg = np.arccos(np.sum(-v*e)) # norm is always 1
        a = np.amin([a_pos, a_neg])
        loss.append(a*a)

def pmi_u(r):
    I = np.eye(3)

    loss = []
    for v, e in zip(r, I):
        a_pos = np.arccos(np.sum(v*e)) # norm is always 1
        a_neg = np.arccos(np.sum(-v*e)) # norm is always 1
        a = np.amin([a_pos, a_neg])
        loss.append(a*a)

    return np.sum(loss)

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
    vc = v # real 
    N = 3
    # wg, vg = g          # Gradient w.r.t. eigenvalues, eigenvectors.
    w_repeated = np.repeat(w[..., np.newaxis], N, axis=-1)
    # Eigenvalue part
    vjp_temp = np.dot(vc * wg[..., np.newaxis, :], v.T) 

    # Add eigenvector part only if non-zero backward signal is present.
    # This can avoid NaN results for degenerate cases if the function depends
    # on the eigenvalues only.
    if np.any(vg):
        off_diag = np.ones((N, N)) - np.eye(N)
        F = off_diag / (w_repeated.T - w_repeated + np.eye(N))
        vjp_temp += np.dot(np.dot(vc, F * np.dot(v.T, vg)), v.T)
    else:
        assert 0

    off_diag_mask = (onp.ones((3,3)) - onp.eye(3))/2

    return vjp_temp*np.eye(vjp_temp.shape[-1]) + (vjp_temp + vjp_temp.T)*off_diag_mask
    # return vjp_temp*np.eye(vjp_temp.shape[-1]) + (vjp_temp + vjp_temp.T) * tri



def simplified_u(a_tensor, b_tensor):
    a_eval, a_evec = np.linalg.eigh(a_tensor)
    b_eval, b_evec = np.linalg.eigh(b_tensor)
    r = np.matmul(np.transpose(a_evec), b_evec)
    I = np.eye(3)
    rI = r*I # 3x3 -> 3x3
    pos = np.sum(rI, axis=-1)
    neg = np.sum(-rI, axis=-1)
    acos_pos = np.arccos(pos)
    acos_neg = np.arccos(neg)
    # [a,b,c]
    # [d,e,f]
    # -------
    # [min(a,d), min(b,e), min(c,f)]
    a = np.amin([acos_pos, acos_neg], axis=0)
    return np.sum(a*a)

def test_force(a_tensor, b_tensor):
    # a_eval, a_evec = np.linalg.eigh(a_tensor)
    # b_eval, b_evec = np.linalg.eigh(b_tensor)

    a_eval, a_evec = evp.dsyevv3(a_tensor)
    b_eval, b_evec = evp.dsyevv3(b_tensor)


    # print("ref w", a_eval)
    # print("test w", evp.dsyevc3(a_tensor))

    # print("ref v", a_evec)
    # print("test v", evp.dsyevv3(a_tensor))

    # assert 0

    r = np.matmul(np.transpose(a_evec), b_evec)
    I = np.eye(3)
    rI = r*I # 3x3 -> 3x3
    pos = np.sum(rI, axis=-1) # 3x3 -> 3
    neg = -np.sum(rI, axis=-1) # 3x3 -> 3
    acos_pos = np.arccos(pos) # 3 -> 3
    acos_neg = np.arccos(neg) # 3 -> 3
    a = np.amin([acos_pos, acos_neg], axis=0) # 2x3 -> 3
    a2 = a*a # 3->3
    l = np.sum(a2) # 3->1

    # derivatives, start backprop
    dl_da2 = np.ones(3) # 1 x 3
    da2_da = 2*a*np.eye(3) # 3 x 3
    da_darg = np.stack([
        np.eye(3)*(acos_pos < acos_neg),
        np.eye(3)*(acos_neg < acos_pos)
    ])

    darg_dpn = np.stack([
        np.eye(3)*(-1/np.sqrt(1-pos*pos)),
        np.eye(3)*(-1/np.sqrt(1-neg*neg))
    ])

    dl_darg = np.matmul(np.matmul(dl_da2, da2_da), da_darg)
    dpos = dl_darg[0]*(-1/np.sqrt(1-pos*pos))
    dneg = dl_darg[1]*(-1/np.sqrt(1-neg*neg))
    dneg = -dneg

    dpn_dr = np.array([
        [[1,1,1],
         [0,0,0],
         [0,0,0]],
        [[0,0,0],
         [1,1,1],
         [0,0,0]],
        [[0,0,0],
         [0,0,0],
         [1,1,1]],
    ])

    # element wise
    dr = (np.matmul(dpos, dpn_dr) + np.matmul(dneg, dpn_dr)) * np.eye(3)

    dr_daevec = np.matmul(b_evec, dr.T)
    dr_dbevec = np.matmul(a_evec, dr.T)

    dl_datensor = grad_eigh(a_eval, a_evec, np.zeros_like(a_eval), dr_daevec)
    dl_dbtensor = grad_eigh(b_eval, b_evec, np.zeros_like(b_eval), dr_dbevec)

    return dl_datensor, dl_dbtensor

 
def test1():

    grad_fn = jax.jacobian(simplified_u, argnums=(0,1))

    for _ in range(10):
        N = 50
        x_a = onp.random.rand(N,3)
        x_b = onp.random.rand(N,3)


        a_com, a_tensor = inertia_tensor(x_a, onp.ones(N, dtype=np.float64))
        b_com, b_tensor = inertia_tensor(x_b, onp.ones(N, dtype=np.float64))


        a_evec = special_ortho_group.rvs(3)
        b_evec = special_ortho_group.rvs(3)

        rf = onp.asarray(grad_fn(a_tensor, b_tensor))
        tf = onp.asarray(test_force(a_tensor, b_tensor))

        onp.testing.assert_almost_equal(rf, tf, decimal=5)

def test0():

    onp.random.seed(2020)

    for trip in range(10):
        print("trip", trip)
        N = 50
        x_a = onp.random.rand(N,3)

        a_com, a_tensor = inertia_tensor(x_a, onp.ones(N, dtype=np.float64))

        onp_res = onp.linalg.eigh(a_tensor)
        w = onp_res[0]
        Q = onp_res[1]
        for d in range(3):
            onp.testing.assert_almost_equal(
                np.matmul(a_tensor, Q[:, d]),
                w[d]*Q[:, d]
            )

        jnp_res = np.linalg.eigh(a_tensor)
        evp_res = evp.dsyevv3(a_tensor)


        np.set_printoptions(formatter={'float': lambda x: "{0:0.16f}".format(x)})

        onp.testing.assert_almost_equal(onp_res[0], jnp_res[0])
        onp.testing.assert_almost_equal(onp_res[1], jnp_res[1])

        onp.testing.assert_almost_equal(onp_res[0], evp_res[0])
        onp.testing.assert_almost_equal(onp.abs(onp_res[1]), onp.abs(evp_res[1]))


# test0()
test1()