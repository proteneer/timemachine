# a rigid version of the shape potential

# our goal is to compute an optimized quaternion q_opt such that
# the derivative of shape overlap w.r.t. q_i is close to zero. the
# final loss function is then defined on q_opt. The rotation
# quaternion must be of unit length

import functools
import jax
import numpy as onp
import jax.numpy as np
import scipy.optimize

from timemachine.potentials import shape

def quat_mul(q1, q2):
    """
    Multiply two quaternions
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    return np.array([w, x, y, z])

def qmv_rhs(q1, q2):
    """
    Multiply two quaternions
    """
    # w1, x1, y1, z1 = q1
    w1 = q1[:, 0]
    x1 = q1[:, 1]
    y1 = q1[:, 2]
    z1 = q1[:, 3]
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    return np.array([w, x, y, z]).T


def qmv_lhs(q1, q2):
    """
    Multiply two quaternions
    """
    w1, x1, y1, z1 = q1
    w2 = q2[:, 0]
    x2 = q2[:, 1]
    y2 = q2[:, 2]
    z2 = q2[:, 3]
    # w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    return np.array([w, x, y, z]).T


def rotate(x, q):
    """
    Rotate a set of points by the quaternion q.
    """
    q_T = q*np.array([1,-1,-1,-1])
    N = x.shape[0]
    x_q = np.zeros((N, 4))
    x_q = jax.ops.index_update(x_q, jax.ops.index[:, 1:], x)
    # x_q[:, 1:] = x
    x_r = qmv_lhs(q, qmv_rhs(x_q, q_T))[:, 1:]
    return x_r 

def q_from_p(p):
    w = np.sqrt(1-np.dot(p, p))
    q = np.array([w, p[0], p[1], p[2]])
    return q


# original
# def rotated_normalized_overlap_3(p, x_a, x_b, params_a, params_b):
#     """
#     Leading quaternion parameterized from p via:
#     x, y, z = p
#     q = (1-sqrt(x^2+y^2+z^2), x, y, z)
#     """
#     w = np.sqrt(1-np.dot(p, p))
#     q = np.array([w, p[0], p[1], p[2]])
#     x_r = rotate(x_b, q)
#     return -shape.volume(
#         x_a,
#         params_a,
#         x_r,
#         params_b
#     )


def rotated_normalized_overlap_xyz(p, x_a, x_b, params_a, params_b):
    """
    Leading quaternion parameterized from p via:
    x, y, z = p
    q = (1-sqrt(x^2+y^2+z^2), x, y, z)
    """
    w = np.sqrt(1-np.dot(p, p))
    q = np.array([w, p[0], p[1], p[2]])
    x_r = rotate(x_b, q)
    return -shape.volume(
        x_a,
        params_a,
        x_r,
        params_b
    )

def polar_to_cartesian(rpt):
    r, phi, theta = rpt
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return np.array([x,y,z])

def cartesian_to_polar(xyz):
    x, y, z = xyz
    r = np.sqrt(x*x+y*y+z*z)
    phi = np.arctan(y/x)
    theta = np.arccos(z/r)
    return np.array([r,phi,theta])


def rotated_normalized_overlap_polar(rpt, x_a, x_b, params_a, params_b):
    """
    Leading quaternion parameterized from p via:
    x, y, z = p
    q = (1-sqrt(x^2+y^2+z^2), x, y, z)
    """
    p = polar_to_cartesian(rpt)
    w = np.sqrt(1-np.dot(p, p))
    q = np.array([w, p[0], p[1], p[2]])
    x_r = rotate(x_b, q)
    return -shape.volume(
        x_a,
        params_a,
        x_r,
        params_b
    )

def rotate_euler(x, abc):
    a,b,c = abc
    Rx = np.array([
        [1,         0,          0],
        [0, np.cos(a), -np.sin(a)],
        [0, np.sin(a),  np.cos(a)]
    ])
    Ry = np.array([
        [ np.cos(b), 0, np.sin(b)],
        [ 0        , 1,         0],
        [-np.sin(b), 0, np.cos(b)]
    ])
    Rz = np.array([
        [np.cos(c), -np.sin(c), 0],
        [np.sin(c),  np.cos(c), 0],
        [        0,          0, 1]
    ])
    R = Rx @ Ry @ Rz
    return x@R

def rotated_normalized_overlap_euler(abc, x_a, x_b, params_a, params_b):
    """
    Leading quaternion parameterized from p via:
    x, y, z = p
    q = (1-sqrt(x^2+y^2+z^2), x, y, z)
    """
    x_r = rotate_euler(x_b, abc)
    return -shape.volume(
        x_a,
        params_a,
        x_r,
        params_b
    )


def rotated_normalized_overlap_q(q, x_a, x_b, params_a, params_b):
    x_r = rotate(x_b, q)
    return -shape.volume(
        x_a,
        params_a,
        x_r,
        params_b
    )

# first order
jit_u_fn = jax.jit(rotated_normalized_overlap_euler)
jit_du_dq_fn = jax.jit(jax.grad(rotated_normalized_overlap_euler, argnums=(0,)))

# second order
jit_d2u_dq2_fn = jax.jit(jax.hessian(rotated_normalized_overlap_euler, argnums=(0,)))
# first reverse mode differentiate w.r.t. x, then fwd differentiate w.r.t. q as it's more efficient
jit_d2u_dxadq_fn = jax.jit(jax.jacfwd(jax.grad(rotated_normalized_overlap_euler, argnums=(1,)), argnums=(0,)))
jit_d2u_dxbdq_fn = jax.jit(jax.jacfwd(jax.grad(rotated_normalized_overlap_euler, argnums=(2,)), argnums=(0,)))

def bfgs_minimize(x_a, x_b, params_a, params_b):
    """
    Minimize the quaternion such that u_fn is a minima and grad_fn has a zero norm. Note that the returned
    quaternion is implicitly parameterized.
    """

    # (ytz): Several attempts were made in order satisfy the unit norm requirement for rotational
    # quaternions.

    # 1) Constrained SLSQP: while this minimizes q adequately it does not generate generate a zero norm
    # on the gradient.
    # 2) Unconstrained BFGS: attempts to find a q that results in taking the square root of a negative
    # number.
    # 3) Bounded L-BFGS-B:  This is done by parameterizing q=(sqrt(1-x^2+y^2+z^2), x, y, z).
    # this is the current implementation which bounds the individual imag components  to +/- sqrt(1/3),
    # which guarantees a unit norm. While this would be insufficient for finding a suitable
    # global minima, this is okay since we assume that a minimizing quaternion close to the identity
    # rotation can be found (as the identity is where we start from!)

    # 4) A polar attempt has been made, however the projected norm always seems to default to zero, which
    # causes massive problem for lfgs

    u_fn = functools.partial(jit_u_fn, x_a=x_a, x_b=x_b, params_a=params_a, params_b=params_b)
    grad_fn = functools.partial(jit_du_dq_fn, x_a=x_a, x_b=x_b, params_a=params_a, params_b=params_b)
    hess_fn = functools.partial(jit_d2u_dq2_fn, x_a=x_a, x_b=x_b, params_a=params_a, params_b=params_b)

    # not needed
    def sanitize_hess(v):
        h = hess_fn(v)[0][0]
        return onp.array(h)

    def v_and_grad_fn(v):
        # the onp casts are necessary for L-BFGS-B since it requires mutability
        # (scipy actually gives an inaccurate warning re: fortran ordering)
        return onp.array(u_fn(v)), onp.array(grad_fn(v)[0])

    pi = np.array([0.0, 0.0, 0.0]) # x y z = 0, w is implicitly set to 1

    res = scipy.optimize.minimize(
        v_and_grad_fn,
        pi,
        method='BFGS',
        jac=True,
        options={'disp': False, 'gtol':1e-8}
    )

    if np.linalg.norm(grad_fn(res.x)) >= 1e-6:
        print("FAILED:", res.x, grad_fn(res.x))

    return res.x



@jax.custom_vjp
def q_opt(x_a, x_b, params_a, params_b):
    return bfgs_minimize(x_a, x_b, params_a, params_b)

def q_opt_fwd(x_a, x_b, params_a, params_b):
    po = bfgs_minimize(x_a, x_b, params_a, params_b)
    return bfgs_minimize(x_a, x_b, params_a, params_b), (po, x_a, x_b, params_a, params_b)

def q_opt_bwd(res, dl_dq):
    # (ytz): the reverse mode derivative basically uses an analytical form
    # of the implicit function theorem in explicit form.

    # This can probably be made more efficient as the hessian vector product combined
    # with the mixed partial derivatives is computed in a rather inefficient way that
    # can probably be linearized out.

    po, x_a, x_b, params_a, params_b = res
    MA = jit_d2u_dxadq_fn(po, x_a, x_b, params_a, params_b)[0][0]
    MB = jit_d2u_dxbdq_fn(po, x_a, x_b, params_a, params_b)[0][0]

    # (ytz): H is *mostly* positive semi-definite, come up with proof later
    H = jit_d2u_dq2_fn(po, x_a, x_b, params_a, params_b)[0][0]
    H_inv = np.linalg.inv(H)

    dl_dxa = np.transpose(-np.matmul(MA, H_inv), axes=(2,0,1))
    dl_dxb = np.transpose(-np.matmul(MB, H_inv), axes=(2,0,1))
 
    dl_dxa = np.einsum('i,ijk->jk', dl_dq, dl_dxa)
    dl_dxb = np.einsum('i,ijk->jk', dl_dq, dl_dxb)

    return (dl_dxa, dl_dxb, None, None)


q_opt.defvjp(q_opt_fwd, q_opt_bwd)


def q_loss(x_a, x_b, params_a, params_b):
    p_final = q_opt(x_a, x_b, params_a, params_b)

    # tbd periodic distance if necessary
    # period = 2*np.pi
    # diff -= box[d]*np.floor(np.expand_dims(diff[...,d], axis=-1)/box[d][d]+0.5)
    return 100*np.dot(p_final, p_final)



def rigid_energy(conf, params, box, lamb, a_idxs, b_idxs, alphas, weights, k):

    conf_a = conf[a_idxs]
    conf_b = conf[b_idxs]

    com_a = np.mean(conf_a, axis=0)
    com_b = np.mean(conf_b, axis=0)

    conf_a = conf_a - com_a
    conf_b = conf_b - com_b

    params_c = np.stack([alphas, weights], axis=1)

    params_a = params_c[a_idxs]
    params_b = params_c[b_idxs]

    return q_loss(conf_a, conf_b, params_a, params_b) + k*np.linalg.norm(com_a - com_b)