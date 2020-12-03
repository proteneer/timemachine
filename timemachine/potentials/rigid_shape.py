# a rigid version of the shape potential

# our goal is to compute an optimized quaternion q_opt such that
# the derivative of shape overlap w.r.t. q_i is close to zero. the
# final loss function is then defined on q_opt. The rotation
# quaternion must be of unit length

from optimizers import ncg, trust_region
import functools
import jax
import numpy as onp
import jax.numpy as np
import scipy.optimize

from timemachine.potentials import shape


from tensorflow_probability.substrates import jax as tfp

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

def rotated_normalized_overlap_q(q, x_a, x_b, params_a, params_b):
    """
    Explicit quaternion parameterization
    """
    x_r = rotate(x_b, q)
    return -shape.volume(
        x_a,
        params_a,
        x_r,
        params_b
    )

def rotated_normalized_overlap_xyz(p, x_a, x_b, params_a, params_b):
    """
    Implicit parameterization of quaternions
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
    Euler angle parameterization
    """
    x_r = rotate_euler(x_b, abc)
    return -shape.volume(
        x_a,
        params_a,
        x_r,
        params_b
    )


def expm(xyz):
    theta2 = np.dot(xyz, xyz)
    machine_eps = np.sqrt(np.finfo(np.float64).eps)

    def taylor(_):
        real = 1-theta2/8
        imag = (0.5+theta2/48)*xyz
        return real, imag

    def standard(_):
        theta = np.linalg.norm(xyz)
        real = np.cos(theta/2)
        imag = np.sin(theta/2)*(xyz/theta)
        return real, imag

    real, imag = jax.lax.cond(theta2 < machine_eps, taylor, standard, None)
    return np.array([real, imag[0], imag[1], imag[2]])


def rotated_normalized_overlap_expm(xyz, x_a, x_b, params_a, params_b):
    """
    Exponential map parameterization.
    """
    q = expm(xyz)
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

# tbd rename me
def bfgs_minimize(x_a, x_b, params_a, params_b):
    """
    Minimize the quaternion such that u_fn is a minima and grad_fn has a zero norm. Note that the returned
    quaternion is implicitly parameterized.
    """
    u_fn = functools.partial(jit_u_fn, x_a=x_a, x_b=x_b, params_a=params_a, params_b=params_b)
    grad_fn = functools.partial(jit_du_dq_fn, x_a=x_a, x_b=x_b, params_a=params_a, params_b=params_b)
    hess_fn = functools.partial(jit_d2u_dq2_fn, x_a=x_a, x_b=x_b, params_a=params_a, params_b=params_b)

    # not needed
    def sanitize_hess(v):
        return hess_fn(v)[0][0]

    def sanitize_grad(v):
        return grad_fn(v)[0]

    def v_and_grad_fn(v):
        # the onp casts are necessary for L-BFGS-B since it requires mutability
        # (scipy actually gives an inaccurate warning re: fortran ordering)
        # return onp.array(u_fn(v)), onp.array(grad_fn(v)[0])
        return u_fn(v), grad_fn(v)[0]

    pi = np.array([0.0, 0.0, 0.0]) # x y z = 0, w is implicitly set to 1

    # def body_fun_cauchy(v):
    #     x, count = v

    #     tr = 0.1

    #     H = sanitize_hess(x) # hessian
    #     J = sanitize_grad(x) # jacobian
    #     U = u_fn(x)
    #     J_norm = np.linalg.norm(J)

    #     K = J.T @ H @ J

    #     # t = jax.lax.cond(K <= 0, lambda _: 1.0, lambda _: np.minimum(J_norm**3/(tr*K), 1.0), None)
    #     t = np.where(K <= 0, 1.0, np.minimum(J_norm**3/(tr*K), 1.0))

    #     print("t", t, "K <= 0", K <= 0, "H is PD?", np.all(np.linalg.eigh(H)[0]) > 0)

    #     pk = -t*(tr/J_norm)*J

    #     x = x + pk
    #     return (x, count+1)

    def get_boundaries_intersections(z, d, trust_radius):
        """
        Solve the scalar quadratic equation ||z + t d|| == trust_radius.
        This is like a line-sphere intersection.
        Return the two values of t, sorted from low to high.
        """
        a = np.dot(d, d)
        b = 2 * np.dot(z, d)
        c = np.dot(z, z) - trust_radius**2
        sqrt_discriminant = np.sqrt(b*b - 4*a*c)

        # The following calculation is mathematically
        # equivalent to:
        # ta = (-b - sqrt_discriminant) / (2*a)
        # tb = (-b + sqrt_discriminant) / (2*a)
        # but produce smaller round off errors.
        # Look at Matrix Computation p.97
        # for a better justification.
        aux = b + np.copysign(sqrt_discriminant, b)
        ta = -aux / (2*a)
        tb = -2*c / aux

        return np.maximum(ta, tb)
        # return np.sort([ta, tb])


    # def solve_subproblem_cauchy_like(x, tr):
    #     """
    #     Standard cauchy solver except we fast escape to a newton point if its within TR.
    #     """
    #     H = sanitize_hess(x) # hessian
    #     J = sanitize_grad(x) # jacobian
    #     U = u_fn(x)
    #     J_norm = np.linalg.norm(J)

    #     K = J.T @ H @ J

    #     t = np.where(K <= 0, 1.0, np.minimum(J_norm**3/(tr*K), 1.0))

    #     if np.all(np.linalg.eigh(H)[0] > 0):

    #         cho_f = jax.scipy.linalg.cho_factor(H)
    #         newton_point = -jax.scipy.linalg.cho_solve(cho_f, J)

    #         if np.linalg.norm(newton_point) < tr:
    #             return newton_point

    #     pk = -t*(tr/J_norm)*J

    #     return pk

    def jittable_solve_subproblem_cauchy_like(x, tr):
        H = sanitize_hess(x) # hessian
        J = sanitize_grad(x) # jacobian
        U = u_fn(x)
        J_norm = np.linalg.norm(J)

        K = J.T @ H @ J

        t = np.where(K <= 0, 1.0, np.minimum(J_norm**3/(tr*K), 1.0))

        pd = np.all(np.linalg.eigh(H)[0] > 0)
        newton_point = -np.linalg.inv(H) @ J

        predicate = np.logical_and(pd, np.linalg.norm(newton_point) < tr)

        pk = -t*(tr/J_norm)*J

        return jax.lax.cond(predicate, lambda _: newton_point, lambda _: pk, None) # return pk


    # def solve_subproblem_dogleg(x, tr):

    #     H = sanitize_hess(x) # hessian
    #     J = sanitize_grad(x) # jacobian

    #     U = u_fn(x)
    #     J_norm = np.linalg.norm(J)

    #     # always project down to a positive definite cone
    #     ew, ev = np.linalg.eigh(H)


    #     # H_pd = ev @ np.eye(3)*np.abs(ew) @ ev.T

    #     ew = np.where(ew < 0, 0.0001, ew) # make this pd, psd is insufficient
    #     H_pd = ev @ np.eye(3)*ew @ ev.T

    #     # compute Newton point 
    #     cho_f = scipy.linalg.cho_factor(H_pd)
    #     newton_point = -scipy.linalg.cho_solve(cho_f, J)

    #     K = J.T @ H @ J
    #     cauchy_point = -(np.dot(J, J) / K) * J

    #     if np.linalg.norm(newton_point) < tr:
    #         print("A")
    #         return newton_point, False

    #     # If the Cauchy point is outside the trust region,
    #     # then return the point where the path intersects the boundary.
    #     cauchy_point_norm = scipy.linalg.norm(cauchy_point)
    #     if cauchy_point_norm >= tr:
    #         print("B")
    #         p_boundary = cauchy_point * (tr / cauchy_point_norm)
    #         return p_boundary, True

    #     tb = get_boundaries_intersections(
    #         cauchy_point,
    #         newton_point - cauchy_point,
    #         tr
    #     )

    #     p_boundary = cauchy_point + tb * (newton_point - cauchy_point)
    #     print("C")
    #     return p_boundary, True

    subsolver = jittable_solve_subproblem_cauchy_like

    def body_fun(v):
        x, count = v

        tr = 0.1
        p = subsolver(x, tr)

        x = x + p
        return x, count+1


    def cond_fun(v):
        x, count = v
        return np.logical_and(np.linalg.norm(sanitize_grad(x)) > 1e-8, count < 100000)

    res, count = jax.lax.while_loop(cond_fun, body_fun, (pi, 0))

    # debug version
    # v = (pi, 0)
    # while cond_fun(v):
        # v = body_fun(v)
    # res, count = v

    return res, count

@jax.custom_vjp
def q_opt(x_a, x_b, params_a, params_b):
    return bfgs_minimize(x_a, x_b, params_a, params_b)[0]

def q_opt_fwd(x_a, x_b, params_a, params_b):
    po, count = bfgs_minimize(x_a, x_b, params_a, params_b)
    return po, (po, x_a, x_b, params_a, params_b)

def q_opt_bwd(res, dl_dq):
    # (ytz): the reverse mode derivative basically uses an analytical form
    # of the implicit function theorem in explicit form.

    # this requires q_opt to have a zero gradient at the objective function.

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
    # box = 2*np.pi
    # diff -= box[d]*np.floor(np.expand_dims(diff[...,d], axis=-1)/box[d][d]+0.5)
    # results are between -pi and pi
    # p_final = np.min
    # p_final is between 0 and 2pi, so measure the circular distance
    # p_final = p_final - (2*np.pi)*np.floor(p_final/(2*np.pi))
    # p_final = np.where(p_final > (2*np.pi - p_final), (2*np.pi - p_final), p_final)

    box = 2*np.pi
    p_final = p_final - box*np.floor(p_final/box + 0.5)

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