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
    q = expm(xyz)
    x_r = rotate(x_b, q)
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

@jax.jit
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

    # trust region
    # alpha = 0.05
    # we actually set a max limit on the trust region based on distance from sphere

    # determine max radius based on pos later.
    # cur_radii = 0.1
    # 
    # for _ in range(50):

    #     H = sanitize_hess(pi) # hessian
    #     J = sanitize_grad(pi) # jacobian
    #     U = u_fn(pi)

    #     if np.linalg.norm(J) < 1e-8:
    #         break

    #     J_norm = np.linalg.norm(J)

    #     K = J.T @ H @ J

    #     if K <= 0:
    #         t = 1
    #     else:
    #         t = min(J_norm**3/(0.1*K), 1)

    #     pk = -t*(cur_radii/J_norm)*J

    #     pi = pi + pk


    def body_fun(x):

        cur_radii = 0.1

        H = sanitize_hess(x) # hessian
        J = sanitize_grad(x) # jacobian
        U = u_fn(x)
        J_norm = np.linalg.norm(J)

        K = J.T @ H @ J

        t = jax.lax.cond(K <= 0, lambda _: 1.0, lambda _: np.minimum(J_norm**3/(0.1*K), 1.0), None)

        pk = -t*(cur_radii/J_norm)*J

        x = x + pk
        return x

    def cond_fun(x):
        return np.linalg.norm(sanitize_grad(x)) > 1e-8

    return jax.lax.while_loop(cond_fun, body_fun, pi)


    print("\njax cauchy", "radii", np.linalg.norm(pi), "xyz", pi, "|doverlap/dangles|", np.linalg.norm(grad_fn(pi)))

    pi = np.array([0.0, 0.0, 0.0]) 

    res = scipy.optimize.minimize(
        v_and_grad_fn,
        pi,
        method='trust-ncg',
        jac=True,
        hess=sanitize_hess,
        options={'disp': False, 'gtol':1e-8}
    )
    print("Scipy: trust-ncg expm", "radii", np.linalg.norm(res.x), "xyz", res.x, "|doverlap/dangles|", np.linalg.norm(grad_fn(res.x)))

    # res = scipy.optimize.minimize(
    #     v_and_grad_fn,
    #     pi,
    #     method='trust-exact',
    #     jac=True,
    #     hess=sanitize_hess,
    #     options={'disp': False, 'gtol':1e-8}
    # )
    # print("Scipy: trust-exact expm", "radii", np.linalg.norm(res.x), "xyz", res.x, "|doverlap/dangles|", np.linalg.norm(grad_fn(res.x)))


    # works
    res = scipy.optimize.minimize(
        v_and_grad_fn,
        pi,
        method='BFGS',
        jac=True,
        options={'disp': False, 'gtol':1e-8}
    )

    print("Scipy: BFGS expm", "radii", np.linalg.norm(res.x), "xyz", res.x,  "|doverlap/dxyz|", np.linalg.norm(grad_fn(res.x)))
    return res.x



    for _ in range(100):

        for r in np.linspace(0.24, 0.5, 100):
            for theta in np.linspace(0, 2*np.pi, 20):
                for phi in np.linspace(0, 2*np.pi, 20):
                    pk = np.array([
                        np.sin(theta)*np.cos(phi),
                        np.sin(theta)*np.sin(phi),
                        np.cos(theta)
                    ])
                    h = hess_fn(pi - r*pk)[0][0]
                    w, v = np.linalg.eigh(h)
                    if np.all(w > 0):
                        print("FOUND PD hessian", r, theta, phi, w)

                        # newton cg time!

                        x = pi - r*pk

                        for _ in range(20):
                            hh = hess_fn(x)[0][0]
                            jj = grad_fn(x)[0]
                            pk = np.linalg.inv(hh)@jj

                            qa = np.dot(pk, pk)
                            qb = -2*np.dot(pk, x)
                            qc = np.dot(x, x) - 1

                            max_dt = (-qb + np.sqrt(qb**2 - 4*qa*qc))/(2*qa)

                            # find nearest minima
                            for alpha in np.linspace(-max_dt, max_dt, 100):
                                print("test with alpha", alpha, u_fn(x - alpha*pk), "w", np.linalg.eigh(sanitize_hess(x - alpha*pk))[0])

                            assert 0
                            # print("max_dt", max_dt)

                            # alpha, fc, gc, new_fval, old_fval, new_slope = scipy.optimize.line_search(u_fn, sanitize_grad, x, -pk, amax=max_dt-1e-8)

                            # print("fc, gc", fc, gc)

                            # x = x - alpha*pk
                            # print("minimizing, x:", x, "jac", jj, "pk", np.linalg.inv(hh)@jj, "w", np.linalg.eigh(hh)[0])
                        # print(x, linalg.norm)

                        break

        assert 0

    # def body_fn(val):

    #     x, count = val
    #     H = hess_fn(x)[0][0]
    #     J = grad_fn(x)[0]

    #     w, v = np.linalg.eigh(H)
    #     w = np.where(np.abs(w) <= 0.1, 0.1*np.sign(w), w)
    #     H_pd = v @ np.eye(3)*w @ v.T 
    #     pk = np.linalg.inv(H_pd) @ J

    #     qa = np.dot(pk, pk)
    #     qb = -2*np.dot(pk, x)
    #     qc = np.dot(x, x) - 1

    #     max_dt = (-qb + np.sqrt(qb**2 - 4*qa*qc))/(2*qa)
    #     dt = np.clip(max_dt-0.1, 0, 1)

    #     # max_dt = np.amax(np.array([max_dt_plus, max_dt_minus]))

    #     # stay away from the boundary
    #     # dt = np.amin(np.array([max_dt-0.1, 1.0]))
    #     x = x - dt*pk

    #     return (x, count+1)

    # def cond_fn(val):
    #     x, count = val
    #     return count < 100
    #     return np.linalg.norm(grad_fn(x)) > 1e-6

    # count = 0
    # x = pi

        # A = np.array([
        #     [0,    J[0],    J[1],    J[2]   ],
        #     [J[0], H[0][0], H[0][1], H[0][2]],
        #     [J[1], H[1][0], H[1][1], H[1][2]],
        #     [J[2], H[2][0], H[2][1], H[2][2]]
        # ]) # augmented Hessian

        # w, v = np.linalg.eigh(A)
        # vt = v[:, 0]
        # qk = vt[1:]/vt[0]

        # pi = pi + 0.5*qk
    # def body_fn(x):

    #     H = hess_fn(x)[0][0]
    #     J = grad_fn(x)[0]

    #     # print(x, J, u_fn(x))
    #     w, v = np.linalg.eigh(H)
    #     w = np.abs(w)
    #     H_pd = v @ np.eye(3)*w @ v.T 
    #     pk = np.linalg.inv(H_pd) @ J
    #     x = x - 0.5*pk
    #     x = x - (2*np.pi)*np.floor(x/(2*np.pi)) # shove between 0 and 2pi
    #     return x

    # def cond_fn(x):
    #     return np.linalg.norm(grad_fn(x)) > 1e-6

    res = jax.lax.while_loop(cond_fn, body_fn, pi)

    print("JAX: homebrew quaternions", res, "|doverlap/dxyz|", np.linalg.norm(grad_fn(res)), "count", count)

    return res

def foo():
        # assert 0

    assert 0

    # def body_fun(v):
    #     H = sanitize_hess(v)
    #     # H_off = np.sum(np.where(np.eye(3), 0, np.abs(H)), axis=0)
    #     # H_on = np.diag(H)
    #     # delta = H_off - H_on
    #     # eps = 1.0
    #     # delta = np.where(delta > -eps, delta+eps, 0)
    #     # H = H + np.eye(3)*delta
    #     # J = sanitize_grad(v)
    #     # v = v - np.linalg.inv(H) @ J
    #     # return v


    #     w,v = np.linalg.eigh(H)
    #     np.where(w < 0, 0, w)
    #     print(w)

    #     assert 0

    # def cond_fun(v):
    #     return np.linalg.norm(grad_fn(v)) > 1e-6

    # pi = jax.lax.while_loop(cond_fun, body_fun, pi)
    # return pi
    # x = pi

    # for _ in range(100):
    #     H = sanitize_hess(x)
    #     w, v = np.linalg.eigh(H)
    #     w = np.where(w < 0, 0, w)
    #     H_psd = v @ np.eye(3)*w @ v.T
    #     J = sanitize_grad(x)
    #     x = x - np.linalg.inv(H) @ J
    #     x = x - (2*np.pi)*np.floor(x/(2*np.pi)) # shove between 0 and 2pi

    # def body_fun(x):
    #     H = sanitize_hess(x)
    #     # w, v = np.linalg.eigh(H)
    #     # w = np.where(w < 0, 0, w)
    #     # H_psd = v @ np.eye(3)*w @ v.T
    #     J = sanitize_grad(x)
    #     x = x - np.linalg.inv(H) @ J
    #     x = x - (2*np.pi)*np.floor(x/(2*np.pi)) # shove between 0 and 2pi
    #     return x

    # def cond_fun(v):
    #     return np.linalg.norm(grad_fn(v)) > 1e-6

    # pi = jax.lax.while_loop(cond_fun, body_fun, pi)

    # return pi

    # print("Full Newton euler angles", x, "|doverlap/dangles|", np.linalg.norm(sanitize_grad(x)))
    # return pi
    # # # assert 0
    # pi = np.array([0.0, 0.0, 0.0]) # x y z = 0, w is implicitly set to 1

    # # assert 0



    # # res = ncg.minimize(
    # #     u_fn,
    # #     pi,
    # #     jac=sanitize_grad,
    # #     hess=sanitize_hess
    # # )

    # # print("JAX Newton-CG euler angles", res, "|doverlap/dangles|", np.linalg.norm(grad_fn(res)))

    # # assert 0



    # res = trust_region._minimize_trustregion_exact(
    #     u_fn,
    #     pi,
    #     jac=sanitize_grad,
    #     hess=sanitize_hess
    # )
    # print("\n")

    # print("JAX: trust-exact euler angles", res, "|doverlap/dangles|", np.linalg.norm(grad_fn(res)))

    # # assert 0

    # res = scipy.optimize.minimize(
    #     v_and_grad_fn,
    #     pi,
    #     method='trust-exact',
    #     jac=True,
    #     hess=sanitize_hess,
    #     options={'disp': False, 'gtol':1e-8}
    # )
    # print("Scipy: trust-exact euler angles", res.x, "|doverlap/dangles|", np.linalg.norm(grad_fn(res.x)))
    # # assert 0

    jax_trust_ncg = functools.partial(trust_region._minimize_trust_ncg, fun=u_fn, jac=sanitize_grad, hess=sanitize_hess)
    jax_trust_ncg = jax.jit(jax_trust_ncg)

    res = jax_trust_ncg(x0=pi)

    print(res)

    assert 0


    res = trust_region._minimize_trust_ncg(
        u_fn,
        pi,
        jac=sanitize_grad,
        hess=sanitize_hess
    )

    print("JAX: trust-ncg euler angles", res, "|doverlap/dangles|", np.linalg.norm(grad_fn(res)))

    # assert 0

    res = scipy.optimize.minimize(
        v_and_grad_fn,
        pi,
        method='trust-ncg',
        jac=True,
        hess=sanitize_hess,
        options={'disp': False, 'gtol':1e-8}
    )
    print("Scipy: trust-ncg euler angles", res.x, "|doverlap/dangles|", np.linalg.norm(grad_fn(res.x)))
    # assert 0




    return res.x


    # works
    res = scipy.optimize.minimize(
        v_and_grad_fn,
        pi,
        method='Newton-CG',
        jac=True,
        hess=sanitize_hess,
        options={'disp': False}
    )
    print("Newton-CG euler angles", res.x, "|doverlap/dangles|", np.linalg.norm(grad_fn(res.x)))
    # return res.x

    # works
    res = scipy.optimize.minimize(
        v_and_grad_fn,
        pi,
        method='BFGS',
        jac=True,
        options={'disp': False, 'gtol':1e-8}
    )

    print("Scipy-BFGS euler angles", res.x, "|doverlap/dangles|", np.linalg.norm(grad_fn(res.x)))
    return res.x


    res = tfp.optimizer.bfgs_minimize(
        initial_position=pi,
        value_and_gradients_function=v_and_grad_fn
    )

    print("TFP BFGS euler angles", res.position, "|doverlap/dangles|", np.linalg.norm(grad_fn(res.position)))

    return res.position
    assert 0
    return res.position

    for count in range(50):
        fx, gx = v_and_grad_fn(pi)
        hx = sanitize_hess(pi)
        pk = np.matmul(np.linalg.inv(hx), gx)

        # qa = np.dot(pk, pk)
        # qb = -2*np.dot(pk, pi)
        # qc = np.dot(pi, pi) - 1

        # this can be solved analytically
        # roots = np.real(np.roots(np.array([qa, qb, qc]))) # assert imag part being zero
        # max_dt = np.amax(roots) # complement is just -max_root+2 (2 is diameter of sphere)

        # for dt in np.linspace(0, max_dt, 100):
        #     tpi = pi - dt*pk
        #     print(dt, u_fn(tpi))

        # alpha, fc, gc, new_fval, old_fval, new_slope = scipy.optimize.line_search(u_fn, grad_fn, pi, pk, amax=max_dt-1e-8)

        # print("alpha bounded", alpha)

        dt, fc, gc, new_fval, old_fval, new_slope = scipy.optimize.line_search(u_fn, grad_fn, pi, pk)

        print("DT", dt)

        # print("alpha unbounded", alpha)

        # assert 0
        # # print("DT", dt)

        # # assert 0

        print(pi, gx)

        # dt = 1
        pi = pi - dt*pk


        # print("pi", pi, "gx", gx)


        # assert max_dt > 0

        # # print("max", u_fn(pi - max_dt*pk)) # maybe will nan depending on machine precision
        # # print("max minus eps", u_fn(pi - (max_dt - 1e-4)*pk)) # should not nan
        # # print("max plus eps", u_fn(pi - (max_dt + 1e-4)*pk)) # should nan 

        # # assert 0

        # for dt in np.linspace(0, max_dt, 100):
        #     tpi = pi - dt*pk
        #     print(dt, u_fn(tpi))

        # assert 0


    # if np.linalg.norm(grad_fn(pi)) >= 1e-6:
    print("FAILED:", pi, grad_fn(pi))

    return pi

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
    # box = 2*np.pi
    # diff -= box[d]*np.floor(np.expand_dims(diff[...,d], axis=-1)/box[d][d]+0.5)
    # results are between -pi and pi
    # p_final = np.min
    # p_final is between 0 and 2pi, so measure the circular distance
    p_final = p_final - (2*np.pi)*np.floor(p_final/(2*np.pi))
    p_final = np.where(p_final > (2*np.pi - p_final), (2*np.pi - p_final), p_final)
    # print("p_final", p_final)

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