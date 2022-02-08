# (ytz): eigenvalue problem solvers come from
# ported from Numerical diagonalization of 3x3 matrcies (sic), v1.1
# https://www.mpi-hd.mpg.de/personalhomes/globes/3x3/index.html
import jax
import jax.numpy as np
import numpy as onp

DBL_EPSILON = 2.2204460492503131e-16


def dsyevc3(A):

    de = A[0][1] * A[1][2]
    dd = onp.square(A[0][1])
    ee = onp.square(A[1][2])
    ff = onp.square(A[0][2])
    m = A[0][0] + A[1][1] + A[2][2]
    c1 = (A[0][0] * A[1][1] + A[0][0] * A[2][2] + A[1][1] * A[2][2]) - (dd + ee + ff)
    c0 = A[2][2] * dd + A[0][0] * ee + A[1][1] * ff - A[0][0] * A[1][1] * A[2][2] - 2.0 * A[0][2] * de

    p = onp.square(m) - 3.0 * c1
    q = m * (p - (3.0 / 2.0) * c1) - (27.0 / 2.0) * c0
    sqrt_p = onp.sqrt(onp.fabs(p))

    phi = 27.0 * (0.25 * onp.square(c1) * (p - c1) + c0 * (q + 27.0 / 4.0 * c0))
    phi = (1.0 / 3.0) * onp.arctan2(onp.sqrt(onp.fabs(phi)), q)

    c = sqrt_p * onp.cos(phi)
    s = (1.0 / onp.sqrt(3)) * sqrt_p * onp.sin(phi)

    w = onp.zeros(3)

    w[1] = (1.0 / 3.0) * (m - c)
    w[2] = w[1] + s
    w[0] = w[1] + c
    w[1] -= s

    return onp.sort(w)


def dsyevv3(input_tensor):

    A = onp.asarray(input_tensor).copy()
    w = dsyevc3(A)
    Q = onp.zeros((3, 3))  # column eigenvectors

    wmax = onp.fabs(w[0])
    if onp.fabs(w[1]) > wmax:
        wmax = onp.fabs(w[1])
    if onp.fabs(w[2]) > wmax:
        wmax = onp.fabs(w[2])
    thresh = onp.square(8.0 * DBL_EPSILON * wmax)

    # # Prepare calculation of eigenvectors
    n0tmp = onp.square(A[0][1]) + onp.square(A[0][2])
    n1tmp = onp.square(A[0][1]) + onp.square(A[1][2])
    Q[0][1] = A[0][1] * A[1][2] - A[0][2] * A[1][1]
    Q[1][1] = A[0][2] * A[0][1] - A[1][2] * A[0][0]
    Q[2][1] = onp.square(A[0][1])

    # # Calculate first eigenvector by the formula
    # #   v[0] = (A - w[0]).e1 x (A - w[0]).e2
    A[0][0] -= w[0]
    A[1][1] -= w[0]
    Q[0][0] = Q[0][1] + A[0][2] * w[0]
    Q[1][0] = Q[1][1] + A[1][2] * w[0]
    Q[2][0] = A[0][0] * A[1][1] - Q[2][1]
    norm = onp.square(Q[0][0]) + onp.square(Q[1][0]) + onp.square(Q[2][0])
    n0 = n0tmp + onp.square(A[0][0])
    n1 = n1tmp + onp.square(A[1][1])
    error = n0 * n1

    if n0 <= thresh:  # If the first column is zero, then (1,0,0) is an eigenvector
        Q[0][0] = 1.0
        Q[1][0] = 0.0
        Q[2][0] = 0.0
    elif n1 <= thresh:  # If the second column is zero, then (0,1,0) is an eigenvector
        Q[0][0] = 0.0
        Q[1][0] = 1.0
        Q[2][0] = 0.0
    elif norm < onp.square(64.0 * DBL_EPSILON) * error:  # If angle between A[0] and A[1] is too small, don't use
        # (ytz): don't handle this
        assert 0
        t = onp.square(A[0][1])
        # cross product, but calculate v ~ (1, -A0/A1, 0)
        f = -A[0][0] / A[0][1]

        if onp.square(A[1][1]) > t:
            t = onp.square(A[1][1])
            f = -A[0][1] / A[1][1]

        if onp.square(A[1][2]) > t:
            f = -A[0][2] / A[1][2]
        norm = 1.0 / onp.sqrt(1 + onp.square(f))
        Q[0][0] = norm
        Q[1][0] = f * norm
        Q[2][0] = 0.0
    else:  # This is the standard branch
        norm = onp.sqrt(1.0 / norm)
        for j in range(3):
            Q[j][0] = Q[j][0] * norm

    # Prepare calculation of second eigenvector
    t = w[0] - w[1]
    if onp.fabs(t) > 8.0 * DBL_EPSILON * wmax:
        # For non-degenerate eigenvalue, calculate second eigenvector by the formula
        #   v[1] = (A - w[1]).e1 x (A - w[1]).e2
        A[0][0] += t
        A[1][1] += t
        Q[0][1] = Q[0][1] + A[0][2] * w[1]
        Q[1][1] = Q[1][1] + A[1][2] * w[1]
        Q[2][1] = A[0][0] * A[1][1] - Q[2][1]
        norm = onp.square(Q[0][1]) + onp.square(Q[1][1]) + onp.square(Q[2][1])
        n0 = n0tmp + onp.square(A[0][0])
        n1 = n1tmp + onp.square(A[1][1])
        error = n0 * n1

        if n0 <= thresh:  # If the first column is zero, then (1,0,0) is an eigenvector
            Q[0][1] = 1.0
            Q[1][1] = 0.0
            Q[2][1] = 0.0
        elif n1 <= thresh:  # If the second column is zero, then (0,1,0) is an eigenvector
            Q[0][1] = 0.0
            Q[1][1] = 1.0
            Q[2][1] = 0.0
        elif norm < onp.square(64.0 * DBL_EPSILON) * error:
            t = onp.square(A[0][1])
            # cross product, but calculate v ~ (1, -A0/A1, 0)
            f = -A[0][0] / A[0][1]
            if onp.square(A[1][1]) > t:
                t = onp.square(A[1][1])
                f = -A[0][1] / A[1][1]
            if onp.square(A[1][2]) > t:
                f = -A[0][2] / A[1][2]
            norm = 1.0 / onp.sqrt(1 + onp.square(f))
            Q[0][1] = norm
            Q[1][1] = f * norm
            Q[2][1] = 0.0
        else:
            norm = onp.sqrt(1.0 / norm)
            for j in range(3):
                Q[j][1] = Q[j][1] * norm

    else:
        assert 0

    # Calculate third eigenvector according to
    #   v[2] = v[0] x v[1]
    Q[0][2] = Q[1][0] * Q[2][1] - Q[2][0] * Q[1][1]
    Q[1][2] = Q[2][0] * Q[0][1] - Q[0][0] * Q[2][1]
    Q[2][2] = Q[0][0] * Q[1][1] - Q[1][0] * Q[0][1]

    # (ytz): sanity check that Ax=lx
    # eigenvectors are column vectors of Q per "standard" convention
    for d in range(3):
        onp.testing.assert_almost_equal(onp.matmul(input_tensor, Q[:, d]), w[d] * Q[:, d])

    return w, Q


def recenter(conf):
    return conf - np.mean(conf, axis=0)


def inertia_tensor(conf, masses):
    xs = conf[:, 0]
    ys = conf[:, 1]
    zs = conf[:, 2]
    xx = np.average(ys * ys + zs * zs, weights=masses)
    yy = np.average(xs * xs + zs * zs, weights=masses)
    zz = np.average(xs * xs + ys * ys, weights=masses)
    xy = np.average(-xs * ys, weights=masses)
    xz = np.average(-xs * zs, weights=masses)
    yz = np.average(-ys * zs, weights=masses)
    tensor = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])

    return tensor


def grad_inertia_tensor(conf, masses, tensor_grad):
    xs = conf[:, 0]
    ys = conf[:, 1]
    zs = conf[:, 2]
    # xx = np.average(ys * ys + zs * zs, weights=masses)
    # yy = np.average(xs * xs + zs * zs, weights=masses)
    # zz = np.average(xs * xs + ys * ys, weights=masses)
    # xy = np.average(-xs * ys, weights=masses)
    # xz = np.average(-xs * zs, weights=masses)
    # yz = np.average(-ys * zs, weights=masses)
    # tensor = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])

    [[dxx, dxy, dxz], [dxy, dyy, dyz], [dxz, dyz, dzz]] = tensor_grad  #

    mass_sum = np.sum(masses)

    dxs = (dyy * 2 * xs + dzz * 2 * xs + -dxy * 2 * ys + -dxz * 2 * zs) * (masses / mass_sum)
    dys = (dzz * 2 * ys + dxx * 2 * ys + -dxy * 2 * xs + -dyz * 2 * zs) * (masses / mass_sum)
    dzs = (dxx * 2 * zs + dyy * 2 * zs + -dxz * 2 * xs + -dyz * 2 * ys) * (masses / mass_sum)

    dconf = np.stack([dxs, dys, dzs], axis=-1)

    return dconf


def inertial_restraint(conf, params, box, lamb, a_idxs, b_idxs, masses, k):

    a_conf = conf[a_idxs]
    b_conf = conf[b_idxs]

    a_masses = masses[a_idxs]
    b_masses = masses[b_idxs]

    a_com_conf = a_conf - np.average(a_conf, axis=0, weights=a_masses)
    b_com_conf = b_conf - np.average(b_conf, axis=0, weights=b_masses)

    a_tensor = inertia_tensor(a_com_conf, a_masses)
    b_tensor = inertia_tensor(b_com_conf, b_masses)

    a_eval, a_evec = np.linalg.eigh(a_tensor)
    b_eval, b_evec = np.linalg.eigh(b_tensor)

    # eigenvalues are needed for derivatives
    # a_eval, a_evec = dsyevv3(a_tensor)
    # b_eval, b_evec = dsyevv3(b_tensor)

    loss = []
    # (ytz): .T is because the eigenvectors are stored in columns
    for a, b in zip(a_evec.T, b_evec.T):
        delta = 1 - np.abs(np.dot(a, b))
        loss.append(delta * delta)

    return np.sum(loss) * k


def pmi_u(r):
    I = np.eye(3)

    loss = []
    for v, e in zip(r, I):
        a_pos = np.arccos(np.sum(v * e))  # norm is always 1
        a_neg = np.arccos(np.sum(-v * e))  # norm is always 1
        a = np.amin([a_pos, a_neg])
        loss.append(a * a)

    return np.sum(loss)


# (ytz): ported over from autograd. We never use the eigenvalues so wg is removed.
# https://github.com/HIPS/autograd/blob/c6f630a5ec18bd30f1485bc0dbbccb8664c77510/autograd/numpy/linalg.py#L115-L150
def grad_eigh(w, v, vg):
    """Gradient for eigenvalues and vectors of a symmetric matrix.

    Parameters
    ----------
    w: eigenvalues

    v: eigenvectors

    vg: adjoint eigenvectors
    """
    vc = v  # real
    N = 3
    # wg, vg = g          # Gradient w.r.t. eigenvalues, eigenvectors.
    w_repeated = np.repeat(w[..., np.newaxis], N, axis=-1)
    # Eigenvalue part (disabled)
    # vjp_temp = np.dot(vc * wg[..., np.newaxis, :], v.T)

    # Add eigenvector part only if non-zero backward signal is present.
    # This can avoid NaN results for degenerate cases if the function depends
    # on the eigenvalues only.

    if np.any(vg):
        off_diag = np.ones((N, N)) - np.eye(N)
        F = off_diag / (w_repeated.T - w_repeated + np.eye(N))
        # (this used to be += but we never do derivatives w.r.t. eigenvalues)
        vjp_temp = np.dot(np.dot(vc, F * np.dot(v.T, vg)), v.T)
    else:
        assert 0

    off_diag_mask = (onp.ones((3, 3)) - onp.eye(3)) / 2

    final = vjp_temp * np.eye(vjp_temp.shape[-1]) + (vjp_temp + vjp_temp.T) * off_diag_mask

    return final


def simplified_u(a_conf, b_conf, a_masses, b_masses):

    a_com_conf = a_conf - np.average(a_conf, axis=0, weights=a_masses)
    b_com_conf = b_conf - np.average(b_conf, axis=0, weights=b_masses)

    a_tensor = inertia_tensor(a_com_conf, a_masses)
    b_tensor = inertia_tensor(b_com_conf, b_masses)

    a_eval, a_evec = np.linalg.eigh(a_tensor)
    b_eval, b_evec = np.linalg.eigh(b_tensor)

    loss = []
    for a, b in zip(a_evec.T, b_evec.T):
        delta = 1 - np.abs(np.dot(a, b))
        loss.append(delta * delta)

    return np.sum(loss)


def analytic_restraint_force(conf, params, box, lamb, a_idxs, b_idxs, masses, k):

    a_conf = conf[a_idxs]
    b_conf = conf[b_idxs]

    a_masses = masses[a_idxs]
    b_masses = masses[b_idxs]

    a_com_conf = a_conf - np.average(a_conf, axis=0, weights=a_masses)
    b_com_conf = b_conf - np.average(b_conf, axis=0, weights=b_masses)

    a_tensor = inertia_tensor(a_com_conf, a_masses)
    b_tensor = inertia_tensor(b_com_conf, b_masses)

    # a_eval, a_evec = np.linalg.eigh(a_tensor)
    # b_eval, b_evec = np.linalg.eigh(b_tensor)

    # eigenvalues are needed for derivatives
    a_eval, a_evec = dsyevv3(a_tensor)
    b_eval, b_evec = dsyevv3(b_tensor)

    loss = []
    for a, b in zip(a_evec.T, b_evec.T):
        delta = 1 - np.abs(np.dot(a, b))
        loss.append(delta * delta)

    dl_daevec_T = []
    dl_dbevec_T = []
    for a, b in zip(a_evec.T, b_evec.T):
        delta = 1 - np.abs(np.dot(a, b))
        prefactor = -np.sign(np.dot(a, b)) * 2 * delta * k
        dl_daevec_T.append(prefactor * b)
        dl_dbevec_T.append(prefactor * a)

    dl_daevec = np.transpose(np.array(dl_daevec_T))
    dl_dbevec = np.transpose(np.array(dl_dbevec_T))

    dl_datensor = grad_eigh(a_eval, a_evec, np.array(dl_daevec))
    dl_dbtensor = grad_eigh(b_eval, b_evec, np.array(dl_dbevec))

    dl_da_com_conf = grad_inertia_tensor(a_com_conf, a_masses, dl_datensor)
    dl_db_com_conf = grad_inertia_tensor(b_com_conf, b_masses, dl_dbtensor)

    du_dx = onp.zeros_like(conf)

    du_dx[a_idxs] += dl_da_com_conf
    du_dx[b_idxs] += dl_db_com_conf

    print("ref du_dx", du_dx)
    # conservative forces are not affected by center of mass changes.
    # the vjp w.r.t. to the center of mass yields zeros since sum dx=0, dy=0 and dz=0
    # return dl_da_com_conf, dl_db_com_conf
    return du_dx


# (ytz): handwritten backpropagation to assist in C++ implementation.
def test_force(a_conf, b_conf, a_masses, b_masses):

    a_com_conf = a_conf - np.average(a_conf, axis=0, weights=a_masses)
    b_com_conf = b_conf - np.average(b_conf, axis=0, weights=b_masses)

    a_tensor = inertia_tensor(a_com_conf, a_masses)
    b_tensor = inertia_tensor(b_com_conf, b_masses)

    a_eval, a_evec = np.linalg.eigh(a_tensor)
    b_eval, b_evec = np.linalg.eigh(b_tensor)

    # eigenvalues are needed for derivatives
    # a_eval, a_evec = dsyevv3(a_tensor)
    # b_eval, b_evec = dsyevv3(b_tensor)

    loss = []
    for a, b in zip(a_evec.T, b_evec.T):
        delta = 1 - np.abs(np.dot(a, b))
        loss.append(delta * delta)

    dl_daevec_T = []
    dl_dbevec_T = []
    for a, b in zip(a_evec.T, b_evec.T):
        delta = 1 - np.abs(np.dot(a, b))
        prefactor = -np.sign(np.dot(a, b)) * 2 * delta
        dl_daevec_T.append(prefactor * b)
        dl_dbevec_T.append(prefactor * a)

    dl_daevec = np.transpose(np.array(dl_daevec_T))
    dl_dbevec = np.transpose(np.array(dl_dbevec_T))

    dl_datensor = grad_eigh(a_eval, a_evec, np.array(dl_daevec))
    dl_dbtensor = grad_eigh(b_eval, b_evec, np.array(dl_dbevec))

    dl_da_com_conf = grad_inertia_tensor(a_com_conf, a_masses, dl_datensor)
    dl_db_com_conf = grad_inertia_tensor(b_com_conf, b_masses, dl_dbtensor)

    # conservative forces are not affected by center of mass changes.
    # the vjp w.r.t. to the center of mass yields zeros since sum dx=0, dy=0 and dz=0
    return dl_da_com_conf, dl_db_com_conf


def test1():
    # test hand written backprop
    onp.random.seed(2020)
    grad_fn = jax.jacobian(simplified_u, argnums=(0, 1))

    for _ in range(10):
        N = 50
        x_a = onp.random.rand(N, 3)
        x_b = onp.random.rand(N, 3)

        a_masses = onp.random.rand(N)
        b_masses = onp.random.rand(N)

        rf = onp.asarray(grad_fn(x_a, x_b, a_masses, b_masses))
        tf = onp.asarray(test_force(x_a, x_b, a_masses, b_masses))

        # onp.testing.assert_almost_equal(rf, tf, decimal=5)
        onp.testing.assert_allclose(rf, tf, rtol=1e-5)


def test0():
    # test np.linalg.eigh against analytical eigensolver.

    onp.random.seed(2020)

    for trip in range(10):
        print("trip", trip)
        N = 50
        x_a = onp.random.rand(N, 3)

        a_com, a_tensor = inertia_tensor(x_a, onp.ones(N, dtype=np.float64))

        onp_res = onp.linalg.eigh(a_tensor)
        w = onp_res[0]
        Q = onp_res[1]
        for d in range(3):
            onp.testing.assert_almost_equal(np.matmul(a_tensor, Q[:, d]), w[d] * Q[:, d])

        jnp_res = np.linalg.eigh(a_tensor)
        evp_res = dsyevv3(a_tensor)

        np.set_printoptions(formatter={"float": lambda x: "{0:0.16f}".format(x)})

        onp.testing.assert_almost_equal(onp_res[0], jnp_res[0])
        onp.testing.assert_almost_equal(onp_res[1], jnp_res[1])

        onp.testing.assert_almost_equal(onp_res[0], evp_res[0])
        onp.testing.assert_almost_equal(onp.abs(onp_res[1]), onp.abs(evp_res[1]))
