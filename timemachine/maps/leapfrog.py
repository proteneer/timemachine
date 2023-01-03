import jax


def construct_leapfrog_map(force_fxn, lambdas, masses=1.0, dt=0.1):
    """construct a volume-preserving map on augmented space (x,v)
    using a sequence of vector valued functions X -> X
    (represented by [force_fxn(x, lam) for lam in lambdas])
    """

    def update(xvf, lam):
        x_prev, v_prev, f_prev = xvf

        v_mid = v_prev + 0.5 * (dt / masses) * f_prev
        x = x_prev + dt * v_mid
        f = force_fxn(x, lam)
        v = v_mid + 0.5 * (dt / masses) * f

        return (x, v, f), None

    def leapfrog_map(x, v):
        f = force_fxn(x, lambdas[0])
        (x_, v_, _), _ = jax.lax.scan(update, (x, v, f), lambdas)

        return (x_, v_)

    return leapfrog_map
