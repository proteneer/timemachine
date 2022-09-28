import jax.numpy as jnp
import timemachine.fe.bar as tmbar


def trapz(y, x, dx=1.0, axis=-1):
    """
    Isomorphic API to numpy's trapz. Refer to np.trapz for documentation.

    y and x must be numpy arrays.
    """
    assert dx == 1.0

    d = jnp.diff(x)
    # reshape to correct shape
    shape = [1] * y.ndim
    shape[axis] = d.shape[0]
    d = d.reshape(shape)

    nd = y.ndim
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)

    ret = (d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0).sum(axis)
    return ret


def compute_work_noneq_ti(du_dls, lambda_schedule):
    """

    Parameters
    ----------
    du_dls : array
        in reduced units
    lambda_schedule

    Returns
    -------
    work array

    Notes
    -----
    * This way to compute the work can be biased.
        See https://github.com/proteneer/timemachine/pull/442#issuecomment-855904110
        for references about different ways to compute nonequilibrium work.
    """
    return trapz(du_dls, lambda_schedule)


def EXP_from_du_dls(all_du_dls, lambda_schedule, kT):
    """
    Run exponential averaging on a list of du_dls that may contain None elements.

    The inputs for du_dls should be in units of 1/kT

    Returns
    -------
    dG: float
        units of kJ/mol
    """
    proper_du_dls = []

    for d in all_du_dls:
        if d is not None:
            proper_du_dls.append(d)

    proper_du_dls = jnp.array(proper_du_dls)

    work_array = trapz(proper_du_dls, lambda_schedule)
    work_array = work_array / kT

    return tmbar.EXP(work_array) * kT