from timemachine.fe import functional


def bind_potentials(potentials, params):
    """modifies potentials in-place"""
    for U, p in zip(potentials, params):
        U.bind(p)


def construct_potential(potentials, params):
    U_fn = functional.construct_differentiable_interface_fast(potentials, params)

    def potential(xvb, lam):
        return U_fn(xvb.coords, params, xvb.box, lam)

    return potential
