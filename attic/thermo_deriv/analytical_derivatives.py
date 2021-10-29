# import numpy
# import quadpy


# def integrand(x):
#     print(x.shape)
#     res = [[numpy.sin(x), numpy.exp(x)], [numpy.log(x), numpy.exp(x)]]  # ,...
#     res = numpy.asarray(res)
#     print(res.shape)
#     return res


# res, err = quadpy.quad(integrand, 0, 1)
# print(res)
# print(err)
# assert 0


# needed for quad to converge on default settings
import jax
from jax.config import config

config.update("jax_enable_x64", True)

# from scipy.integrate import quad
from quadpy import quad
from thermo_deriv.lj_non_periodic import lennard_jones
import numpy as np
import functools


from timemachine.constants import BOLTZ

lennard_jones = jax.jit(lennard_jones)
dU_dp = jax.jit(jax.grad(lennard_jones, argnums=(1,)))


def pdf(particle_coords, observable_fn, kT):
    probs = []
    for x in particle_coords:
        xs = np.linspace(0, 1.0, 3, endpoint=True)
        xs[1] = x
        conf = np.expand_dims(xs, axis=1)

        sigma = [0.1, 0.2, 0.3]
        eps = [1.0, 1.2, 1.3]

        lj_params = np.stack([sigma, eps], axis=1)
        U = lennard_jones(conf, lj_params)
        p = observable_fn(conf, lj_params) * np.exp(-U / kT)
        probs.append(p)

    probs = np.asarray(probs)

    probs = np.moveaxis(probs, 0, -1)

    return probs

    # return observable_fn(conf, lj_params)*np.exp(-U/kT)


temperature = 300.0
kT = BOLTZ * temperature
int_lower = 0.01
int_upper = 0.99


Z_integrand = functools.partial(pdf, observable_fn=lambda conf, params: 1.0, kT=kT)

# compute the partition function with observable returning identity
Z, Z_err = quad(Z_integrand, int_lower, int_upper)

print("Z", Z)

# return the coordinate of the particle
avg_x_integrand = functools.partial(pdf, observable_fn=lambda conf, params: conf[1][0], kT=BOLTZ * temperature)

avg_x = quad(avg_x_integrand, int_lower, int_upper)[0] / Z  # 0.50000

print("<O(x)>", avg_x)

# deriv w.r.t. the 0.1 value in sigma
avg_du_dp_integrand = functools.partial(
    pdf, observable_fn=lambda conf, params: dU_dp(conf, params)[0], kT=BOLTZ * temperature
)

avg_du_dp = quad(avg_du_dp_integrand, int_lower, int_upper)[0] / Z  # 0.50000

print("<du/dp(x)>", avg_du_dp)

avg_x_dot_du_dp_integrand = functools.partial(
    pdf, observable_fn=lambda conf, params: conf[1][0] * dU_dp(conf, params)[0], kT=BOLTZ * temperature
)


avg_x_dot_du_dp = quad(avg_x_dot_du_dp_integrand, int_lower, int_upper)[0] / Z  # 0.50000
print("<O(x) du/dp(x)>", avg_x_dot_du_dp)

print("(<O(x)><du/dp(x)> - <O(x) du/dp(x)>)/kT", (avg_x * avg_du_dp - avg_x_dot_du_dp) / kT)
# compute the thermodynamic gradient

# print(avg_x)

# print("Z:", Z)
