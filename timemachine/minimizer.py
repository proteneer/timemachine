import jax
import jax.numpy as jnp

def minimize_structure(
    energy_fn,
    optimizer,
    conf,
    iterations=200):
    opt_init, opt_update, get_params = optimizer()
    grad_fn = jax.jit(jax.grad(energy_fn, argnums=(0,)))
    opt_state = opt_init(conf)
    # use lax.scan, way faster compilation times.

    # v0
    # def apply_carry(carry, _):
    #     i, x = carry
    #     g = grad_fn(get_params(x))[0]
    #     new_state = opt_update(i, g, x)
    #     new_carry = (i+1, new_state)
    #     return new_carry, _

    # carry_final, _ = jax.lax.scan(
    #     apply_carry,
    #     (jnp.array(0), opt_state),
    #     jnp.zeros((iterations, 0))
    # )

    # trip, opt_final = carry_final

    # v1
    # def apply_carry(x, i):
    #     g = grad_fn(get_params(x))[0]
    #     return opt_update(i, g, x), i

    # opt_state, _ = jax.lax.scan(
    #     apply_carry,
    #     opt_state,
    #     jnp.arange(iterations)
    # )

    # opt_final = opt_state

    # v2
    for i in range(iterations):
        g = grad_fn(get_params(opt_state))[0]
        opt_state = opt_update(i, g, opt_state)

    return get_params(opt_state)