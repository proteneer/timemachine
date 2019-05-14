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
    def apply_carry(carry, _):
        i, x = carry
        g = grad_fn(get_params(x))[0]
        new_state = opt_update(i, g, x)
        new_carry = (i+1, new_state)
        return new_carry, _

    carry_final, _ = jax.lax.scan(
        apply_carry,
        (jnp.array(0), opt_state),
        jnp.zeros((75, 0))
    )

    trip, opt_final = carry_final

    return opt_final[0][0][0]