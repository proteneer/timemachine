import jax
import jax.numpy as jnp

def minimize_structure(
    energy_fn,
    optimizer,
    conf,
    params,
    iterations=200):
    opt_init, opt_update, get_params = optimizer()
    opt_update = jax.jit(opt_update)

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

    # v2
    # @jax.jit
    # def apply_update(opt_state, i):
    #     g = grad_fn(get_params(opt_state))[0]
    #     return opt_update(i, g, opt_state)

    # for i in range(iterations):
    #     print("inner", i)
    #     opt_state = apply_update(opt_state, i)

    # v3
    learning_rate = 1e-6

    x_grad = jnp.zeros(shape=(params.shape[0], conf.shape[0], conf.shape[1]))

    def update_fn(x_new, params):
        return -learning_rate*grad_fn(x_new, params)[0]

    @jax.jit
    def batch_mult_jvp(x, p, dxdp):
        dpdp = jnp.eye(p.shape[0])
        def apply_one(dxdp_i, dpdp_i):
            return jax.jvp(
                update_fn,
                (x, p),
                (dxdp_i, dpdp_i)
            )
        _, grads = jax.vmap(apply_one)(dxdp, dpdp)

        # print("grads", grads[0], grads[0].shape)

        return grads

    x_new = conf

    for i in range(iterations):
        print(i, energy_fn(x_new, params), "xg max/min", jnp.amax(x_grad), jnp.amin(x_grad))
        x_new = x_new + update_fn(x_new, params)
        x_grad = x_grad + batch_mult_jvp(x_new, params, x_grad)

    # print("xg max/min", jnp.amax(x_grad), jnp.amin(x_grad))

    return x_new, x_grad