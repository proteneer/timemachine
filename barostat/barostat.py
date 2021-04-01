import numpy as np
import functools
import jax
import jax.numpy as jnp
from jankmachine.integrator import langevin_coefficients

from jax.config import config; config.update("jax_enable_x64", True)


from potentials import lennard_jones


from matplotlib import pyplot as plt

# recenter into the home box
# def recenter(conf, b):

#     new_coords = []

#     periodicBoxSize = jnp.array([
#         [b, 0.],
#         [0., b]
#     ])

#     for atom in conf:
#         diff = jnp.array([0., 0.])
#         diff += periodicBoxSize[1]*jnp.floor(atom[1]/periodicBoxSize[1][1]);
#         diff += periodicBoxSize[0]*jnp.floor((atom[0]-diff[0])/periodicBoxSize[0][0]);
#         new_coords.append(atom - diff)

#     return np.array(new_coords)


def recenter_to_com(conf):
    com = jnp.mean(conf, axis=0)
    return conf - com

# relative to first coordinate
def recenter_to_first_atom(conf):
    return conf - conf[0]


@jax.jit
def recenter(conf, b):

    periodicBoxSize = jnp.array([
        [b, 0.],
        [0., b]
    ])

    diff = jnp.zeros_like(conf)
    diff += jnp.expand_dims(periodicBoxSize[1], axis=0)*jnp.expand_dims(jnp.floor(conf[:, 1]/periodicBoxSize[1][1]), axis=-1)
    diff += jnp.expand_dims(periodicBoxSize[0], axis=0)*jnp.expand_dims(jnp.floor((conf[:, 0]-diff[:, 0])/periodicBoxSize[0][0]), axis=-1)

    return conf - diff


def setup_system():

    xs = np.linspace(0, 1.0, 5, endpoint=False)
    ys = np.linspace(0, 1.0, 5, endpoint=False)

    conf = np.transpose([np.tile(xs, len(ys)), np.repeat(ys, len(xs))])
    conf += np.random.rand(*conf.shape)/20

    D = conf.shape[-1]

    sigma = 0.1/1.122
    eps = 1.0

    # lj_params = np.ones_like(conf)
    # lj_params[:, 0] = sigma
    lj_params = np.array([sigma, eps])
    masses = np.ones(conf.shape[0])*1

    dt = 1.5e-3

    ca, cb, cc = langevin_coefficients(
        temperature=300.0,
        dt=dt,
        friction=1.0,
        masses=masses
    )
    cb = -np.expand_dims(cb, axis=-1)
    cc = np.expand_dims(cc, axis=-1)
    
    # minimization
    # ca = np.zeros_like(ca)
    # cc = np.zeros_like(cc)

    # print(ca, cb, cc)
    num_steps = 2000
    volume = 1.0 # or area in our case
    # box_length = np.sqrt(volume)

    p_ext = -25.0

    grad_fn = jax.grad(lennard_jones, argnums=(0,2))
    grad_fn = jax.jit(grad_fn)
    nrg_fn = jax.jit(lennard_jones)

    def integrate_once_through(
        x_t,
        v_t,
        vol_xt,
        vol_vt,
        lj_params,
        xt_noise_buf,
        vol_noise_buf):

        p_ints = []

        # coords = []
        # volumes = []

        print("initial coords", x_t)

        for step in range(num_steps):

            box_length = np.sqrt(volume)

            x_t = recenter(x_t, box_length)
            x_t = recenter_to_first_atom(x_t)

            force, p_int = grad_fn(x_t, lj_params, vol_xt)
            # p_ints.append(p_int)

            if step % 1000 == 0:
                e = nrg_fn(x_t, lj_params, vol_xt)
                print("step", step, "vol_xt", vol_xt, "u", e, "p_int", p_int)

            if step % 50 == 0:
                p_ints.append(p_int)

            if step % 100 == 0:
                e = nrg_fn(x_t, lj_params, vol_xt)
                # plt.xlim(0, box_length)
                # plt.ylim(0, box_length)
                plt.scatter(x_t[:, 0], x_t[:, 1])
                plt.savefig('barostat_frames/'+str(step))
                plt.clf()

            # vol_noise = vol_noise_buf[step]
            # vol_vt = 0.5*vol_vt - 0.01*(p_int - p_ext) + vol_noise
            # vol_xt = vol_xt + vol_vt*1.5e-3

            noise = xt_noise_buf[step]
            v_t = ca*v_t + cb*force + cc*noise
            x_t = x_t + v_t*dt

        print("final coords", x_t)

        # volumes = jnp.array(volumes)
        # expected_volume = 1.15
        # print("expected", expected_volume, "observed", jnp.mean(volumes))

        p_ints = jnp.array(p_ints)
        expected_pressure = -100.0
        computed_pressure = jnp.mean(p_ints)

        print("EP", expected_pressure, "CP", computed_pressure)

        loss = jnp.abs(expected_pressure - computed_pressure)

        return loss

    x0 = np.copy(conf)
    v0 = np.zeros_like(x0)

    vol_xt = volume
    vol_vt = np.zeros_like(vol_xt)

    xt_noise_buffer = np.random.randn(num_steps, *conf.shape)
    vol_noise_buffer = np.random.randn(num_steps)

    x_final = integrate_once_through(
        x0,
        v0,
        vol_xt,
        vol_vt,
        lj_params,
        xt_noise_buffer,
        vol_noise_buffer
    )

    assert 0

    for epoch in range(100):

        print(epoch, lj_params)


        xt_noise_buffer = np.random.randn(num_steps, *conf.shape)
        vol_noise_buffer = np.random.randn(num_steps)


        primals = (
            x0,
            v0, 
            vol_xt,
            vol_vt,
            lj_params,
            xt_noise_buffer,
            vol_noise_buffer
        )



        tangents = (
            np.zeros_like(x0),
            np.zeros_like(v0),
            np.zeros_like(vol_xt),
            np.zeros_like(vol_vt),
            # np.zeros_like(lj_params),
            np.array([1.0, 0.0]),
            np.zeros_like(xt_noise_buffer),
            np.zeros_like(vol_noise_buffer)
        )

        x_primals_out, x_tangents_out = jax.jvp(integrate_once_through, primals, tangents)
        
        sig_grad = np.clip(x_tangents_out, -0.01, 0.01)

        print("loss", x_primals_out, "raw_grad", x_tangents_out, "clip grad", sig_grad)


        # lj_params[0] -= sig_grad
    
    # print(lj_params)
    # print(conf)
    # print(conf.shape)
    # conf = np.random.randn(N, D)*2
    # lj = np.random.randn(N, 2)/2
    # vol = 3.0




setup_system()