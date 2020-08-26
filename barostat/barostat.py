import numpy as np
import functools
import jax
from timemachine.integrator import langevin_coefficients

from potentials import lennard_jones


from matplotlib import pyplot as plt

# recenter into the home box
def recenter(conf, b):

    new_coords = []

    periodicBoxSize = np.array([
        [b, 0.],
        [0., b]
    ])

    for atom in conf:
        diff = np.array([0., 0.])
        diff += periodicBoxSize[1]*np.floor(atom[1]/periodicBoxSize[1][1]);
        diff += periodicBoxSize[0]*np.floor((atom[0]-diff[0])/periodicBoxSize[0][0]);
        new_coords.append(atom - diff)

    return np.array(new_coords)



def setup_system():

    xs = np.linspace(0, 1.0, 5, endpoint=False)
    ys = np.linspace(0, 1.0, 5, endpoint=False)

    conf = np.transpose([np.tile(xs, len(ys)), np.repeat(ys, len(xs))])
    conf += np.random.rand(*conf.shape)/20

    sigma = 0.2/1.122
    eps = 1.0

    lj_params = np.ones_like(conf)
    lj_params[:, 0] = sigma


    masses = np.ones(conf.shape[0])*6

    dt = 3e-3

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
    num_steps = 20000
    volume = 1.0
    box_length = np.cbrt(volume)

    p_ext = -25.0

    def integrate_once_through(
        x_t,
        v_t,
        vol_xt,
        vol_vt,
        lj_params):

        grad_fn = jax.grad(lennard_jones, argnums=(0,2))
        grad_fn = jax.jit(grad_fn)

        nrg_fn = jax.jit(lennard_jones)

        p_ints = []

        for step in range(num_steps):

            force, p_int = grad_fn(x_t, lj_params, vol_xt)
            p_ints.append(p_int)

            if step % 100 == 0:
                e = nrg_fn(x_t, lj_params, vol_xt)
                print("step", step, "vol_xt", vol_xt, "u", e, "p_int", p_int)
                

            if step % 1000 == 0:
                e = nrg_fn(x_t, lj_params, vol_xt)

                box_length = np.cbrt(vol_xt)
                x_centered = recenter(x_t, box_length)
                plt.xlim(0, box_length)
                plt.ylim(0, box_length)
                plt.scatter(x_centered[:, 0], x_centered[:, 1])
                plt.savefig('barostat_frames/'+str(step))
                plt.clf()

            noise = np.random.randn(*conf.shape)

            vol_noise = np.random.randn()/10
            vol_vt = 0.5*vol_vt - 0.01*(p_int - p_ext) + vol_noise
            # vol_vt = 0
            vol_xt = vol_xt + vol_vt*1.5e-3

            v_t = ca*v_t + cb*force + cc*noise
            x_t = x_t + v_t*dt

        p_ints = np.array(p_ints)
        equil_steps = num_steps//2
        print(p_ints)
        print(type(p_ints))
        print("avg pressure", np.mean(p_ints[equil_steps:]))

        return x_t

    x0 = np.copy(conf)
    v0 = np.zeros_like(x0)

    vol_xt = volume
    vol_vt = np.zeros_like(vol_xt)

    x_final = integrate_once_through(x0, v0, vol_xt, vol_vt, lj_params)

    print(x_final)
    
    # print(lj_params)
    # print(conf)
    # print(conf.shape)
    # conf = np.random.randn(N, D)*2
    # lj = np.random.randn(N, 2)/2
    # vol = 3.0




setup_system()