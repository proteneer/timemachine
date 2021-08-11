# can we optimize an MD engine using the thermodynamic gradient?
import jax
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as jnp

from quadpy import quad
from thermo_deriv.lj import lennard_jones
from timemachine.integrator import langevin_coefficients
import numpy as np
import functools
from timemachine.constants import BOLTZ

from matplotlib import pyplot as plt

# recenter into the home box
def recenter(conf, b):

    new_coords = []

    periodicBoxSize = jnp.array([
        [b, 0.],
        [0., b]
    ])

    for atom in conf:
        diff = jnp.array([0., 0.])
        diff += periodicBoxSize[1]*jnp.floor(atom[1]/periodicBoxSize[1][1]);
        diff += periodicBoxSize[0]*jnp.floor((atom[0]-diff[0])/periodicBoxSize[0][0]);
        new_coords.append(atom - diff)

    return np.array(new_coords)



U_fn = jax.jit(lennard_jones)
dU_dV = jax.jit(jax.grad(lennard_jones, argnums=(2,)))


class MDEngine():


    def __init__(self, U_fn, O_fn, temperature):


        self.kT = BOLTZ*temperature
        # self.temperature = temperature
        self.U_fn = U_fn # (x, p) -> R^1
        self.O_fn = O_fn # (R^1 -> R^N)
        self.dO_dp_fn = jax.jit(jax.grad(O_fn, argnums=(1,)))

        xs = np.linspace(0, 1.0, 5, endpoint=False)
        ys = np.linspace(0, 1.0, 5, endpoint=False)
        conf = np.transpose([np.tile(xs, len(ys)), np.repeat(ys, len(xs))])

        N = conf.shape[0]
        self.conf = conf
        D = conf.shape[-1]

        masses = np.ones(conf.shape[0])
        dt = 1.5e-3

        ca, cb, cc = langevin_coefficients(
            temperature=300.0,
            dt=dt,
            friction=1.0,
            masses=masses
        )
        cb = -np.expand_dims(cb, axis=-1)
        cc = np.expand_dims(cc, axis=-1)

        num_steps = 200000

        grad_fn = jax.grad(lennard_jones, argnums=(0,1))
        grad_fn = jax.jit(grad_fn)
        nrg_fn = jax.jit(lennard_jones)

        def integrate_once_through(
            x_t,
            v_t,
            lj_params):

            Os = []
            dU_dps = []
            O_dot_dU_dps = []
            dO_dps = []

            volume = 1.0

            for step in range(num_steps):
                # lennard_jones(x_t, lj_params, volume)
                # assert 0
                dU_dx, dU_dp = grad_fn(x_t, lj_params, volume)

                # x_t = recenter(x_t, np.sqrt(volume))

                # if step % 1000 == 0:
                #     e = nrg_fn(x_t, lj_params)
                #     print("step", step, "x_t", x_t)

                if step % 10 == 0 and step > 10000:
                # if step % 10 == 0:
                    # obs = self.O_fn(x_t, lj_params)
                    obs = self.O_fn(x_t, lj_params, volume)
                    Os.append(obs)
                    dU_dps.append(dU_dp)
                    # print(dU_dp)
                    O_dot_dU_dps.append(obs * dU_dp)
                    dO_dp = self.dO_dp_fn(x_t, lj_params, volume)[0]
                    dO_dps.append(dO_dp)

                # if step % 5000 == 0:
                #     print("step", step)
                #     xx_t = recenter(x_t, np.sqrt(volume))
                #     e = nrg_fn(xx_t, lj_params, volume)
                #     plt.xlim(0, 1.0)
                #     plt.ylim(0, 1.0)
                #     plt.scatter(xx_t[:, 0], xx_t[:, 1])
                #     # plt.scatter(x_t, np.zeros_like(x_t))
                #     plt.savefig('barostat_frames/'+str(step))
                #     plt.clf()

                noise = np.random.randn(*x_t.shape)
                v_t = ca*v_t + cb*dU_dx + cc*noise
                x_t = x_t + v_t*dt

            Os = np.asarray(Os)
            dU_dps = np.asarray(dU_dps)
            O_dot_dU_dps = np.asarray(O_dot_dU_dps)

            # print(Os.shape, dU_dps.shape, O_dot_dU_dps.shape)

            return np.mean(Os, axis=0), np.mean(dU_dps, axis=0), np.mean(O_dot_dU_dps, axis=0), np.mean(dO_dps, axis=0)

        self.integrator = integrate_once_through

    def O_and_dO_dp(self, params):

        x0 = np.copy(self.conf)
        v0 = np.zeros_like(x0)

        avg_O, avg_dU_dp, avg_O_dot_dU_dp, avg_dO_dps = self.integrator(x0, v0, params)

        dO_dp = (avg_O*avg_dU_dp - avg_O_dot_dU_dp)/self.kT + avg_dO_dps

        return avg_O, dO_dp



def O_fn(conf, params, vol):
    dv = dU_dV(conf, params, vol)[0]/conf.shape[0]
    # print(dv)
    # assert 0
    return dv

    # u = U_fn(conf, params, vol)
    # return u
# O_fn = lambda conf: conf[1][0]

mde = MDEngine(U_fn, O_fn, 300.0)

# sigma = [0.2/1.122]*25
# sigma = [0.2]*25
# eps = [1.0]*25
# lj_params = np.stack([sigma, eps], axis=1)
lj_params = np.array([0.2/1.122, 1.0])


def loss_fn(O_pred):
    # return O_pred
    O_true = -20.0
    return jnp.abs(O_pred-O_true)

loss_grad_fn = jax.grad(loss_fn)

for epoch in range(100):
    print("epoch", epoch, "start params", lj_params)
    O_pred, dO_dp = mde.O_and_dO_dp(lj_params)
    loss = loss_fn(O_pred)
    dL_dO = loss_grad_fn(O_pred)
    
    dL_dp = dL_dO * dO_dp
    print("dL_dO", dL_dO, "dO_dp", dO_dp, "dL_dp", dL_dp)
    sig_lr = 0.003
    eps_lr = 0.003

    lj_sig_scale = np.amax(np.abs(dL_dp[0]))/sig_lr
    lj_eps_scale = np.amax(np.abs(dL_dp[1]))/eps_lr
    lj_scale_factor = np.array([lj_sig_scale, lj_eps_scale])
    print("dL_dp", dL_dp)
    print("epoch", epoch, "loss", loss, "O", O_pred)

    # assert 0
    lj_params -= dL_dp/lj_scale_factor
    # lj_params -= dL_dp*0.00001
