# free energy fitting - how do we want to do this?


# compute the free energy of removing a single lennard jones particle out of a 2D box
# (and we do this at varying lambda windows)



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

from scipy.special import logsumexp

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


class FreeEnergyEngine():

    def __init__(self, U_A_fn, U_B_fn, temperature):

        self.kT = BOLTZ*temperature
        self.U_A_fn = jax.jit(U_A_fn) # (x, p) -> R^1
        self.U_B_fn = jax.jit(U_B_fn) # (x, p) -> R^1

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

        num_steps = 100000

        # grad_fn = jax.grad(self.U_A_fn, argnums=(0,1))
        # grad_fn = jax.jit(grad_fn)
        dU_A_dx_fn = jax.jit(jax.grad(self.U_A_fn, argnums=(0,)))
        dU_B_dx_fn = jax.jit(jax.grad(self.U_B_fn, argnums=(0,)))

        dU_A_dp_fn = jax.jit(jax.grad(self.U_A_fn, argnums=(1,)))
        dU_B_dp_fn = jax.jit(jax.grad(self.U_B_fn, argnums=(1,)))

        def integrate_once_through(
            x_0,
            v_0,
            lj_params):

            volume = 1.0

            # simulate "target state", denominator of ratio
            x_t = np.copy(x_0)
            v_t = np.copy(v_0)

            dUB_dps = []

            for step in range(num_steps):
                dU_dx = dU_B_dx_fn(x_t, lj_params, volume)[0]

                if step % 10 == 0 and step > 10000:

                    dUB_dp = dU_B_dp_fn(x_t, lj_params, volume)[0]
                    dUB_dps.append(dUB_dp)

                noise = np.random.randn(*x_t.shape)
                v_t = ca*v_t + cb*dU_dx + cc*noise
                x_t = x_t + v_t*dt

            x_t = np.copy(x_0)
            v_t = np.copy(v_0)

            dUA_dps = []
            deltaUs = []

            # simulate "reference state", numerator of ratio
            for step in range(num_steps):
                dU_dx = dU_A_dx_fn(x_t, lj_params, volume)[0]
                # x_t = recenter(x_t, np.sqrt(volume))
                # if step % 1000 == 0:
                #     e = nrg_fn(x_t, lj_params)
                #     print("step", step, "x_t", x_t)

                if step % 10 == 0 and step > 10000:

                    dUA_dp = dU_A_dp_fn(x_t, lj_params, volume)[0]
                    dUA_dps.append(dUA_dp)

                    U_A = self.U_A_fn(x_t, lj_params, volume)
                    U_B = self.U_B_fn(x_t, lj_params, volume)

                    delta_U = U_B - U_A

                    deltaUs.append(-delta_U/self.kT)

                # if step % 10 == 0:
                    # obs = self.O_fn(x_t, lj_params)
                    # obs = self.O_fn(x_t, lj_params, volume)
                    # Os.append(obs)
                    # dU_dps.append(dU_dp)
                    # # print(dU_dp)
                    # O_dot_dU_dps.append(obs * dU_dp)
                    # dO_dp = self.dO_dp_fn(x_t, lj_params, volume)[0]
                    # dO_dps.append(dO_dp)

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


            # delta_G = -self.kT*np.log(np.mean(edus))
            # us = np.asarray(us)/len(us)
            avg_dUA_dps = np.mean(dUA_dps, axis=0)
            avg_dUB_dps = np.mean(dUB_dps, axis=0)

            print(avg_dUA_dps)
            print(avg_dUB_dps)
            delta_G = -self.kT*(logsumexp(deltaUs)-np.log(len(deltaUs)))

            return delta_G, avg_dUB_dps - avg_dUA_dps

            # return np.mean(Os, axis=0), np.mean(dU_dps, axis=0), np.mean(O_dot_dU_dps, axis=0), np.mean(dO_dps, axis=0)

        self.integrator = integrate_once_through

    def O_and_dO_dp(self, params):

        x0 = np.copy(self.conf)
        v0 = np.zeros_like(x0)

        delta_G, ddG_dp = self.integrator(x0, v0, params)

        print(delta_G, ddG_dp)

        return delta_G, ddG_dp

        # dO_dp = (avg_O*avg_dU_dp - avg_O_dot_dU_dp)/self.kT + avg_dO_dps

        # return avg_O, dO_dp

N = 25
U_B_flags = np.ones(N)
U_A_flags = np.ones(N)
U_A_flags[0] = 0 # first particle is non-interacting

U_A_fn = functools.partial(lennard_jones, lambda_flags=U_A_flags)
U_B_fn = functools.partial(lennard_jones, lambda_flags=U_B_flags)

mde = FreeEnergyEngine(U_A_fn, U_B_fn, 300.0)

# sigma = [0.2/1.122]*25
# sigma = [0.2]*25
# eps = [1.0]*25
# lj_params = np.stack([sigma, eps], axis=1)
lj_params = np.array([0.2/1.122, 1.0])
# lj_params = np.array([0.12/1.122, 1.0])


def loss_fn(O_pred):
    # return O_pred
    O_true = 5.0
    return jnp.abs(O_pred-O_true)

loss_grad_fn = jax.grad(loss_fn)

for epoch in range(100):
    print("epoch", epoch, "start params", lj_params)
    delta_G, ddG_dp = mde.O_and_dO_dp(lj_params)

    loss = loss_fn(delta_G)
    dL_dO = loss_grad_fn(delta_G)
    
    dL_dp = dL_dO * ddG_dp
    # print("dL_dO", dL_dO, "dO_dp", dO_dp, "dL_dp", dL_dp)
    sig_lr = 0.003
    eps_lr = 0.003

    lj_sig_scale = np.amax(np.abs(dL_dp[0]))/sig_lr
    lj_eps_scale = np.amax(np.abs(dL_dp[1]))/eps_lr
    lj_scale_factor = np.array([lj_sig_scale, lj_eps_scale])


    print("dL_dp", dL_dp)
    print("epoch", epoch, "loss", loss, "O", delta_G)

    raw_grad = np.asarray(dL_dp/lj_scale_factor)
    grad = np.zeros_like(raw_grad)
    grad[0] = raw_grad[0] # eps is nan
    print("grad", grad)
    # # assert 0
    lj_params -= grad
