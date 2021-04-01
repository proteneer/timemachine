# can we optimize an MD engine using the thermodynamic gradient?
import jax
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as jnp

from quadpy import quad
from thermo_deriv.lj import lennard_jones
from jankmachine.integrator import langevin_coefficients
import numpy as np
import functools
from jankmachine.constants import BOLTZ

from matplotlib import pyplot as plt

class MDEngine():


    def __init__(self, U_fn, O_fn, temperature):


        self.kT = BOLTZ*temperature
        # self.temperature = temperature
        self.U_fn = U_fn # (x, p) -> R^1
        self.O_fn = O_fn # (R^1 -> R^N)        

        xs = np.linspace(0, 1.0, 3, endpoint=True)
        conf = np.expand_dims(xs, axis=1)
        self.conf = conf
        D = conf.shape[-1]

        # sigma = 0.2/1.122
        # sigma = 0.1
        # eps = 1.0

        # lj_params = np.array([sigma, eps])

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

        # setup bounded particles
        cb[0] = 0.0
        cb[-1] = 0.0
        cc[0] = 0.0
        cc[-1] = 0.0

        num_steps = 50000

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

            for step in range(num_steps):
                dU_dx, dU_dp = grad_fn(x_t, lj_params)

                # if step % 1000 == 0:
                #     e = nrg_fn(x_t, lj_params)
                #     print("step", step, "x_t", x_t)

                if step % 10 == 0 and step > 2000:
                    obs = self.O_fn(x_t, lj_params)
                    Os.append(obs)
                    dU_dps.append(dU_dp)
                    O_dot_dU_dps.append(obs * dU_dp)

                # if step % 10 == 0:
                #     e = nrg_fn(x_t, lj_params, vol_xt)
                #     # plt.xlim(0, box_length)
                #     # plt.ylim(0, box_length)
                #     # plt.scatter(xx_t[:, 0], x_t[:, 1])
                #     plt.scatter(x_t, np.zeros_like(x_t))
                #     plt.savefig('barostat_frames/'+str(step))
                #     plt.clf()

                noise = np.random.randn(*x_t.shape)
                v_t = ca*v_t + cb*dU_dx + cc*noise
                x_t = x_t + v_t*dt

            # print(observables)
            # plt.hist(observables)
            # plt.show()
            Os = np.asarray(Os)
            dU_dps = np.asarray(dU_dps)
            O_dot_dU_dps = np.asarray(O_dot_dU_dps)

            # print(Os.shape, dU_dps.shape, O_dot_dU_dps.shape)

            return np.mean(Os, axis=0), np.mean(dU_dps, axis=0), np.mean(O_dot_dU_dps, axis=0)

        self.integrator = integrate_once_through

    def O_and_dO_dp(self, params):

        x0 = np.copy(self.conf)
        v0 = np.zeros_like(x0)

        avg_O, avg_dU_dp, avg_O_dot_dU_dp = self.integrator(x0, v0, params)

        dO_dp = (avg_O*avg_dU_dp - avg_O_dot_dU_dp)/self.kT

        return avg_O, dO_dp



U_fn = jax.jit(lennard_jones)
O_fn = lambda conf, params: conf[1][0]

mde = MDEngine(U_fn, O_fn, 300.0)

sigma = [0.1, 0.2, 0.3]
eps = [1.0, 1.2, 1.3]

lj_params = np.stack([sigma, eps], axis=1)

lj_params = np.array([[ 0.46376733, 0.98690623],
 [ 0.22144344, 1.1992469 ],
 [-0.04232389, 1.30928384]])
# lj_params = np.array([0.1, 1.0])


def loss_fn(O_pred):
    O_true = 0.65
    return jnp.abs(O_pred-O_true)

loss_grad_fn = jax.grad(loss_fn)

for epoch in range(100):
    O_pred, dO_dp = mde.O_and_dO_dp(lj_params)
    loss = loss_fn(O_pred)
    dL_dO = loss_grad_fn(O_pred)
    dL_dp = dL_dO * dO_dp
    print("epoch", epoch, "params", lj_params, "loss", loss, "O", O_pred, "dL_dp", dL_dp, "dL_dO", dL_dO, "dO_dp", dO_dp)
    lj_params -= 0.1*dL_dp

# print(mde.O_and_dO_dp(lj_params))

# def loss_fn(O_pred):
#     O_true = 0.5
#     return jnp.abs(O_pred-O_true)

# loss_grad_fn = jax.grad(loss_fn)

# for epoch in range(10):
#     O_pred, dO_dp = te.O_and_dO_dp(lj_params)
#     loss = loss_fn(O_pred)
#     dL_dO = loss_grad_fn(O_pred)
#     dL_dp = dL_dO * dO_dp
#     print("epoch", epoch, "params", lj_params, "loss", loss, "O", O_pred)
    # lj_params -= 0.1*dL_dp

        # xt_noise_buffer = np.random.randn(num_steps, *conf.shape)
        # vol_noise_buffer = np.random.randn(num_steps)

        # x_final = integrate_once_through(
        #     x0,
        #     v0,
        #     vol_xt,
        #     vol_vt,
        #     lj_params,
        #     xt_noise_buffer,
        #     vol_noise_buffer
        # )
        # assert 0

        # for epoch in range(100):

        #     print(epoch, lj_params)


        #     xt_noise_buffer = np.random.randn(num_steps, *conf.shape)
        #     vol_noise_buffer = np.random.randn(num_steps)


        #     primals = (
        #         x0,
        #         v0, 
        #         vol_xt,
        #         vol_vt,
        #         lj_params,
        #         xt_noise_buffer,
        #         vol_noise_buffer
        #     )



        #     tangents = (
        #         np.zeros_like(x0),
        #         np.zeros_like(v0),
        #         np.zeros_like(vol_xt),
        #         np.zeros_like(vol_vt),
        #         # np.zeros_like(lj_params),
        #         np.array([1.0, 0.0]),
        #         np.zeros_like(xt_noise_buffer),
        #         np.zeros_like(vol_noise_buffer)
        #     )

        #     x_primals_out, x_tangents_out = jax.jvp(integrate_once_through, primals, tangents)
            
        #     sig_grad = np.clip(x_tangents_out, -0.01, 0.01)

        #     print("loss", x_primals_out, "raw_grad", x_tangents_out, "clip grad", sig_grad)


        # # raw_dU_dp_fn = jax.jit(jax.grad(lennard_jones, argnums=(1,)))
        # # def dU_dp_fn(*args, **kwargs):
        # #     res = raw_dU_dp_fn(*args, **kwargs)[0]
        # #     return res

        # # def O_dot_dU_dp_fn(*args, **kwargs):
        # #     return O_fn(*args, **kwargs)*dU_dp_fn(*args, **kwargs)

        # # self.dU_dp_fn = dU_dp_fn
        # # self.O_dot_dU_dp_fn = O_dot_dU_dp_fn

        # # self.kT = BOLTZ*temperature

        # # self.int_lower = 0.005
        # # self.int_upper = 0.995

        # # def pdf_fn(particle_coords, rv_fn, lj_params):
        # #     probs = []
        # #     for x in particle_coords:
        # #         xs = np.linspace(0, 1.0, 3, endpoint=True)
        # #         xs[1] = x
        # #         conf = np.expand_dims(xs, axis=1)
        # #         U = lennard_jones(conf, lj_params)
        # #         p = rv_fn(conf, lj_params)*np.exp(-U/self.kT)
        # #         probs.append(p)

        # #     probs = np.asarray(probs)
        # #     probs = np.moveaxis(probs, 0, -1)

        # #     return probs

        # # self.pdf_fn = pdf_fn

        # # def quad_fn(x):
        # #     v, e = quad(x, a=self.int_lower, b=self.int_upper)

        # #     assert np.all(e < 1e-6)
        # #     return v

        # # self.quad_fn = quad_fn