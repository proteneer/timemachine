import numpy as onp
import jax.numpy as jnp
import jax
import unittest
import time
from rdkit import Chem
from rdkit.Chem import AllChem

import functools

from ff import forcefield
from timemachine import minimizer
from openforcefield.typing.engines.smirnoff import ForceField

from jax.experimental import optimizers
from timemachine.observables import rmsd

def conf_to_onp(conformer):
    coords = []
    for i in range(conformer.GetNumAtoms()):
        coords.append(conformer.GetAtomPosition(i))
    return onp.array(coords)/10 # convert to nm for MD's sake


@jax.jit
def jvp_func(fn, c, p):
    return jax.jvp(fn, (c, p), (jnp.ones_like(c), jnp.ones_like(p)))


class TestForcefield(unittest.TestCase):

    def test_minimization(self):

        mol = Chem.MolFromSmiles("CC1CCCCC1")
        mol = Chem.AddHs(mol)
        ids = AllChem.EmbedMultipleConfs(mol, numConfs=10, params=AllChem.ETKDG())
        ffo = ForceField('smirnoff99Frosst.offxml')
        nrg_fns, params = forcefield.parameterize(mol, ffo)

        params = jnp.array(params)
        conf = conf_to_onp(mol.GetConformer(0))

        def total_energy_wrapper(nrg_fns):
            """
            Returns a function that calls all functions in nrg_fns and sums them
            """
            def wrapped(*args, **kwargs):
                nrgs = []
                for fn in nrg_fns:
                    nrgs.append(fn(*args, **kwargs))
                return jnp.sum(nrgs)

            return wrapped

        # generate a reduce set of potentials
        print("NRG_FNs", nrg_fns)

        total_nrg_fn = total_energy_wrapper(nrg_fns)
        total_nrg_fn = functools.partial(total_nrg_fn, box=None)

        total_nrg = total_nrg_fn(
            conf=conf,
            params=params,
            box=None
        )

        # true_conf = conf_to_onp(mol.GetConformer(1))
        true_conf = onp.array([
            [4.8256, -1.0016, -2.0249],
            [4.9784, -0.1068, -0.7656],
            [3.6325,  0.5405, -0.3506],
            [2.9916,  1.3180, -1.5284],
            [2.8322,  0.4279, -2.7902],
            [4.1821, -0.2272, -3.2161],
            [4.0064, -1.1384, -4.4603],
            [4.2110, -1.8708, -1.7720],
            [5.8137, -1.3686, -2.3167],
            [5.7094,  0.6814, -0.9704],
            [5.3607, -0.7077,  0.0635],
            [3.8029,  1.2249,  0.4841],
            [2.9436, -0.2371, -0.0086],
            [2.0120,  1.6938, -1.2222],
            [3.6178,  2.1814, -1.7735],
            [2.4491,  1.0443, -3.6084],
            [2.0903, -0.3500, -2.5862],
            [4.8685,  0.5760, -3.5035],
            [3.3285, -1.9674, -4.2433],
            [4.9721, -1.5463, -4.7663],
            [3.5991, -0.5616, -5.2934]
        ], dtype=onp.float64)

        def loss_fn(iter_params):
            opt_conf = minimizer.minimize_structure(
                functools.partial(total_nrg_fn, box=None),
                functools.partial(optimizers.sgd, 1e-6),
                conf=conf,
                params=iter_params,
                iterations=10000,
            )
            # return jnp.sum(opt_conf)
            return rmsd.opt_rot_rmsd(opt_conf, true_conf)

        print("START")

        for epoch in range(100):
            st = time.time()
            loss_fn(params)
            print(epoch, time.time()-st)

        assert 0

        loss_grad_fn = jax.jit(jax.grad(loss_fn, argnums=(0,)))
        # loss_grad_fn = jax.jit(jax.jacfwd(loss_fn, argnums=(0,)))
        # loss_grad_fn = jax.jacfwd(loss_fn, argnums=(0,))
        loss_opt_init, loss_opt_update, loss_get_params = optimizers.sgd(1e-4)
        loss_opt_state = loss_opt_init(params)

        print("before", loss_fn(loss_get_params(loss_opt_state)))

        for epoch in range(100):
            start_time = time.time()
            epoch_params = loss_get_params(loss_opt_state)
            print(epoch)
            loss_grad = loss_grad_fn(epoch_params)[0]
            loss_opt_state = loss_opt_update(epoch, loss_grad, loss_opt_state)
            print("time per epoch:", time.time() - start_time)

        print("after", loss_fn(loss_get_params(loss_opt_state)))
