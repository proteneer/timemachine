import numpy as onp
import jax.numpy as jnp
import jax
import unittest
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

class TestForcefield(unittest.TestCase):

    def test_minimization(self):

        mol = Chem.MolFromSmiles("CC1CCCCC1")
        mol = Chem.AddHs(mol)
        ids = AllChem.EmbedMultipleConfs(mol, numConfs=10, params=AllChem.ETKDG())
        ffo = ForceField('smirnoff99Frosst.offxml')
        nrg_fns, params = forcefield.parameterize(mol, ffo)

        params = jnp.array(params)

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

        total_nrg_fn = total_energy_wrapper(nrg_fns)
        conf = conf_to_onp(mol.GetConformer(0))
        total_nrg = total_nrg_fn(
            conf=conf,
            params=params,
            box=None
        )

        true_conf = conf_to_onp(mol.GetConformer(1))

        def loss_fn(iter_params):
            opt_conf = minimizer.minimize_structure(
                functools.partial(nrg_fns[0], params=iter_params, box=None),
                functools.partial(optimizers.sgd, 1e-6),
                conf=conf,
                iterations=500
            )
            l = rmsd.opt_rot_rmsd(opt_conf, true_conf)
            return l

        loss_grad_fn = jax.jit(jax.grad(loss_fn, argnums=(0,)))
        loss_opt_init, loss_opt_update, loss_get_params = optimizers.sgd(1e-4)
        loss_opt_state = loss_opt_init(params)

        print("before", loss_fn(loss_get_params(loss_opt_state)))


        for epoch in range(1000):
            epoch_params = loss_get_params(loss_opt_state)
            loss_grad = loss_grad_fn(epoch_params)[0]
            loss_opt_state = loss_opt_update(epoch, loss_grad, loss_opt_state)

        print("before", loss_fn(loss_get_params(loss_opt_state)))
