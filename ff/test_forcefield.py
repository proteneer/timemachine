import numpy as onp
import jax.numpy as jnp
import unittest
from rdkit import Chem
from rdkit.Chem import AllChem


import functools

from ff import forcefield
from timemachine import minimizer
from openforcefield.typing.engines.smirnoff import ForceField

from jax.experimental import optimizers

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

        def loss(params):

            opt_conf = minimizer.minimize_structure(
                functools.partial(total_nrg_fn, params=params, box=None),
                functools.partial(optimizers.sgd, 1e-6),
                conf=conf,
                iterations=5000
            )
            opt_total_nrg = total_nrg_fn(
                conf=opt_conf,
                params=params,
                box=None
            )

            print(total_nrg, conf)
            print(opt_total_nrg, opt_conf)

        # test optimization of forcefield parameters.



        # print(conf)

        # print("OPT_CONF", onp.asarray(opt_conf))

        # opt_total_nrg = total_nrg_fn(
        #     conf=opt_conf,
        #     params=params,
        #     box=None
        # )



        # print(total_nrg, opt_total_nrg)

        # assert total_nrg > 0


