import functools
import unittest

from jax.config import config; config.update("jax_enable_x64", True)
from rdkit import Chem

from system import serialize
from system import forcefield
from system import simulation
from openforcefield.typing.engines.smirnoff import ForceField


from timemachine.lib import custom_ops
import timemachine.potentials.bonded
import timemachine.potentials.nonbonded

import numpy as onp

import jax
import jax.numpy as jnp

potential_map = {
    timemachine.lib.custom_ops.PeriodicTorsion_f64: timemachine.potentials.bonded.periodic_torsion,
    timemachine.lib.custom_ops.HarmonicBond_f64: timemachine.potentials.bonded.harmonic_bond,
    timemachine.lib.custom_ops.HarmonicAngle_f64: timemachine.potentials.bonded.harmonic_angle,
    timemachine.lib.custom_ops.LennardJones_f64: timemachine.potentials.nonbonded.lennard_jones,
    timemachine.lib.custom_ops.Electrostatics_f64: timemachine.potentials.nonbonded.electrostatics
}

class ReferenceLangevin():

    def __init__(self, dt, ca, cb, cc):
        self.dt = dt
        self.coeff_a = ca
        self.coeff_bs = cb
        self.coeff_cs = cc

    def step(self, x_t, v_t, dE_dx):
        noise = onp.random.rand(x_t.shape[0], x_t.shape[1])
        # (ytz): * operator isn't defined for sparse grads (resulting from tf.gather ops), hence the tf.multiply
        v_t_1 = self.coeff_a*v_t - onp.expand_dims(self.coeff_bs, axis=-1)*dE_dx + onp.expand_dims(self.coeff_cs, axis=-1)*noise
        x_t_1 = x_t + v_t_1*self.dt
        return x_t_1, v_t_1

class TestEnthalpyDerivatives(unittest.TestCase):

    def test_system_derivatives(self):
        sdf_file = open("examples/guest-1.mol2").read()
        smirnoff = ForceField("test_forcefields/smirnoff99Frosst.offxml")
        mol = Chem.MolFromMol2Block(sdf_file, sanitize=True, removeHs=False, cleanupSubstructures=True)

        guest_potentials, guest_params, guest_param_groups, guest_conf, guest_masses = forcefield.parameterize(mol, smirnoff)

        ref_nrgs = []
        test_nrgs = []

        for potential, params in guest_potentials:
            jax_potential = potential_map[potential]
            if potential == timemachine.lib.custom_ops.HarmonicBond_f64:
                jp = functools.partial(jax_potential, box=None, bond_idxs=params[0], param_idxs=params[1])
            elif potential == timemachine.lib.custom_ops.HarmonicAngle_f64:
                jp = functools.partial(jax_potential, box=None, angle_idxs=params[0], param_idxs=params[1])
            elif potential == timemachine.lib.custom_ops.PeriodicTorsion_f64:
                jp = functools.partial(jax_potential, box=None, torsion_idxs=params[0], param_idxs=params[1])
            elif potential == timemachine.lib.custom_ops.LennardJones_f64:
                jp = functools.partial(jax_potential, box=None, scale_matrix=params[0], param_idxs=params[1])
            elif potential == timemachine.lib.custom_ops.Electrostatics_f64:
                jp = functools.partial(jax_potential, box=None, scale_matrix=params[0], param_idxs=params[1])
            else:
                raise ValueError("unknown functional form")

            test_nrgs.append(potential(*params))
            ref_nrgs.append(jp)

        def ref_total_nrg(conf, params):
            nrgs = []
            for p in ref_nrgs:
                nrgs.append(p(conf, params))
            return jnp.sum(nrgs)

        dp_idxs = onp.arange(len(params)).astype(onp.int32)

        def test_total_nrg(conf, params):
            nrgs = []
            for p in test_nrgs:
                res = p.derivatives(onp.expand_dims(conf, axis=0), params, dp_idxs)
                nrgs.append(res[0])
            return onp.sum(nrgs)

        num_atoms = guest_conf.shape[0]

        ref_e = ref_total_nrg(guest_conf, guest_params)
        test_e = test_total_nrg(guest_conf, guest_params)

        onp.testing.assert_almost_equal(ref_e, test_e)

        dt = 1e-3
        ca = 0.5
        cb = onp.random.rand(num_atoms)/10
        cc = onp.zeros(num_atoms)

        intg = ReferenceLangevin(dt, ca, cb, cc)

        ref_dE_dx_fn = jax.grad(ref_total_nrg, argnums=(0,))
        ref_dE_dx_fn = jax.jit(ref_dE_dx_fn)

        def integrate(x_t, v_t, params):
            for _ in range(100):
                x_t, v_t = intg.step(x_t, v_t, ref_dE_dx_fn(x_t, params)[0])
            return x_t, v_t

        v0 = onp.random.rand(num_atoms*3).reshape(num_atoms, 3)

        x_f, v_f = integrate(guest_conf, v0, guest_params)

        lo = custom_ops.LangevinOptimizer_f64(
            dt,
            ca,
            cb,
            cc
        )

        ctxt = custom_ops.Context_f64(
            test_nrgs,
            lo,
            guest_params,
            guest_conf,
            v0,
            dp_idxs
        )

        for _ in range(100):
            ctxt.step()

        onp.testing.assert_almost_equal(x_f, ctxt.get_x())
        onp.testing.assert_almost_equal(v_f, ctxt.get_v())

        grad_fn = jax.jacfwd(integrate, argnums=(2))
        dx_dp_f, dv_dp_f = grad_fn(guest_conf, v0, guest_params)
        dx_dp_f = onp.asarray(onp.transpose(dx_dp_f, (2,0,1)))
        dv_dp_f = onp.asarray(onp.transpose(dv_dp_f, (2,0,1)))

        onp.testing.assert_almost_equal(dx_dp_f[dp_idxs], ctxt.get_dx_dp())
        onp.testing.assert_almost_equal(dv_dp_f[dp_idxs], ctxt.get_dv_dp())