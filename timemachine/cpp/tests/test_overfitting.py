import os
import unittest
import numpy as np

from timemachine.lib import custom_ops
from timemachine.integrator import langevin_coefficients
from system import serialize


class TestOverFit(unittest.TestCase):


    def test_overfit_host_acd(self):
        raw_potentials, coords, (params, param_groups), masses = serialize.deserialize_system('examples/host_acd.xml')

        potentials = []
        for p, args in raw_potentials:
            potentials.append(p(*args))

        num_atoms = coords.shape[0]

        dt = 0.001
        ca, cb, cc = langevin_coefficients(
            temperature=100.0,
            dt=dt,
            friction=75,
            masses=masses
        )

        # minimization coefficients
        m_dt, m_ca, m_cb, m_cc = dt, 0.5, cb, np.zeros_like(masses)

        friction = 1.0

        opt = custom_ops.LangevinOptimizer_f64(
            m_dt,
            m_ca,
            m_cb,
            m_cc
        )

        # test getting charges
        dp_idxs = np.argwhere(param_groups == 7).reshape(-1)

        ctxt = custom_ops.Context_f64(
            potentials,
            opt,
            params,
            coords, # x0
            np.zeros_like(coords), # v0
            # np.arange(len(params))
            dp_idxs
        )

        # minimize the system
        for i in range(10000):
            ctxt.step()
            if i % 100 == 0:
                print(i, ctxt.get_E())


        opt.set_dt(dt)
        opt.set_coeff_a(ca)
        opt.set_coeff_b(cb)
        opt.set_coeff_c(cc)

        # tdb reservoir sampler
        for i in range(10000):
            ctxt.step()
            if i % 100 == 0:
                print(i, ctxt.get_E())
            
