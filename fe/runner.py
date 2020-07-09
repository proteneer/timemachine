# runs a system (usually called in a separate process)

import time
import os
import numpy as np

from timemachine.lib import custom_ops, ops

def simulate(system, precision, gpu_idx, pipe):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
    gradients = []
    force_names = []

    for grad_name, grad_args in system.gradients:
        force_names.append(grad_name)
        op_fn = getattr(ops, grad_name)
        grad = op_fn(*grad_args, precision=precision)
        gradients.append(grad)

    integrator = system.integrator

    stepper = custom_ops.AlchemicalStepper_f64(
        gradients,
        integrator.lambs
    )

    ctxt = custom_ops.ReversibleContext_f64(
        stepper,
        system.x0,
        system.v0,
        integrator.cas,
        integrator.cbs,
        integrator.ccs,
        integrator.dts,
        integrator.seed
    )

    start = time.time()
    ctxt.forward_mode()
    print("fwd run time", time.time() - start)

    full_du_dls = stepper.get_du_dl() # [FxT]
    energies = stepper.get_energies()

    pipe.send((full_du_dls, energies))

    du_dl_adjoints = pipe.recv()

    dL_dps = []

    if du_dl_adjoints is not None:
        stepper.set_du_dl_adjoint(du_dl_adjoints)
        ctxt.set_x_t_adjoint(np.zeros_like(system.x0))
        start = time.time()
        print("start backwards mode")
        ctxt.backward_mode()
        print("bkwd run time", time.time() - start)
        # not a valid method, grab directly from handlers

        # note that we have multiple HarmonicBonds/Angles/Torsions that correspond to different parameters
        for f_name, g in zip(force_names, gradients):
            if f_name == 'HarmonicBond':
                dL_dps.append(g.get_du_dp_tangents())
            elif f_name == 'HarmonicAngle':
                dL_dps.append(g.get_du_dp_tangents())
            elif f_name == 'PeriodicTorsion':
                dL_dps.append(g.get_du_dp_tangents())
            elif f_name == 'Nonbonded':
                dL_dps.append((g.get_du_dcharge_tangents(), g.get_du_dlj_tangents()))
            elif f_name == 'GBSA':
                dL_dps.append((g.get_du_dcharge_tangents(), g.get_du_dgb_tangents()))
            elif f_name == 'Restraint':
                dL_dps.append(g.get_du_dp_tangents())
            else:
                raise Exception("Unknown Gradient")

    pipe.send(dL_dps)

    pipe.close()
    return
