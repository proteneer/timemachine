# runs a system.
import os
import numpy as np

from timemachine.lib import custom_ops, ops

def simulate(system, precision, gpu_idx, pipe):

    # try:
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

    ctxt.forward_mode()

    full_du_dls = stepper.get_du_dl() # [FxT]
    energies = stepper.get_energies()

    pipe.send((full_du_dls, energies))


    pipe.close()

        # for e_idx, e in enumerate(energies):
        #     if e_idx % 100 == 0:
        #         print(e_idx, e)

        # return ctxt.get_all_coords()
    # except Exception as e:
        # print("FATAL", e)