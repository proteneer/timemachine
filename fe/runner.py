# runs a system.
import numpy as np

from timemachine.lib import custom_ops, ops

def simulate(system, precision):

    gradients = []
    force_names = []

    for grad_name, grad_args in system.gradients:
        force_names.append(grad_name)
        op_fn = getattr(ops, grad_name)
        # print(grad_name)
        # if grad_name == "GBSA" or grad_name == "Nonbonded":
        if grad_name == "GBSA":
            print("skipping", grad_name)
            continue
        grad = op_fn(*grad_args, precision=precision)
        gradients.append(grad)

    integrator = system.integrator

    for g in gradients:
        forces, du_dl, energy = g.execute_lambda(system.x0, integrator.lambs[0])
        norms = np.linalg.norm(forces, axis=1)
        highest_forces = np.argsort(norms)[::-1][:5]
        print(g, highest_forces)
        print(forces[highest_forces])


    print(integrator.dts)
    print(integrator.cas)

    # assert 0

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



    energies = stepper.get_energies()
    for e_idx, e in enumerate(energies):
        if e_idx % 100 == 0:
            print(e_idx, e)
        # if e_idx > 1600:
            # break
    # assert 0

    # x_final = ctxt.get_last_coords()[:, :3]

    # return all_xs
    # print(x_final)
    return ctxt.get_all_coords()
