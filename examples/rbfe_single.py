# This script repeatedly estimates the relative binding free energy of a single edge, along with the gradient of the
# estimate with respect to force field parameters, and adjusts the force field parameters to improve tha accuracy
# of the free energy prediction.


import argparse
import numpy as np
import jax
from jax import numpy as jnp

from timemachine.fe.free_energy import construct_lambda_schedule
from timemachine.fe.utils import convert_uIC50_to_kJ_per_mole
from timemachine.fe import model
from timemachine.md import builders

from testsystems.relative import hif2a_ligand_pair

from timemachine.ff.handlers.serialize import serialize_handlers
from timemachine.ff.handlers.nonbonded import AM1CCCHandler, LennardJonesHandler
from parallel.client import CUDAPoolClient
from parallel.utils import get_gpu_count

from timemachine.optimize.step import truncated_step
from timemachine.optimize.utils import flatten_and_unflatten

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Relative Binding Free Energy Testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--num_gpus", type=int, help="number of gpus", default=get_gpu_count())

    parser.add_argument("--num_complex_windows", type=int, help="number of vacuum lambda windows", required=True)

    parser.add_argument("--num_solvent_windows", type=int, help="number of solvent lambda windows", required=True)

    parser.add_argument(
        "--num_equil_steps", type=int, help="number of equilibration steps for each lambda window", required=True
    )

    parser.add_argument(
        "--num_prod_steps", type=int, help="number of production steps for each lambda window", required=True
    )

    cmd_args = parser.parse_args()

    client = CUDAPoolClient(max_workers=cmd_args.num_gpus)

    # fetch mol_a, mol_b, core, forcefield from testsystem
    mol_a, mol_b, core = hif2a_ligand_pair.mol_a, hif2a_ligand_pair.mol_b, hif2a_ligand_pair.core
    forcefield = hif2a_ligand_pair.ff

    # compute ddG label from mol_a, mol_b
    # TODO: add label upon testsystem construction
    # (ytz): these are *binding* free energies, i.e. values that are less than zero.
    label_dG_a = convert_uIC50_to_kJ_per_mole(float(mol_a.GetProp("IC50[uM](SPA)")))
    label_dG_b = convert_uIC50_to_kJ_per_mole(float(mol_b.GetProp("IC50[uM](SPA)")))
    label_ddG = label_dG_b - label_dG_a  # complex - solvent

    print("binding dG_a", label_dG_a)
    print("binding dG_b", label_dG_b)

    hif2a_ligand_pair.label = label_ddG

    # construct lambda schedules for complex and solvent
    complex_schedule = construct_lambda_schedule(cmd_args.num_complex_windows)
    solvent_schedule = construct_lambda_schedule(cmd_args.num_solvent_windows)

    # build the protein system.
    complex_system, complex_coords, _, _, complex_box, _ = builders.build_protein_system(
        "tests/data/hif2a_nowater_min.pdb"
    )

    # build the water system.
    solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(4.0)

    binding_model = model.RBFEModel(
        client,
        forcefield,
        complex_system,
        complex_coords,
        complex_box,
        complex_schedule,
        solvent_system,
        solvent_coords,
        solvent_box,
        solvent_schedule,
        cmd_args.num_equil_steps,
        cmd_args.num_prod_steps,
    )

    ordered_params = forcefield.get_ordered_params()
    ordered_handles = forcefield.get_ordered_handles()

    flatten, unflatten = flatten_and_unflatten(ordered_params)

    handle_types_being_optimized = [AM1CCCHandler, LennardJonesHandler]

    # in each optimization step, don't step so far that you think you're jumping to
    #   loss_next = relative_improvement_bound * loss_current
    relative_improvement_bound = 0.95

    flat_theta_traj = []
    flat_grad_traj = []
    loss_traj = []

    vg_fn = jax.value_and_grad(binding_model.loss, argnums=0, has_aux=True)

    for epoch in range(1000):
        epoch_params = serialize_handlers(ordered_handles)

        (loss, aux), loss_grad = vg_fn(ordered_params, mol_a, mol_b, core, label_ddG)

        print("epoch", epoch, "loss", loss)
        flat_loss_grad = flatten(loss_grad)
        flat_theta = flatten(ordered_params)

        step_lower_bound = loss * relative_improvement_bound
        theta_increment = truncated_step(flat_theta, loss, flat_loss_grad, step_lower_bound=step_lower_bound)
        param_increments = unflatten(theta_increment)

        # for any parameter handler types being updated, update in place
        for handle, increment in zip(ordered_handles, param_increments):
            handle_type = type(handle)
            if handle_type in handle_types_being_optimized:
                print(f"updating {handle_type.__name__}")

                print(f"\tbefore update: {handle.params}")
                handle.params += increment  # TODO: careful -- this must be a "+=" or "-=" not an "="!
                print(f"\tafter update:  {handle.params}")

                # useful for debugging to dump out the grads
                # for smirks, dp in zip(handle.smirks, loss_grad):
                # if np.any(dp) > 0:
                # print(smirks, dp)

        # checkpoint results to npz (overwrite
        flat_theta_traj.append(np.array(flat_theta))
        flat_grad_traj.append(flat_loss_grad)
        loss_traj.append(loss)

        path_to_npz = "results_checkpoint.npz"
        print(f"saving theta, grad, loss trajs to {path_to_npz}")
        np.savez(
            path_to_npz,
            theta_traj=np.array(flat_theta_traj),
            grad_traj=np.array(flat_grad_traj),
            loss_traj=np.array(loss_traj),
        )

        # write ff parameters after each epoch
        path_to_ff_checkpoint = f"checkpoint_{epoch}.py"
        print(f"saving force field parameter checkpoint to {path_to_ff_checkpoint}")
        with open(path_to_ff_checkpoint, "w") as fh:
            fh.write(epoch_params)
