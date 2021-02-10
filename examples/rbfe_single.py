# This script repeatedly estimates the relative binding free energy of a single edge, along with the gradient of the
# estimate with respect to force field parameters, and adjusts the force field parameters to improve tha accuracy
# of the free energy prediction.


import argparse
import numpy as np
import jax
from jax import numpy as jnp

from fe.free_energy import construct_lambda_schedule
from fe.utils import convert_uIC50_to_kJ_per_mole
from fe import model
from md import builders

from testsystems.relative import hif2a_ligand_pair

from ff.handlers.serialize import serialize_handlers
from ff.handlers.nonbonded import AM1CCCHandler, LennardJonesHandler
from parallel.client import CUDAPoolClient

from typing import Union, Optional, Iterable, Any, Tuple, Dict

from scipy.optimize import root_scalar

array = Union[np.array, jnp.array]
Handler = Union[AM1CCCHandler, LennardJonesHandler] # TODO: do these all inherit from a Handler class already?


# TODO: move optimizer stuff into timemachine/optimize or so
def _taylor_first_order(x: array, f_x: float, grad: array) -> callable:
    """

    Notes:
        TODO: is it preferable to use jax linearize? https://jax.readthedocs.io/en/latest/jax.html#jax.linearize
    """
    def f_prime(y: array) -> float:
        return f_x + np.dot(grad, y - x)

    return f_prime


def _smart_clip(
        x: array, f_x: float, grad: array,
        step_size: float=0.1, search_direction: Optional[array]=None,
        step_lower_bound: float=0.0,
):
    """ Motivated by https://arxiv.org/abs/1903.08619 , use knowledge of a lower-bound on f_x
    to prevent from taking a step too large

    TODO: consider further damping?

    TODO: rather than truncating at absolute global bound on loss,
        consider truncating at relative bound, like, don't take a step that
        you predict would decrease the loss by more than
            X % ?
            X absolute increment?

            some combination of these?
    TODO: generalize to use local surrogates other than first-order Taylor expansions

    Notes
    -----
    * search_direction not assumed normalized. for example, it could be the raw gradient

    * `step_size` is used to generate an initial proposal `x_proposed`. If `f_prime(x_proposed) < step_lower_bound`,
        then the step will be truncated.

    * The default `step_lower_bound=0` corresponds to a suggestion in the cited study, incorporating the knowledge that
        the loss is bounded below by 0. In the script, we pass in a non-default argument to the `step_lower_bound` to
        make the behavior of the method more conservative, and this is probably something we'll fiddle with a bit.

    * The default value `step_size=0.1` isn't very precisely chosen. The behavior of the method will be insensitive to
        picking `step_size` anywhere between like 1e-3 and +inf for our problems, since this will trigger the
        step-truncating logic on most every step.
        If the `step_size` is chosen sufficiently small that it rarely produces proposals that violate `step_lower_bound`,
        then that will start to have an effect on the behavior of the optimizer.

    """

    # default search direction: SGD
    if search_direction is None:
        search_direction = - grad

    # default local surrogate model: linear
    f_prime = _taylor_first_order(x, f_x, grad)

    # default step: step_size * search_direction
    x_next = x + step_size * search_direction

    # if this is too optimistic, according to local surrogate f_prime
    if f_prime(x_next) < step_lower_bound: # TODO: replace f_prime bound with something more configurable
        x_proposed = x_next

        line_search_fxn = lambda alpha: f_prime(x + alpha * search_direction) - step_lower_bound

        result = root_scalar(line_search_fxn, x0=0, x1=step_size)
        alpha = result.root

        x_next = x + alpha * search_direction

        message = f"""
        f_prime(x_proposed) = {f_prime(x_proposed):.5f}
        using default step size {step_size:.5f}
        is lower than step_lower_bound = {step_lower_bound:.5f}
            
        truncating step size to {alpha:.5f}, 
        so that the predicted f_prime(x_next) = {f_prime(x_next):.5f}"""
        print(message)

    x_increment = np.array(x_next - x)

    return x_increment


def wrap_method(args: Iterable[Any], fxn: callable):
    # TODO: is there a more functools-y approach to make
    #   a function accept tuple instead of positional arguments?
    return fxn(*args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Relative Binding Free Energy Testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--num_gpus",
        type=int,
        help="number of gpus",
        required=True
    )

    parser.add_argument(
        "--num_complex_windows",
        type=int,
        help="number of vacuum lambda windows",
        required=True
    )

    parser.add_argument(
        "--num_solvent_windows",
        type=int,
        help="number of solvent lambda windows",
        required=True
    )

    parser.add_argument(
        "--num_equil_steps",
        type=int,
        help="number of equilibration steps for each lambda window",
        required=True
    )

    parser.add_argument(
        "--num_prod_steps",
        type=int,
        help="number of production steps for each lambda window",
        required=True
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
        'tests/data/hif2a_nowater_min.pdb')
    complex_box += np.eye(3) * 0.1  # BFGS this later

    # build the water system.
    solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(4.0)
    solvent_box += np.eye(3) * 0.1  # BFGS this later

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
        cmd_args.num_prod_steps
    )

    vg_fn = jax.value_and_grad(binding_model.loss, argnums=0)

    ordered_params = forcefield.get_ordered_params()
    ordered_handles = forcefield.get_ordered_handles()

    handle_types_being_optimized = [AM1CCCHandler, LennardJonesHandler]

    def flatten(params) -> Tuple[array, callable]:
        """Turn params dict into flat array, with an accompanying unflatten function

        TODO: note that the result is going to be in the order given by ordered_handles (filtered by presence in hrandle_types)
            rather than in the order they appear in handle_types_being_optimized

        TODO: maybe leave out the reference to handle_types_being optimized altogether

        TODO: does Jax have a pytree-based flatten / unflatten utility?
        """


        theta_list = []
        _shapes = dict()
        _handle_types = []

        for param, handle in zip(params, ordered_handles):
            assert handle.params.shape == param.shape
            key = type(handle)

            if key in handle_types_being_optimized:
                theta_list.append(param.flatten())
                _shapes[key] = param.shape
                _handle_types.append(key)

        theta = np.hstack(theta_list)

        def unflatten(theta: array) -> Dict[Handler, array]:
            params = dict()
            i = 0
            for key in _handle_types:
                shape = _shapes[key]
                num_params = int(np.prod(shape))
                params[key] = np.array(theta[i : i + num_params]).reshape(shape)
                i += num_params
            return params

        return theta, unflatten



    # in each optimization step, don't step so far that you think you're jumping to
    #   loss_next = relative_improvement_bound * loss_current
    relative_improvement_bound = 0.95

    flat_theta_traj = []
    flat_grad_traj = []
    loss_traj = []

    for epoch in range(1000):
        epoch_params = serialize_handlers(ordered_handles)
        loss, loss_grad = vg_fn(ordered_params, mol_a, mol_b, core, label_ddG)

        print("epoch", epoch, "loss", loss)

        # note: unflatten_grad and unflatten_theta have identical definitions for now
        flat_loss_grad, unflatten_grad = flatten(loss_grad)
        flat_theta, unflatten_theta = flatten(ordered_params)

        step_lower_bound = loss * relative_improvement_bound
        theta_increment = _smart_clip(flat_theta, loss, flat_loss_grad, step_lower_bound=step_lower_bound)
        param_increments= unflatten_theta(theta_increment)

        # for any parameter handler types being updated, update in place
        for handle in ordered_handles:
            handle_type = type(handle)
            if handle_type in param_increments:
                print(f'updating {handle_type.__name__}')

                print(f'\tbefore update: {handle.params}')
                handle.params += param_increments[handle_type] # TODO: careful -- this must be a "+=" or "-=" not an "="!
                print(f'\tafter update:  {handle.params}')

                # useful for debugging to dump out the grads
                # for smirks, dp in zip(handle.smirks, loss_grad):
                    # if np.any(dp) > 0:
                        # print(smirks, dp)

        # checkpoint results to npz (overwrite
        flat_theta_traj.append(np.array(flat_theta))
        flat_grad_traj.append(flat_loss_grad)
        loss_traj.append(loss)

        path_to_npz = 'results_checkpoint.npz'
        print(f'saving theta, grad, loss trajs to {path_to_npz}')
        np.savez(
            path_to_npz,
            theta_traj=np.array(flat_theta_traj),
            grad_traj=np.array(flat_grad_traj),
            loss_traj=np.array(loss_traj)
        )

        # write ff parameters after each epoch
        path_to_ff_checkpoint = f"checkpoint_{epoch}.py"
        print(f'saving force field parameter checkpoint to {path_to_ff_checkpoint}')
        with open(path_to_ff_checkpoint, 'w') as fh:
            fh.write(epoch_params)
