# Fit to multiple relative binding free energy edges

import jax
from jax import numpy as jnp
import numpy as np

# forcefield handlers
from ff import Forcefield
from ff.handlers.serialize import serialize_handlers
from ff.handlers.deserialize import deserialize_handlers
from ff.handlers import nonbonded

# free energy classes
from fe.free_energy import RelativeFreeEnergy, construct_lambda_schedule
from fe.model import RBFEModel

# MD initialization
from md import builders

# parallelization across multiple GPUs
from parallel.client import CUDAPoolClient

from collections import namedtuple

from pickle import load
from typing import List, Union
Handler = Union[nonbonded.AM1CCCHandler, nonbonded.LennardJonesHandler] # TODO: relax this assumption
from time import time

# how much MD to run, on how many GPUs
Configuration = namedtuple(
    'Configuration',
    ['num_gpus', 'num_complex_windows', 'num_solvent_windows', 'num_equil_steps', 'num_prod_steps'])

# define a couple configurations: one for quick tests, and one for production
production_configuration = Configuration(
    num_gpus=10,
    num_complex_windows=60,
    num_solvent_windows=60,
    num_equil_steps=10000,
    num_prod_steps=100000,
)

intermediate_configuration = Configuration(
    num_gpus=10,
    num_complex_windows=30,
    num_solvent_windows=30,
    num_equil_steps=10000,
    num_prod_steps=10000,
)

testing_configuration = Configuration(
    num_gpus=10,
    num_complex_windows=10,
    num_solvent_windows=10,
    num_equil_steps=1000,
    num_prod_steps=1000,
)

# TODO: rename this to something more descriptive than "Configuration"...
#   want to distinguish later between an "RBFE configuration"
#       (which describes the computation of a single edge)
#   and a "training configuration"
#       (which describes the overall training loop)

# don't make assumptions about working directory
from pathlib import Path

# locations relative to project root
root = Path(__file__).parent.parent.parent
path_to_protein = str(root.joinpath('tests/data/hif2a_nowater_min.pdb'))
path_to_ff = str(root.joinpath('ff/params/smirnoff_1_1_0_ccc.py'))

# locations relative to example folder
path_to_results = Path(__file__).parent
path_to_transformations = str(path_to_results.joinpath('relative_transformations.pkl'))

# load and construct forcefield
with open(path_to_ff) as f:
    ff_handlers = deserialize_handlers(f.read())

forcefield = Forcefield(ff_handlers)


# TODO: define more flexible update rules here, rather than update parameters
step_sizes = {
    nonbonded.AM1CCCHandler: 1e-3,
    nonbonded.LennardJonesHandler: 1e-3,
    # ...
}

gradient_clip_thresholds = {
    nonbonded.AM1CCCHandler: 0.001,
    nonbonded.LennardJonesHandler: np.array([0.001, 0]),  # TODO: allow to update epsilon also?
    # ...
}


def _clipped_update(gradient, step_size, clip_threshold):
    """Compute an update based on current gradient
        x[k+1] = x[k] + update

    The gradient descent update would be
        update = - step_size * grad(x[k]),

    and to avoid instability, we clip the absolute values of all components of the update
        update = - clip(step_size * grad(x[k]))

    TODO: menu of other, fancier update functions
    """
    return - np.clip(step_size * gradient, -clip_threshold, clip_threshold)


class ParameterUpdate:
    def __init__(self, before, after, gradient, update):
        self.before = before
        self.after = after
        self.gradient = gradient
        self.update = update

    # TODO: def __str__

    def save(self, name='parameter_update.npz'):
        """save numpy arrays to path/to/results/{name}.npz"""
        parameter_update_path = path_to_results.joinpath(name)
        print(f'saving parameter updates to {parameter_update_path}')
        np.savez(
            file=parameter_update_path,
            before=self.before,
            after=self.after,
            gradient=self.gradient,
            update=self.update,
        )


def _update_in_place(loss: float, loss_grads: List[jnp.array], ordered_handles: List[Handler],
                     handle_types_to_update: List[Handler] =[nonbonded.AM1CCCHandler, nonbonded.LennardJonesHandler]):
    """
    TODO: check if it's too expensive to do out-of-place updates with a copy

    TODO: want to be able to say params -= g, which could be done by abstracting the parameters + handlers blob into
        something that supports either:
        * __add__, __mul__, etc.
        * flatten() / unflatten() to / from traced arrays
    """

    unit = "kJ/mol"

    message = f"""
    loss:   {loss:.3f} {unit}
    """
    print(message)
    # TODO: restore pred, target to message
    # TODO: save these also to log file
    # TODO: save dl_dp, update, and params to disk
    # compute updates
    parameter_updates = dict()

    for loss_grad, handle in zip(loss_grads, ordered_handles):
        assert handle.params.shape == loss_grad.shape

        handle_type = type(handle)

        # TODO: make the references to handle vs. type(handle) less runtime-error-prone

        if handle_type in handle_types_to_update:
            update = _clipped_update(loss_grad, step_sizes[handle_type], gradient_clip_thresholds[handle_type])
            print(f'incrementing the {handle_type.__name__} parameters by {update}')
            handle.params += update

            # TODO: define a dict mapping from handle_type to forcefield.q_handle.params or something...
            if handle_type == nonbonded.AM1CCCHandler:
                before = np.array(forcefield.q_handle.params)  # make a copy
                forcefield.q_handle.params += update
                after = np.array(forcefield.q_handle.params)  # make a copy
            elif handle_type == nonbonded.LennardJonesHandler:
                before = np.array(forcefield.lj_handle.params)  # make a copy
                forcefield.lj_handle.params += update
                after = np.array(forcefield.lj_handle.params)  # make a copy
            else:
                message = f"Attempting to update an unsupported ff handle type: {handle_type.__name__} not in {handle_types_to_update}"
                raise (RuntimeError(message))

            parameter_updates[handle_type] = ParameterUpdate(before, after, loss_grad, update)

        # not sure if I want to print these...
        # print("before: ", before)
        # print("after: ", after)

    # TODO: also save dl_dp for the other parameter types we're not necessarily updating, for later analysis
    #   (would be easier with syntax like forcefield[handle_type])

    return parameter_updates


def _save_forcefield(filename, ff_params):
    # TODO: update path

    with open(path_to_results.joinpath(filename), 'w') as fh:
        fh.write(ff_params)


if __name__ == "__main__":
    # which force field components we'll refit
    forces_to_refit = [nonbonded.AM1CCCHandler, nonbonded.LennardJonesHandler]

    # how much computation to spend per refitting step
    # configuration = testing_configuration  # a little
    # configuration = production_configuration  # a lot
    configuration = intermediate_configuration  # goldilocks

    # how many parameter update steps
    num_parameter_updates = 1000
    # TODO: make this configurable also

    # set up multi-GPU client
    client = CUDAPoolClient(max_workers=configuration.num_gpus)

    # load pre-defined collection of relative transformations
    with open(path_to_transformations, 'rb') as f:
        relative_transformations: List[RelativeFreeEnergy] = load(f)

    # update the forcefield parameters for a few steps, each step informed by a single free energy calculation

    # compute and save the sequence of relative_transformation indices
    num_epochs = int(np.ceil(num_parameter_updates / len(relative_transformations)))
    np.random.seed(2021)
    step_inds = []
    for epoch in range(num_epochs):
        inds = np.arange(len(relative_transformations))
        np.random.shuffle(inds)
        step_inds.append(inds)
    step_inds = np.hstack(step_inds)[:num_parameter_updates]
    np.save(path_to_results.joinpath('step_indices.npy'), step_inds)

    # build the complex system
    complex_system, complex_coords, _, _, complex_box, _ = builders.build_protein_system(
        path_to_protein)
    # TODO: optimize box
    complex_box += np.eye(3) * 0.1  # BFGS this later

    # build the water system
    solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(4.0)
    # TODO: optimize box
    solvent_box += np.eye(3) * 0.1  # BFGS this later

    # note: "complex" means "protein + solvent"
    binding_model = RBFEModel(
        client=client,
        ff=forcefield,
        complex_system=complex_system,
        complex_coords=complex_coords,
        complex_box=complex_box,
        complex_schedule=construct_lambda_schedule(configuration.num_complex_windows),
        solvent_system=solvent_system,
        solvent_coords=solvent_coords,
        solvent_box=solvent_box,
        solvent_schedule=construct_lambda_schedule(configuration.num_solvent_windows),
        equil_steps=configuration.num_equil_steps,
        prod_steps=configuration.num_prod_steps,
    )

    # TODO: use binding_model.predict rather than binding_model.loss
    binding_estimate_and_grad_fxn = jax.value_and_grad(binding_model.loss, argnums=0)

    ordered_params = forcefield.get_ordered_params()
    ordered_handles = forcefield.get_ordered_handles()

    # in each optimizer step, look at one transformation from relative_transformations
    for step, rfe_ind in enumerate(step_inds):
        rfe = relative_transformations[rfe_ind]

        # compute a step, measuring total wall-time
        t0 = time()

        loss, loss_grads = binding_estimate_and_grad_fxn(ordered_params, rfe.mol_a, rfe.mol_b, rfe.core, rfe.label)
        # TODO: perhaps update this to accept an rfe argument, instead of all of rfe's attributes as arguments

        # TODO: how to get intermediate results from the computational pipeline encapsulated in binding_model.loss ?
        #   e.g. stage_results, and further diagnostic information
        #   * x trajectories,
        #   * d U / d parameters trajectories,
        #   * matrix of U(x; lambda) for all x, lambda
        #   * the deltaG pred
        #   (proper way is probably something like has_aux=True https://jax.readthedocs.io/en/latest/jax.html#jax.value_and_grad)

        # update forcefield parameters in-place, hopefully to match an experimental label
        parameter_updates = _update_in_place(loss, loss_grads, ordered_handles=ordered_handles, handle_types_to_update=forces_to_refit)
        # Note: for certain kinds of method-validation tests, these labels could also be synthetic

        t1 = time()
        elapsed = t1 - t0

        print(f'completed forcefield-updating step {step} in {elapsed:.3f} s !')

        # save updated forcefield files after every gradient step
        step_params = serialize_handlers(ff_handlers)
        # TODO: consider if there's a more modular way to keep track of ff updates
        _save_forcefield(f"forcefield_checkpoint_{step}.py", ff_params=step_params)

        # also save information about this step's parameter gradient and parameter update
        for handle_type in forces_to_refit:
            fname = f'parameter update (handle_type={handle_type.__name__}, step={step}).npz'
            parameter_updates[handle_type].save(fname)
