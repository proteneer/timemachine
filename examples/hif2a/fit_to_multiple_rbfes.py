# Fit to the multiple relative binding free energy edges

import numpy as np

import functools

# forcefield handlers
from ff import Forcefield
from ff.handlers.serialize import serialize_handlers
from ff.handlers.deserialize import deserialize_handlers
from ff.handlers import nonbonded

# free energy classes
from fe.free_energy import RelativeFreeEnergy, construct_lambda_schedule

# MD initialization
from md import builders
from md import minimizer

# parallelization across multiple GPUs
from parallel.client import AbstractClient, CUDAPoolClient

from collections import namedtuple

from pickle import load
from typing import List
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


def wrap_method(args, fxn):
    return fxn(*args)


def type_check_handlers(handlers):
    """check that handlers for charges and vdW parameters are compatible with those in forcefield"""
    for handle_type in handlers:
        if handle_type == nonbonded.AM1CCCHandler:
            # sanity check as we have other charge methods that exist
            assert handle_type == type(forcefield.q_handle)

        elif handle_type == nonbonded.LennardJonesHandler:
            # sanity check again, even though we don't have other lj methods currently
            assert handle_type == type(forcefield.lj_handle)


def _print_result(lamb, result):
    """
    TODO: include type hint here for ambiguous argument name "result"
    TODO: add units
    TODO: move message into a __str__ method for a result object?
    TODO: replace print with logger
    """
    bonded_du_dl, nonbonded_du_dl, grads_and_handles = result

    unit = "kJ/mol"

    message = f"""
    lambda {lamb:.3f}
    bonded dU/dlambda (in {unit})
        mean = {np.mean(bonded_du_dl):.3f}
        stddev = {np.std(bonded_du_dl):.3f}
    nonbonded dU/dlambda (in {unit})
        mean = {np.mean(nonbonded_du_dl):.3f}
        stddev = {np.std(nonbonded_du_dl):.3f}
    """
    print(message)


def _mean_du_dlambda(result):
    """summarize result of rfe.host_edge into mean du/dl

    TODO: refactor where this analysis step occurs
    """
    bonded_du_dl, nonbonded_du_dl, _ = result
    return np.mean(bonded_du_dl + nonbonded_du_dl)


def _update_combined_handle_and_grads(ghs, combined_handle_and_grads):
    # use gradient information from the endpoints
    for (grad_lhs, handle_type_lhs), (grad_rhs, handle_type_rhs) in zip(ghs[0], ghs[-1]):
        assert handle_type_lhs == handle_type_rhs  # ffs are forked so the return handler isn't same object as that of ff
        grad = grad_rhs - grad_lhs

        # complex - solvent
        if handle_type_lhs not in combined_handle_and_grads:
            combined_handle_and_grads[handle_type_lhs] = grad
        else:
            combined_handle_and_grads[handle_type_lhs] -= grad


class ThermodynamicIntegrationResult:
    def __init__(self, lambda_schedule, results):
        self.lambda_schedule = lambda_schedule
        self.results = results

    @staticmethod
    def _save_result(name, lamb, result, rfe, configuration):
        """
        TODO: more compact way to store the relevant information about which transformation was computed
        TODO: include type hint here for ambiguous argument name "result"
        TODO: add units
        TODO: move this into part of a .save() method for a RelativeFreeEnergy result object?
        TODO: use state index i rather than lambda in filename
        TODO: include also dU/dparams here?
        """
        # unpack result tuple
        bonded_du_dl, nonbonded_du_dl, grads_and_handles = result

        # save du/dlambda trajectories
        du_dl_filename = path_to_results.joinpath(
            f"du_dlambda trajectories ({name}, lambda={lamb:.3f})")
        print(f'saving bonded and nonbonded du/dlambda trajectories to {du_dl_filename}...')
        np.savez(du_dl_filename, bonded_du_dl=bonded_du_dl, nonbonded_du_dl=nonbonded_du_dl, rfe=rfe,
                 configuration=configuration)

    def save(self, name=''):
        for lamb, result in zip(self.lambda_schedule, self.results):
            self._save_result(name, lamb, result, rfe, configuration)


# TODO: shorten this function
def predict_dG_and_grad(rfe: RelativeFreeEnergy, conf: Configuration, client: AbstractClient):
    """
    rfe defines the free energy transformation,
    conf specifies the computational details
    client specifies a multi-node distributed computing client environment

    TODO: add an argument for force field parameters, then register this function with Jax as value_and_grad or similar...
    """

    # build the complex system
    complex_system, complex_coords, _, _, complex_box, _ = builders.build_protein_system(
        path_to_protein)
    # TODO: optimize box
    complex_box += np.eye(3) * 0.1  # BFGS this later

    # build the water system
    solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(4.0)
    # TODO: optimize box
    solvent_box += np.eye(3) * 0.1  # BFGS this later

    combined_handle_and_grads = {}
    stage_dGs = dict()
    stage_results = dict()

    # TODO: break up this giant loop body
    for stage, host_system, host_coords, host_box, num_host_windows in [
        ("complex", complex_system, complex_coords, complex_box, conf.num_complex_windows),
        ("solvent", solvent_system, solvent_coords, solvent_box, conf.num_solvent_windows)]:

        lambda_schedule = construct_lambda_schedule(num_host_windows)

        print("Minimizing the host structure to remove clashes...")
        minimized_host_coords = minimizer.minimize_host_4d(rfe.mol_a, host_system, host_coords, rfe.ff, host_box)

        # one GPU job per lambda window
        print('submitting tasks to client!')
        do_work = functools.partial(wrap_method, fxn=rfe.host_edge)
        futures = []
        for lambda_idx, lamb in enumerate(lambda_schedule):
            arg = (lamb, host_system, minimized_host_coords, host_box, conf.num_equil_steps, conf.num_prod_steps)
            futures.append(client.submit(do_work, arg))

        results = []
        for fut in futures:
            results.append(fut.result())

        stage_results[stage] = ThermodynamicIntegrationResult(lambda_schedule, results)

        print(f'{stage} results by lambda window:')
        for lamb, result in zip(lambda_schedule, results):
            _print_result(lamb, result)

        # estimate dG for this stage
        pred_dG = np.trapz([_mean_du_dlambda(x) for x in results], lambda_schedule)
        # TODO: refactor this to be a call to something in a generic free_energy.analysis module
        #   to allow comparison / swapping between TI estimator, MBAR estimator, and others
        stage_dGs[stage] = pred_dG

        # collect derivative handlers
        ghs = []
        for lamb, result in zip(lambda_schedule, results):
            bonded_du_dl, nonbonded_du_dl, grads_and_handles = result
            ghs.append(grads_and_handles)

        # update forcefield gradient handlers in-place
        _update_combined_handle_and_grads(ghs, combined_handle_and_grads)

    pred_rbfe = stage_dGs['complex'] - stage_dGs['solvent']
    return pred_rbfe, combined_handle_and_grads, stage_results


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


def _update_in_place(pred, grads, label,
                     handle_types_to_update=[nonbonded.AM1CCCHandler, nonbonded.LennardJonesHandler]):
    """
    Notes
    -----
    * Currently hard-codes l1 loss -- may want to later break this up

    TODO: check if it's too expensive to do out-of-place updates with a copy
    """

    # TODO: refactor loss definition out of this function
    loss = np.abs(pred - label)
    unit = "kJ/mol"

    message = f"""
    pred:   {pred:.3f} {unit}
    target: {label:.3f} {unit} 
    loss:   {loss:.3f} {unit}
    """
    print(message)
    # TODO: save these also to log file

    dl_dpred = np.sign(pred - label)

    # compute updates
    parameter_updates = dict()

    for handle_type in handle_types_to_update:
        # TODO: move this out of update function to help with saving and logging, and eventually move the chain-rule
        #  step out of manually written function
        dl_dp = dl_dpred * grads[handle_type]  # chain rule
        # TODO: save dl_dp, update, and params to disk

        update = _clipped_update(dl_dp, step_sizes[handle_type], gradient_clip_thresholds[handle_type])
        print(f'incrementing the {handle_type.__name__} parameters by {update}')

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
            raise (RuntimeError("Attempting to update an unsupported ff handle type"))

        parameter_updates[handle_type] = ParameterUpdate(before, after, dl_dp, update)

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
    configuration = testing_configuration  # a little
    # configuration = production_configuration  # a lot

    # how many parameter update steps
    num_parameter_updates = 100
    # TODO: make this configurable also

    # set up multi-GPU client
    client = CUDAPoolClient(max_workers=configuration.num_gpus)

    # load pre-defined collection of relative transformations
    with open(path_to_transformations, 'rb') as f:
        relative_transformations: List[RelativeFreeEnergy] = load(f)

    # update the forcefield parameters for a few steps, each step informed by a single free energy calculation

    # compute and save the sequence of relative_transformation indices
    num_epochs = np.ceil(num_parameter_updates / len(relative_transformations))
    step_inds = []
    for epoch in range(num_epochs):
        inds = np.arange(len(relative_transformations))
        np.random.shuffle(inds)
        step_inds.append(inds)
    step_inds = np.hstack(step_inds)[:num_parameter_updates]
    np.save(path_to_results.joinpath('step_indices.npy'), step_inds)

    # in each optimizer step, look at one transformation from relative_transformations
    for step in step_inds:
        rfe = relative_transformations[step]

        # compute a step, measuring total wall-time
        t0 = time()
        # estimate predicted dG, and gradient of predicted dG w.r.t. params
        pred, grads, stage_results = predict_dG_and_grad(rfe, configuration, client)

        # update forcefield parameters in-place, hopefully to match an experimental label
        parameter_updates = _update_in_place(pred, grads, label=rfe.label, handle_types_to_update=forces_to_refit)
        # Note: for certain kinds of method-validation tests, these labels could also be synthetic

        t1 = time()
        elapsed = t1 - t0

        print(f'completed forcefield-updating step {step} in {elapsed:.3f} s !')

        # also save some additional diagnostic information to disk (du/dlambda trajectory)
        print('saving results')
        for stage in stage_results:
            stage_results[stage].save(name=f'step={step}, stage={stage}')

        # TODO: save some further diagnostic information
        #   * x trajectories,
        #   * d U / d parameters trajectories,
        #   * matrix of U(x; lambda) for all x, lambda

        # save updated forcefield files after every gradient step
        step_params = serialize_handlers(ff_handlers)
        # TODO: consider if there's a more modular way to keep track of ff updates
        _save_forcefield(f"forcefield_checkpoint_{step}.py", ff_params=step_params)

        # also save information about this step's parameter gradient and parameter update
        for handle_type in forces_to_refit:
            fname = f'parameter update (handle_type={handle_type.__name__}, step={step}).npz'
            parameter_updates[handle_type].save(fname)
