# Fit to multiple relative binding free energy edges
import sys
from argparse import ArgumentParser
import jax
from jax import numpy as jnp
import numpy as np

# forcefield handlers
from ff import Forcefield
from ff.handlers.serialize import serialize_handlers
from ff.handlers.deserialize import deserialize_handlers
from ff.handlers.nonbonded import AM1CCCHandler, LennardJonesHandler

# free energy classes
from fe.free_energy import RelativeFreeEnergy, construct_lambda_schedule
from fe.estimator import SimulationResult
from fe.model import RBFEModel
from fe.loss import pseudo_huber_loss#, l1_loss, flat_bottom_loss

# MD initialization
from md import builders

# parallelization across multiple GPUs
from parallel.client import CUDAPoolClient, GRPCClient
from parallel.utils import get_gpu_count

from collections import namedtuple

from pickle import load
from typing import List, Union, Tuple, Dict

from optimize.step import truncated_step

from typing import List, Union
from pathlib import Path
from time import time

array = Union[np.array, jnp.array]

Handler = Union[AM1CCCHandler, LennardJonesHandler]  # TODO: relax this assumption

NUM_GPUS = get_gpu_count()

# how much MD to run, on how many GPUs
Configuration = namedtuple(
    'Configuration',
    ['num_complex_windows', 'num_solvent_windows', 'num_equil_steps', 'num_prod_steps'])

# define a couple configurations: one for quick tests, and one for production
production_configuration = Configuration(
    num_complex_windows=60,
    num_solvent_windows=60,
    num_equil_steps=10000,
    num_prod_steps=100000,
)

intermediate_configuration = Configuration(
    num_complex_windows=60,
    num_solvent_windows=60,
    num_equil_steps=10000,
    num_prod_steps=10000,
)

testing_configuration = Configuration(
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

# locations relative to project root
root = Path(__file__).absolute().parent.parent.parent
path_to_protein = str(root.joinpath('tests/data/hif2a_nowater_min.pdb'))
path_to_ff = str(root.joinpath('ff/params/smirnoff_1_1_0_ccc.py'))

# locations relative to example folder
path_to_results = Path(__file__).absolute().parent
path_to_transformations = str(path_to_results.joinpath('relative_transformations.pkl'))

# load and construct forcefield
with open(path_to_ff) as f:
    ff_handlers = deserialize_handlers(f.read())

forcefield = Forcefield(ff_handlers)


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


def _save_forcefield(filename, ff_params):
    # TODO: update path

    with open(path_to_results.joinpath(filename), 'w') as fh:
        fh.write(ff_params)


if __name__ == "__main__":
    parser = ArgumentParser(description="Fit Forcefield parameters to hif2a")
    parser.add_argument("--num-gpus", default=None, type=int, help=f"Number of GPUs to run against, defaults to {NUM_GPUS} if no hosts provided")
    parser.add_argument("--hosts", nargs="*", default=None, help="Hosts running GRPC worker to use for compute")
    parser.add_argument("--param-updates", default=1000, type=int, help="Number of updates for parameters")
    parser.add_argument("--seed", default=2021, type=int, help="Seed for shuffling ordering of transformations")
    parser.add_argument("--config", default="intermediate", choices=["intermediate", "production", "test"])
    args = parser.parse_args()
    if args.num_gpus is not None and args.hosts is not None:
        print("Unable to provide --num-gpus and --hosts together")
        sys.exit(1)

    # which force field components we'll refit
    forces_to_refit = [AM1CCCHandler, LennardJonesHandler]

    # how much computation to spend per refitting step
    configuration = None
    if args.config == "intermediate": # goldilocks
        configuration = intermediate_configuration
    elif args.config == "test":
        configuration = testing_configuration # a little
    elif args.config == "production":
        configuration = production_configuration # a lot
    assert configuration is not None, "No configuration provided"

    if not args.hosts:
        num_gpus = args.num_gpus
        if num_gpus is None:
            num_gpus = NUM_GPUS
        # set up multi-GPU client
        client = CUDAPoolClient(max_workers=num_gpus)
    else:
        # Setup GRPC client
        client = GRPCClient(hosts=args.hosts)

    # load pre-defined collection of relative transformations
    with open(path_to_transformations, 'rb') as f:
        relative_transformations: List[RelativeFreeEnergy] = load(f)

    # update the forcefield parameters for a few steps, each step informed by a single free energy calculation

    # compute and save the sequence of relative_transformation indices
    num_epochs = int(np.ceil(args.param_updates / len(relative_transformations)))
    np.random.seed(args.seed)
    step_inds = []
    for epoch in range(num_epochs):
        inds = np.arange(len(relative_transformations))
        np.random.shuffle(inds)
        step_inds.append(inds)
    step_inds = np.hstack(step_inds)[:args.param_updates]
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

    def loss_fxn(ff_params, mol_a, mol_b, core, label_ddG, callback=None):
        pred_ddG = binding_model.predict(ff_params, mol_a, mol_b, core, callback)
        return pseudo_huber_loss(pred_ddG - label_ddG)

    # TODO: how to get intermediate results from the computational pipeline encapsulated in binding_model.loss ?
    #   e.g. stage_results, and further diagnostic information
    #   * x trajectories,
    #   * d U / d parameters trajectories,
    #   * matrix of U(x; lambda) for all x, lambda
    #   * the deltaG pred
    #   (proper way is probably something like has_aux=True https://jax.readthedocs.io/en/latest/jax.html#jax.value_and_grad)

    ordered_params = forcefield.get_ordered_params()
    ordered_handles = forcefield.get_ordered_handles()

    handle_types_being_optimized = [AM1CCCHandler, LennardJonesHandler]

    # TODO: move flatten into optimize.utils
    def flatten(params) -> Tuple[np.array, callable]:
        """Turn params dict into flat array, with an accompanying unflatten function

        TODO: note that the result is going to be in the order given by ordered_handles (filtered by presence in handle_types)
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
                params[key] = np.array(theta[i: i + num_params]).reshape(shape)
                i += num_params
            return params

        return theta, unflatten


    relative_improvement_bound = 0.8

    def _compute_step_lower_bound(loss, blown_up):
        """problem this addresses: on a small fraction of steps, the free energy estimate may be grossly unreliable
        away from target, typically indicating an instability was encountered.
        detect if this occurs, and don't allow a step.

        """
        if not blown_up:
            return loss * relative_improvement_bound
        else:
            return loss # don't move!


    def _results_to_arrays(results: List[SimulationResult]):
        """each result object was constructed by SimulationResult(xs=xs, du_dls=full_du_dls, du_dps=grads)

        for each field, concatenate into an array
        """

        xs = np.array([r.xs for r in results])
        du_dls = np.array([r.du_dls for r in results])
        du_dps = np.array([r.du_dps for r in results])

        return xs, du_dls, du_dps


    def _blew_up(results: List[SimulationResult]):
        """if stddev(du_dls) for any window exceeded 1000 kJ/mol, don't trust result enough to take a step
        if du_dls contains any nans, don't trust result enough to take a step"""
        du_dls = _results_to_arrays(results)[1]

        # TODO: adjust this threshold a bit, move reliability calculations into fe/estimator.py or fe/model.py
        return np.isnan(du_dls).any() or (du_dls.std(1).max() > 1000)


    def _stringify_key(key: Tuple[int, str]):
        return f'{key[0]}_{key[1]}'


    flat_theta_traj = []
    flat_grad_traj = []
    loss_traj = []

    results_traj = dict() # indexed by (step, stage) tuples # TODO: proper type hint


    def save_in_memory_callback(results, stage):
        # TODO: replace with saving this to disk rather than accumulating in memory
        global results_traj
        results_traj[(step, stage)] = results
        print(f'collected {stage} results!')

    # in each optimizer step, look at one transformation from relative_transformations
    for step, rfe_ind in enumerate(step_inds):
        rfe = relative_transformations[rfe_ind]

        # compute a step, measuring total wall-time
        t0 = time()

        # TODO: perhaps update this to accept an rfe argument, instead of all of rfe's attributes as arguments
        # TODO: pass callback function here to save intermediate results

        loss, loss_grads = jax.value_and_grad(loss_fxn, argnums=0)(ordered_params, rfe.mol_a, rfe.mol_b, rfe.core, rfe.label, callback=save_in_memory_callback)

        blown_up = _blew_up(results_traj[(step, 'complex')]) or _blew_up(results_traj[(step, 'solvent')])

        print(f"at optimizer step {step}, loss={loss:.3f}")

        # note: unflatten_grad and unflatten_theta have identical definitions for now
        flat_loss_grad, unflatten_grad = flatten(loss_grads)
        flat_theta, unflatten_theta = flatten(ordered_params)

        theta_increment = truncated_step(flat_theta, loss, flat_loss_grad,
                                         step_lower_bound=_compute_step_lower_bound(loss, blown_up))
        param_increments = unflatten_theta(theta_increment)

        # for any parameter handler types being updated, update in place
        for handle in ordered_handles:
            handle_type = type(handle)
            if handle_type in param_increments:
                print(f'updating {handle_type.__name__}')

                # TODO: careful -- this must be a "+=" or "-=" not an "="!
                handle.params += param_increments[handle_type]

                increment = param_increments[handle_type]
                update_mask = increment != 0

                # TODO: replace with a function that knows what to report about each handle type
                print(f'updated {int(np.sum(update_mask))} params by between {np.min(increment[update_mask]):.4f} and {np.max(increment[update_mask])}')

        # Note: for certain kinds of method-validation tests, these labels could also be synthetic
        t1 = time()
        elapsed = t1 - t0

        print(f'completed forcefield-updating step {step} in {elapsed:.3f} s !')

        # also save information about this step's parameter gradient and parameter update
        # checkpoint results to npz (overwrite
        flat_theta_traj.append(np.array(flat_theta))
        flat_grad_traj.append(flat_loss_grad)
        loss_traj.append(loss)

        # TODO: update path
        path_to_du_dls = 'du_dls_checkpoint.npz'
        print(f'saving du_dl trajs to {path_to_du_dls}')
        du_dls_dict = dict() # keywords here must be strings
        for key in results_traj:
            du_dls_dict[_stringify_key(key)] = _results_to_arrays(results_traj[key])[1] # TODO: make this less gross
        np.savez(path_to_du_dls, **du_dls_dict)

        # TODO: same for xs and du_dps checkpoints
        # TODO: don't do so much re-writing: can we incrementally write into an existing .npz archive?

        # TODO: update path
        path_to_npz = 'results_checkpoint.npz'
        print(f'saving theta, grad, loss trajs to {path_to_npz}')
        np.savez(
            path_to_npz,
            theta_traj=np.array(flat_theta_traj),
            grad_traj=np.array(flat_grad_traj),
            loss_traj=np.array(loss_traj)
        )

        # save updated forcefield .py files after every gradient step
        step_params = serialize_handlers(ff_handlers)
        # TODO: consider if there's a more modular way to keep track of ff updates
        _save_forcefield(f"forcefield_checkpoint_{step}.py", ff_params=step_params)
