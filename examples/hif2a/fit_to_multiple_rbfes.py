# Fit to multiple relative binding free energy edges
import sys
from argparse import ArgumentParser
import jax
from jax import numpy as jnp
import numpy as np
import datetime
import timemachine
from training.dataset import Dataset

# forcefield handlers
from ff import Forcefield
from ff.handlers.serialize import serialize_handlers
from ff.handlers.deserialize import deserialize_handlers
from ff.handlers.nonbonded import AM1CCCHandler, LennardJonesHandler

# free energy classes
from fe.free_energy import (
    RelativeFreeEnergy,
    construct_lambda_schedule,
    RBFETransformIndex,
)
from fe.estimator import SimulationResult
from fe.model import RBFEModel
from fe.cycles import construct_mle_layer, DisconnectedEdgesError
from fe.loss import pseudo_huber_loss  # , l1_loss, flat_bottom_loss

# MD initialization
from md import builders

# parallelization across multiple GPUs
from parallel.client import CUDAPoolClient, GRPCClient
from parallel.utils import get_gpu_count

from collections import namedtuple

from pickle import load, dump

from optimize.step import truncated_step

from typing import Tuple, Dict, List, Union

from pathlib import Path
from time import time

array = Union[np.array, jnp.array]

Handler = Union[AM1CCCHandler, LennardJonesHandler]  # TODO: relax this assumption

NUM_GPUS = get_gpu_count()

# how much MD to run
# TODO: rename this to something more descriptive than "Configuration"...
#   want to distinguish later between an "RBFE configuration"
#       (which describes the computation of a single edge)
#   and a "training configuration"
#       (which describes the overall training loop)
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


# locations relative to project root
root = Path(timemachine.__file__).parent.parent


def _save_forcefield(fname, ff_params):
    with open(fname, 'w') as fh:
        fh.write(ff_params)


def _compute_step_lower_bound(loss: float, blown_up: bool, relative_improvement_bound: float = 0.8) -> float:
    """problem this addresses: on a small fraction of steps, the free energy estimate may be grossly unreliable
    away from target, typically indicating an instability was encountered.
    detect if this occurs, and don't allow a step.

    """
    if not blown_up:
        return loss * relative_improvement_bound
    else:
        return loss  # don't move!


def _results_to_arrays(results: List[SimulationResult]):
    """each result object was constructed by SimulationResult(xs=xs, du_dls=full_du_dls, du_dps=grads)

    for each field, concatenate into an array
    """

    xs = np.array([r.xs for r in results])
    du_dls = np.array([r.du_dls for r in results])
    # without dtype=object get a warning about ragged arrays (inconsistent size/type)
    du_dps = np.array([r.du_dps for r in results], dtype=object)

    return xs, du_dls, du_dps


def _blew_up(results: List[SimulationResult]) -> bool:
    """if stddev(du_dls) for any window exceeded 1000 kJ/mol, don't trust result enough to take a step
    if du_dls contains any nans, don't trust result enough to take a step"""
    du_dls = _results_to_arrays(results)[1]

    # TODO: adjust this threshold a bit, move reliability calculations into fe/estimator.py or fe/model.py
    return np.isnan(du_dls).any() or (du_dls.std(1).max() > 1000)


def loss_fxn(ff_params, batch: List[Tuple[RelativeFreeEnergy, RBFEModel]]):
    index = RBFETransformIndex()
    index.build([edge[0] for edge in batch])
    indices = []
    all_results = []
    preds = []
    for rfe, model in batch:
        indices.append(list(index.get_transform_indices(rfe)))
        pred_ddG, stage_results = model.predict(ff_params, rfe.mol_a, rfe.mol_b, rfe.core)
        all_results.extend(list(stage_results))
        preds.append(pred_ddG)
    # If the edges in the batch are within a cycle, correct the ddGs
    indices = jnp.asarray(indices)
    ind_l, ind_r = indices.T
    try:
        layer = construct_mle_layer(len(index), indices)
        corrected_dg = layer(jnp.asarray(preds))
        cycle_corrected_rbfes = corrected_dg[ind_r] - corrected_dg[ind_l]
    except DisconnectedEdgesError:
        # Provided non connected graph, use the ddGs directly
        cycle_corrected_rbfes = jnp.asarray(preds)
    labels = jnp.asarray([rfe.label for rfe, _ in batch])
    loss = pseudo_huber_loss(cycle_corrected_rbfes - labels)
    # Aggregate the pseudo huber loss using mean
    loss = jnp.mean(loss)
    return loss, (cycle_corrected_rbfes, all_results)

# TODO: move flatten into optimize.utils
def flatten(params, handles) -> Tuple[np.array, callable]:
    """Turn params dict into flat array, with an accompanying unflatten function

    TODO: note that the result is going to be in the order given by ordered_handles (filtered by presence in handle_types)
        rather than in the order they appear in forces_to_refit

    TODO: maybe leave out the reference to handle_types_being optimized altogether

    TODO: does Jax have a pytree-based flatten / unflatten utility?
    """

    theta_list = []
    _shapes = dict()
    _handle_types = []

    for param, handle in zip(params, handles):
        assert handle.params.shape == param.shape
        key = type(handle)

        if key in forces_to_refit:
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


def run_validation_edges(validation: Dataset, params, systems, epoch, inference: bool = False):
    if len(validation) <= 0:
        return
    message_prefix = "Validation"
    if inference:
        message_prefix = "Inference"
    val_loss = np.zeros(len(validation))
    for i, rfe in enumerate(validation.data):
        if getattr(rfe, "complex_path", None) is not None:
            model = systems[rfe.complex_path]
        else:
            model = systems[protein_path]
        start = time()

        loss, (preds, stage_results) = loss_fxn(params, [(rfe, model)])
        elapsed = time() - start
        print(f"{message_prefix} edge {i}: time={elapsed:.2f}s, loss={loss:.2f}")
        du_dls_dict = {stage: _results_to_arrays(results)[1] for stage, results in stage_results}
        np.savez(output_path.joinpath(f"validation_du_dls_snapshot_{epoch}_{i}.npz"), **du_dls_dict)
        val_loss[i] = loss
    np.savez(
        output_path.joinpath(f"validation_edge_losses_{epoch}.npz"),
        loss=val_loss
    )


if __name__ == "__main__":
    default_output_path = f"results_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    parser = ArgumentParser(description="Fit Forcefield parameters to hif2a")
    parser.add_argument("--num_gpus", default=None, type=int,
                        help=f"Number of GPUs to run against, defaults to {NUM_GPUS} if no hosts provided")
    parser.add_argument("--hosts", nargs="*", default=None, help="Hosts running GRPC worker to use for compute")
    parser.add_argument("--param_updates", default=1000, type=int, help="Number of updates for parameters")
    parser.add_argument("--seed", default=2021, type=int, help="Seed for shuffling ordering of transformations")
    parser.add_argument("--config", default="intermediate", choices=["intermediate", "production", "test"])
    parser.add_argument("--batch_size", default=1, type=int, help="Number of items to batch together for training")

    parser.add_argument("--path_to_ff", default=str(root.joinpath('ff/params/smirnoff_1_1_0_ccc.py')))
    parser.add_argument("--path_to_edges", default=["relative_transformations.pkl"], nargs="+",
                        help="Path to pickle file containing list of RelativeFreeEnergy objects")
    parser.add_argument("--split", action="store_true", help="Split edges into train and validation set")
    parser.add_argument("--output_path", default=default_output_path, help="Path to output directory")
    parser.add_argument("--protein_path", default=None, help="Path to protein if edges don't provide protein")
    parser.add_argument("--inference_only", action="store_true", help="Disable training, run all edges as validation edges")
    # TODO: also make configurable: forces_to_refit, optimizer params, path_to_protein, path_to_protein_ff, ...
    args = parser.parse_args()
    protein_path = None
    if args.protein_path:
        prot_path = Path(args.protein_path).expanduser()
        protein_path = prot_path.as_posix()
        if not prot_path.is_file():
            print(f"Unable to find path: {protein_path}")
            sys.exit(1)

    # xor num_gpus and hosts args
    if args.num_gpus is not None and args.hosts is not None:
        print("Unable to provide --num-gpus and --hosts together")
        sys.exit(1)

    # which force field components we'll refit
    forces_to_refit = [AM1CCCHandler, LennardJonesHandler]

    # how much computation to spend per refitting step
    configuration = None
    if args.config == "intermediate":  # goldilocks
        configuration = intermediate_configuration
    elif args.config == "test":
        configuration = testing_configuration  # a little
    elif args.config == "production":
        configuration = production_configuration  # a lot
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
    client.verify()


    # load and construct forcefield
    with open(args.path_to_ff) as f:
        ff_handlers = deserialize_handlers(f.read())

    forcefield = Forcefield(ff_handlers)

    relative_transformations: List[RelativeFreeEnergy] = []
    # load pre-defined collection of relative transformations
    for edge_path in args.path_to_edges:
        with open(edge_path, "rb") as f:
            relative_transformations.extend(load(f))

    # if older transformation lack a complex_path, rely on --protein_path to set
    protein_paths = set(x.complex_path for x in relative_transformations if hasattr(x, "complex_path"))
    if protein_path is not None:
        protein_paths.add(protein_path)
    if len(protein_paths) == 0:
        print("No proteins provided by edges or with --protein_path")
        sys.exit(1)

    # create path if it doesn't exist
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f'Storing results in {output_path}')

    dataset = Dataset(relative_transformations)
    if not args.inference_only:
        if args.split:
            # TODO: More physically meaningful split
            # 80, 20 split on transformations
            training, validation = dataset.random_split(0.8)
        else:
            validation = Dataset([])
            training = dataset
    else:
        validation = dataset
        training = Dataset([])



    with open(output_path.joinpath("training_edges.pk"), "wb") as ofs:
        dump(training.data, ofs)
    if len(validation):
        with open(output_path.joinpath("validation_edges.pk"), "wb") as ofs:
            dump(validation.data, ofs)

    # Build all of the different protein systems
    systems = {}
    for prot_path in protein_paths:
        # build the complex system
        # note: "complex" means "protein + solvent"
        complex_system, complex_coords, _, _, complex_box, _ = builders.build_protein_system(prot_path)

        # build the water system
        solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(4.0)

        systems[prot_path] = RBFEModel(
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

    # TODO: how to get intermediate results from the computational pipeline encapsulated in binding_model.loss ?
    #   e.g. stage_results, and further diagnostic information
    #   * x trajectories,
    #   * d U / d parameters trajectories,
    #   * matrix of U(x; lambda) for all x, lambda
    #   * the deltaG pred
    #   (proper way is probably something like has_aux=True https://jax.readthedocs.io/en/latest/jax.html#jax.value_and_grad)

    ordered_params = forcefield.get_ordered_params()
    ordered_handles = forcefield.get_ordered_handles()

    # compute and save the sequence of relative_transformation indices
    num_epochs = int(np.ceil(args.param_updates / len(relative_transformations)))
    np.random.seed(args.seed)

    batch_size = args.batch_size
    step_inds = []
    for epoch in range(num_epochs):
        inds = np.arange(len(training.data))
        np.random.shuffle(inds)
        batched_inds = []
        num_steps = (len(inds) + batch_size - 1) // batch_size
        for i in range(num_steps):
            offset = i * batch_size
            batched_inds.append(inds[offset:offset+batch_size])
        step_inds.append(np.asarray(batched_inds, dtype=object))

    np.save(output_path.joinpath('step_indices.npy'), np.hstack(step_inds)[:args.param_updates])

    step = 0
    # in each optimizer step, look at one transformation from relative_transformations
    for epoch in range(num_epochs):
        # Run Validation edges at start of epoch. Unlike NNs we have a reasonable starting
        # point that is worth knowing
        run_validation_edges(validation, ordered_params, systems, epoch+1, inference=args.inference_only)
        print(f"Epoch: {epoch+1}/{num_epochs}")
        for batch in step_inds[epoch]:
            batch_data = []
            for i in batch:
                rfe = training.data[i]
                if getattr(rfe, "complex_path", None):
                    model = systems[rfe.complex_path]
                else:
                    model = systems[protein_path]
                batch_data.append((rfe, model))
            # compute a batch, measuring total wall-time
            t0 = time()

            (loss, (predictions, stage_results)), loss_grads = jax.value_and_grad(loss_fxn, argnums=0, has_aux=True)(
                ordered_params,
                batch_data
            )

            results_this_step = {stage: result for stage, result in stage_results}

            print(f"at optimizer step {step}, loss={loss:.3f}")

            # check if it's probably okay to take an optimizer step on the basis of this result
            # TODO: move responsibility for returning error flags / simulation uncertainty estimates further upstream
            blown_up = False
            for stage, results in results_this_step.items():
                if _blew_up(results):
                    blown_up = True
                    print(f"step {step} blew up in {stage} stage")

            # note: unflatten_grad and unflatten_theta have identical definitions for now
            flat_loss_grad, unflatten_grad = flatten(loss_grads, ordered_handles)
            flat_theta, unflatten_theta = flatten(ordered_params, ordered_handles)

            # based on current estimate of (loss, grad, and simulation stability), return a conservative step to take in parameter space
            theta_increment = truncated_step(flat_theta, loss, flat_loss_grad,
                                             step_lower_bound=_compute_step_lower_bound(loss, blown_up))
            param_increments = unflatten_theta(theta_increment)

            # for any parameter handler types being updated, update in place
            for handle in ordered_handles:
                handle_type = type(handle)
                if handle_type in param_increments:

                    # TODO: careful -- this must be a "+=" or "-=" not an "="!
                    handle.params += param_increments[handle_type]

                    increment = param_increments[handle_type]
                    increment = increment[increment != 0]
                    min_update = 0.0
                    max_update = 0.0
                    if len(increment):
                        min_update = np.min(increment)
                        max_update = np.max(increment)
                    # TODO: replace with a function that knows what to report about each handle type
                    print(
                        f'updated {len(increment)} {handle_type.__name__} params by between {min_update:.4f} and {max_update:.4f}')

            t1 = time()
            elapsed = t1 - t0

            print(f'completed forcefield-updating step {step} in {elapsed:.3f} s !')

            # save du_dls snapshot
            path_to_du_dls = output_path.joinpath(f'du_dls_snapshot_{step}.npz')
            print(f'saving du_dl trajs to {path_to_du_dls}')
            du_dls_dict = dict()  # keywords here must be strings
            for stage, results in results_this_step.items():
                du_dls_dict[stage] = _results_to_arrays(results)[1]
            np.savez(path_to_du_dls, **du_dls_dict)

            # also save information about this step's parameter gradient and parameter update
            # results to npz
            path_to_npz = output_path.joinpath(f'theta_grad_loss_snapshot_{step}.npz')
            print(f'saving theta, grad, loss snapshot to {path_to_npz}')
            np.savez(
                path_to_npz,
                theta=np.array(flat_theta),
                grad=np.array(flat_loss_grad),
                loss=loss
            )

            # TODO: same for xs and du_dps snapshots

            # save updated forcefield .py files after every gradient step
            step_params = serialize_handlers(ff_handlers)
            # TODO: consider if there's a more modular way to keep track of ff updates
            _save_forcefield(output_path.joinpath(f"forcefield_checkpoint_{step}.py"), ff_params=step_params)
            step += 1
            if step >= args.param_updates:
                break
    if not args.inference_only:
        run_validation_edges(validation, ordered_params, systems, epoch+1)
