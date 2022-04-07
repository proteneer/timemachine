# Fit to multiple relative binding free energy edges
import datetime
import sys
from argparse import ArgumentParser
from collections import defaultdict, namedtuple
from pathlib import Path
from pickle import dump, load
from time import time
from typing import Any, Dict, List, Tuple, Union

import jax
import numpy as np
from jax import numpy as jnp

import timemachine
from timemachine.fe.estimator import SimulationResult
from timemachine.fe.free_energy import RBFETransformIndex, RelativeFreeEnergy

# free energy classes
from timemachine.fe.lambda_schedule import construct_lambda_schedule
from timemachine.fe.loss import pseudo_huber_loss  # , l1_loss, flat_bottom_loss
from timemachine.fe.model import RBFEModel

# forcefield handlers
from timemachine.ff import Forcefield
from timemachine.ff.handlers.nonbonded import AM1CCCHandler, LennardJonesHandler
from timemachine.ff.handlers.serialize import serialize_handlers

# MD initialization
from timemachine.md import builders
from timemachine.optimize.step import truncated_step
from timemachine.optimize.utils import flatten_and_unflatten

# parallelization across multiple GPUs
from timemachine.parallel.client import CUDAPoolClient, GRPCClient
from timemachine.parallel.utils import get_gpu_count
from timemachine.training.dataset import Dataset

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
    "Configuration", ["num_complex_windows", "num_solvent_windows", "num_equil_steps", "num_prod_steps"]
)

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
    with open(fname, "w") as fh:
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
        pred_ddG, stage_results = model.predict(ff_params, rfe.mol_a, rfe.mol_b, rfe.top.core)
        all_results.extend(list(stage_results))
        preds.append(pred_ddG)
    labels = jnp.asarray([rfe.label for rfe, _ in batch])
    loss = pseudo_huber_loss(jnp.asarray(preds) - labels)
    # Aggregate the pseudo huber loss using mean
    loss = jnp.mean(loss)
    return loss, (preds, all_results)


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
    np.savez(output_path.joinpath(f"validation_edge_losses_{epoch}.npz"), loss=val_loss)


def equilibrate_edges(datasets: List[Dataset], systems: List[Dict[str, Any]], num_steps: int, cache_path: str):
    model_set = defaultdict(list)
    for dataset in datasets:
        for rfe in dataset.data:
            if getattr(rfe, "complex_path", None) is not None:
                model_set[rfe.complex_path].append(rfe)
            else:
                model_set[protein_path].append(rfe)
    for path, edges in model_set.items():
        model = systems[path]
        model.equilibrate_edges(
            [(edge.mol_a, edge.mol_b, edge.top.core) for edge in edges],
            equilibration_steps=num_steps,
            cache_path=cache_path,
        )


if __name__ == "__main__":
    default_output_path = f"results_{datetime.datetime.utcnow().isoformat(timespec='seconds').replace(':', '_')}"
    parser = ArgumentParser(description="Fit Forcefield parameters to hif2a")
    parser.add_argument(
        "--num_gpus",
        default=None,
        type=int,
        help=f"Number of GPUs to run against, defaults to {NUM_GPUS} if no hosts provided",
    )
    parser.add_argument("--hosts", nargs="*", default=None, help="Hosts running GRPC worker to use for compute")
    parser.add_argument("--param_updates", default=1000, type=int, help="Number of updates for parameters")
    parser.add_argument("--seed", default=2021, type=int, help="Seed for shuffling ordering of transformations")
    parser.add_argument("--config", default="intermediate", choices=["intermediate", "production", "test"])
    parser.add_argument("--batch_size", default=1, type=int, help="Number of items to batch together for training")

    parser.add_argument("--path_to_ff", default=str(root.joinpath("timemachine/ff/params/smirnoff_1_1_0_ccc.py")))
    parser.add_argument(
        "--path_to_edges",
        default=["relative_transformations.pkl"],
        nargs="+",
        help="Path to pickle file containing list of RelativeFreeEnergy objects",
    )
    parser.add_argument("--split", action="store_true", help="Split edges into train and validation set")
    parser.add_argument(
        "--pre_equil",
        default=None,
        help="Number of pre equilibration steps or path to cached equilibrated edges, if not provided no pre equilibration performed",
    )
    parser.add_argument("--hmr", action="store_true", help="Enable HMR")
    parser.add_argument("--output_path", default=default_output_path, help="Path to output directory")
    parser.add_argument("--protein_path", default=None, help="Path to protein if edges don't provide protein")
    parser.add_argument(
        "--inference_only", action="store_true", help="Disable training, run all edges as validation edges"
    )
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

    forcefield = Forcefield.load_from_file(args.path_to_ff)

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
    print(f"Storing results in {output_path}")

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
            pre_equilibrate=args.pre_equil is not None,
            hmr=args.hmr,
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
            batched_inds.append(inds[offset : offset + batch_size])
        step_inds.append(np.asarray(batched_inds, dtype=object))

    np.save(output_path.joinpath("step_indices.npy"), np.hstack(step_inds)[: args.param_updates])

    pre_equil = args.pre_equil
    if pre_equil is not None:
        steps = 0
        cache_path = output_path.joinpath("equilibration_cache.pkl")
        if pre_equil.isdigit():
            steps = int(pre_equil)
        elif Path(pre_equil).is_file():
            cache_path = pre_equil
        else:
            print(f"Must provide either an integer or a valid path for --pre_equil, got {pre_equil}")
            sys.exit(1)

        equilibrate_edges([training, validation], systems, steps, cache_path)

    flatten, unflatten = flatten_and_unflatten(ordered_params)

    step = 0
    # in each optimizer step, look at one transformation from relative_transformations
    for epoch in range(num_epochs):
        # Run Validation edges at start of epoch. Unlike NNs we have a reasonable starting
        # point that is worth knowing
        run_validation_edges(validation, ordered_params, systems, epoch + 1, inference=args.inference_only)
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
                ordered_params, batch_data
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

            flat_loss_grad = flatten(loss_grads)
            flat_theta = flatten(ordered_params)

            # based on current estimate of (loss, grad, and simulation stability), return a conservative step to take in parameter space
            theta_increment = truncated_step(
                flat_theta, loss, flat_loss_grad, step_lower_bound=_compute_step_lower_bound(loss, blown_up)
            )
            param_increments = unflatten(theta_increment)

            # for any parameter handler types being updated, update in place
            for handle, increment in zip(ordered_handles, param_increments):
                handle_type = type(handle)
                if handle_type in forces_to_refit:

                    # TODO: careful -- this must be a "+=" or "-=" not an "="!
                    handle.params += increment

                    nonzero_increments = increment[increment != 0]
                    min_update = 0.0
                    max_update = 0.0
                    if len(nonzero_increments):
                        min_update = np.min(nonzero_increments)
                        max_update = np.max(nonzero_increments)
                    # TODO: replace with a function that knows what to report about each handle type
                    print(
                        f"updated {len(nonzero_increments)} {handle_type.__name__} params by between {min_update:.4f} and {max_update:.4f}"
                    )

            t1 = time()
            elapsed = t1 - t0

            print(f"completed forcefield-updating step {step} in {elapsed:.3f} s !")

            # save du_dls snapshot
            path_to_du_dls = output_path.joinpath(f"du_dls_snapshot_{step}.npz")
            print(f"saving du_dl trajs to {path_to_du_dls}")
            du_dls_dict = dict()  # keywords here must be strings
            for stage, results in results_this_step.items():
                du_dls_dict[stage] = _results_to_arrays(results)[1]
            np.savez(path_to_du_dls, **du_dls_dict)

            # also save information about this step's parameter gradient and parameter update
            # results to npz
            path_to_npz = output_path.joinpath(f"theta_grad_loss_snapshot_{step}.npz")
            print(f"saving theta, grad, loss snapshot to {path_to_npz}")
            np.savez(path_to_npz, theta=np.array(flat_theta), grad=np.array(flat_loss_grad), loss=loss)

            # TODO: same for xs and du_dps snapshots

            # save updated forcefield .py files after every gradient step
            step_params = serialize_handlers(ordered_handles)
            # TODO: consider if there's a more modular way to keep track of ff updates
            _save_forcefield(output_path.joinpath(f"forcefield_checkpoint_{step}.py"), ff_params=step_params)
            step += 1
            if step >= args.param_updates:
                break
    if not args.inference_only:
        run_validation_edges(validation, ordered_params, systems, epoch + 1)
