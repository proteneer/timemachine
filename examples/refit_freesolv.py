import argparse
import copy
import pickle
from csv import DictWriter
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pymbar
from jax import jit
from jax import numpy as jnp
from jax import value_and_grad, vmap
from numpy.typing import NDArray as Array
from rdkit import Chem
from scipy.optimize import minimize

from timemachine.constants import DEFAULT_FF, DEFAULT_KT, KCAL_TO_DEFAULT_KT, KCAL_TO_KJ
from timemachine.datasets import fetch_freesolv
from timemachine.fe import absolute_hydration, topology
from timemachine.fe.reweighting import one_sided_exp
from timemachine.fe.utils import get_mol_name
from timemachine.ff import Forcefield
from timemachine.ff.handlers import nonbonded, openmm_deserializer
from timemachine.ff.handlers.serialize import serialize_handlers
from timemachine.md.builders import build_water_system
from timemachine.md.smc import Samples, effective_sample_size
from timemachine.md.states import CoordsVelBox
from timemachine.parallel.client import AbstractClient, CUDAPoolClient, SerialClient
from timemachine.parallel.utils import batch_list, get_gpu_count
from timemachine.potentials.nonbonded import (
    convert_exclusions_to_rescale_masks,
    coulomb_interaction_group_energy,
    coulomb_prefactors_on_traj,
    nonbonded_v3,
)

# This is needed for pickled mols to have their properties preserved
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

ActiveTypesArray = Array
Gradient = Array
Mols = List[Chem.rdchem.Mol]

LossAux = Dict[str, Any]  # {property: value}
AuxDict = Dict[str, LossAux]
LossResult = Tuple[Tuple[float, AuxDict], Gradient]  # ((loss, aux_dict), grads)


@dataclass
class EndpointSamples:
    samples_0: Samples
    samples_1: Samples


@dataclass
class Prefactor:
    prefactor: Array  # Array of prefactors to compute the electrostatic potentials
    charge_rescale_mask: Array  # rescale mask for the ligand
    ligand_charges: Array  # ligand charges used to generate the samples
    ff: Forcefield  # ff used to generate the samples
    samples: EndpointSamples
    ref_delta_f: float


PrefactorDict = Dict[str, Prefactor]


@dataclass
class LossArgs:
    ff: Forcefield
    mols: Mols
    initial_q_params: Array
    active_q_types: Array
    prefactor_dict: PrefactorDict
    n_gpus: int
    pred_dg_dict: Dict[str, float]


def parse_options():
    parser = argparse.ArgumentParser(
        description="Refit the BCC parameters to minimize error in predicted hydration free energies on Free Solv. "
        "The refit forcefield is written to fit_ffld_all_final.py. "
        "To refit a ff: python refit_freesolv.py [--ff ff_used_to_generate_samples.py] --result_path smc_results. "
        "To analyze a refit ff on a new set of molecules: python refit_freesolv.py [--ff ff_used_to_generate_samples.py] --ff_refit fit_ffld_all_final.py --result_path new_smc_results --loss_only. "
    )
    parser.add_argument(
        "--result_path", type=str, help="path with smc results, generated using 'run_smc_on_freesolv.py'", default="."
    )
    parser.add_argument("--prefactor_path", type=str, help="path to store prefactors if not empty.")
    parser.add_argument("--n_gpus", type=int, help="number of gpus", default=get_gpu_count())
    parser.add_argument(
        "--ff", type=str, help="path to forcefield file or use default SMIRNOFF ff if not set", default=DEFAULT_FF
    )
    parser.add_argument(
        "--ff_refit",
        type=str,
        help="path to read in the refit forcefield, useful to test the refit ff on a new set of molecules along with --loss_only",
        default=None,
    )
    parser.add_argument("--n_mols", type=int, help="how many freesolv molecules to run on")
    parser.add_argument(
        "--exclude_mols", type=str, help="exclude the given molecules from the run", nargs="+", default=[]
    )
    parser.add_argument("--seed", type=int, help="random seed", default=2022)
    parser.add_argument(
        "--loss_only",
        help="only compute the loss (possibly resampling if needed) but do not optimize",
        action="store_true",
    )
    cmd_args = parser.parse_args()

    return cmd_args


def get_client(num_workers: int) -> AbstractClient:
    # Return the client for the given number of workers
    if num_workers == 1:
        client = SerialClient()
    else:
        client = CUDAPoolClient(max_workers=num_workers)

    client.verify()
    return client


def get_exp_dg(mol: Chem.rdchem.Mol) -> float:
    # Return the exp dg for the molecule in kJ/mol
    return mol.GetPropsAsDict()["dG"] * KCAL_TO_KJ


def compute_ligand_charges(q_params: Array, mol: Chem.rdchem.Mol, ff: Forcefield) -> List[float]:
    # Return the ligand charges corresponding to the input q_params and mol.
    return ff.q_handle.static_parameterize(q_params, ff.q_handle.smirks, mol)


def get_active_types(mol: Chem.rdchem.Mol, ff: Forcefield) -> ActiveTypesArray:
    # Return active BCC types for a single molecule.
    _, bond_types = nonbonded.compute_or_load_bond_smirks_matches(mol, ff.q_handle.smirks)
    return np.array(sorted(set(bond_types)))


def get_all_active_types(mols: Mols, ff: Forcefield) -> ActiveTypesArray:
    # Return active BCC types given a list of molecules.
    active_types = set()
    for mol in mols:
        active_types.update(set(get_active_types(mol, ff)))
    return np.array(sorted(active_types))


def get_compress_expand_fxns(params: Array, active_types: ActiveTypesArray) -> Tuple[Callable, Callable]:
    """
    Parameters
    ----------
    params:
        Initial full parameter array from ff.q_handle.params
    active_types:
        Array listing the active BCC parameters. See `get_all_active_types`.

    Returns
    -------
    compress(params: Array) -> Array
        compresses the full charge parameters, returning the active subset
    expand(active_params: Array) -> Array
        expands the compressed parameters to the full charge parameters
    """

    def compress(params: Array) -> Array:
        return params[active_types]

    def expand(active_params: Array) -> Array:
        full_params = jnp.array(params)
        return full_params.at[active_types].set(active_params)

    return compress, expand


def mols_to_resample(mols: Mols, aux_dict: AuxDict, cutoff=50):
    """
    Parameters
    ----------
    aux_dict:
        Maps mol name to LossAux which has an ess attribute.
        This is the effective sample size at each end state.
    cutoff: float
        Resample if either value is below this cutoff.

    Returns
    -------
    mols:
        list of molecules that should be resampled.
    """
    resample_mols = []
    for mol in mols:
        ess0, ess1 = aux_dict[get_mol_name(mol)]["ess"]
        should_resample = ess0 < cutoff or ess1 < cutoff
        if should_resample:
            resample_mols.append(mol)
    return resample_mols


def load_smc_results(path: Path, keep_mols=None) -> Tuple[Mols, List[EndpointSamples], List[float]]:
    """
    Load SMC free solv results generated by `run_smc_on_freesolv`.

    Parameters
    ----------
    path:
        Path containing the ouptut of `run_smc_on_freesolv`.
    keep_mols:
        List of molecule names to keep. Default of None means to keep all molecules.

    Returns
    -------
    mols:
        List of mols loaded.
    samples:
        Corresponding list of samples.
    ref_delta_fs:
        Corresponding list of reference delta fs.
    """
    mols = []
    samples = []
    ref_delta_fs = []

    for pkl_path in path.glob("summary_smc_result_*.pkl"):
        smc_result = pickle.load(pkl_path.open("rb"))
        if keep_mols and get_mol_name(smc_result["mol"]) not in keep_mols:
            continue
        mols.append(smc_result["mol"])

        # final then initial because the lambas were swapped in the initial smc run
        samples.append(EndpointSamples(smc_result["final_samples_refined"], smc_result["initial_samples_refined"]))
        ref_delta_fs.append(one_sided_exp(-smc_result["final_log_weights"]))

    return mols, samples, ref_delta_fs


def _loss_serial(x: Array, loss_args: LossArgs) -> Tuple[float, AuxDict]:
    """
    Compute the loss serially for the given parameters.

    Returns
    -------
    loss:
        Value for the loss
    resample_dict:
        Dict mapping mol name to the corresponding ESS value at each end point.
    """
    _, expand = get_compress_expand_fxns(loss_args.initial_q_params, loss_args.active_q_types)

    def make_prediction(q_params, ff, mol, prefactor):
        ligand_charges = compute_ligand_charges(q_params, mol, ff)
        return compute_delta_f_and_resample_value(ligand_charges, mol, prefactor)

    def make_all_predictions(active_params, expand, ff, mols, prefactor_dict):
        predictions = []
        aux_values = {}
        q_params = expand(active_params)
        for i in range(len(mols)):
            pred_dg, resample_value = make_prediction(q_params, ff, mols[i], prefactor_dict[get_mol_name(mols[i])])
            predictions.append(pred_dg)
            aux_values[get_mol_name(mols[i])] = {
                "ess": resample_value,
                "pred_dg": pred_dg,
            }

        return jnp.array(predictions), aux_values

    def data_loss(active_params, expand, ff, mols, prefactor_dict):
        preds, aux_values = make_all_predictions(active_params, expand, ff, mols, prefactor_dict)
        labels = np.array([get_exp_dg(mol) * (KCAL_TO_DEFAULT_KT / KCAL_TO_KJ) for mol in mols])
        # pred and labels in kT
        return np.sum((preds - labels) ** 2), aux_values

    return data_loss(x, expand, loss_args.ff, loss_args.mols, loss_args.prefactor_dict)


def loss_serial(x: Array, loss_args: LossArgs) -> LossResult:
    """
    Compute the loss and gradient serially for the given parameters.

    Returns
    -------
    LossResult:
        ((loss, resample_dict), grad)
    """
    loss_func = partial(_loss_serial, loss_args=loss_args)
    return value_and_grad(loss_func, has_aux=True)(x)


def loss_parallel(num_workers: int, x: Array, loss_args: LossArgs) -> LossResult:
    """
    Compute the loss in parallel.

    Parameters
    ----------
    num_workers:
        Number of workers to use for the parallel client.
    x:
        Current compressed charge parameters.
    loss_args:
        Contains auxiliary information needed to compute the predicted dG.

    Returns
    -------
    LossResult:
        ((loss, resample_dict), grad)
    """
    mols = loss_args.mols
    print("loss for", len(mols), "mols using", num_workers, "gpus")
    client = get_client(num_workers)
    futures = []
    for mol_subset in batch_list(mols, num_workers):
        prefactors_subset = {get_mol_name(mol): loss_args.prefactor_dict[get_mol_name(mol)] for mol in mol_subset}
        serial_loss_args = copy.copy(loss_args)
        serial_loss_args.mols = mol_subset
        serial_loss_args.prefactor_dict = prefactors_subset
        futures.append(client.submit(loss_serial, x, serial_loss_args))
    results = [fut.result() for fut in futures]

    # Combine results [((loss, aux), g), ...]
    v = np.sum([result[0][0] for result in results])
    aux = {}
    for result in results:
        aux.update(result[0][1])
    grad = np.sum([result[1] for result in results], axis=0)
    return (v, aux), grad


def loss(x, loss_args: LossArgs) -> Tuple[float, Gradient]:
    """
    Main loss function. This will:
        1) compute the loss using the new parameters
        2) determine if new samples should be generated
        3) generate new samples and update the prefactors
        4) recompute the loss using the new samples

    Parameters
    ----------
    x:
        Current compressed charge parameters.
    loss_args:
        Contains auxiliary information needed to compute the predicted dG.

    Returns
    -------
    loss:
        Value for the loss.
    grad:
        Gradient with resepect to x.
    """
    # compute the loss
    (v, aux), g = loss_parallel(loss_args.n_gpus, x, loss_args)

    mols = loss_args.mols
    resampled_mols = mols_to_resample(mols, aux)
    if len(resampled_mols) > 0:
        print("Resampling", len(resampled_mols), "molecules")

    # update ff params
    _, expand = get_compress_expand_fxns(loss_args.initial_q_params, loss_args.active_q_types)
    new_ff = copy.deepcopy(loss_args.ff)
    new_ff.q_handle.params = expand(x)

    if len(resampled_mols):
        # generate new samples if needed
        samples = generate_samples_for_mols_parallel(loss_args.n_gpus, resampled_mols, loss_args.prefactor_dict, new_ff)
        samples = [samples[get_mol_name(mol)] for mol in resampled_mols]

        # recompute the prefactors for the new samples
        ref_delta_fs = [None] * len(resampled_mols)  # Will be updated below
        new_prefactor_dict = compute_prefactors(expand(x), new_ff, resampled_mols, samples, ref_delta_fs)

        # update ref_delta_f to match the new samples
        update_delta_fs(resampled_mols, loss_args.prefactor_dict, new_prefactor_dict)

        # overwrite the old prefactors
        loss_args.prefactor_dict.update(new_prefactor_dict)

        # recompute the loss
        (v, aux), g = loss_parallel(loss_args.n_gpus, x, loss_args)

        # check that no molecules are marked to be resampled
        new_resampled_mols = mols_to_resample(resampled_mols, aux)
        assert (
            len(new_resampled_mols) == 0
        ), f"no molecules should need to be resampled here {[get_mol_name(mol) for mol in new_resampled_mols]}"

    # update the predictions to pass back to main
    pred_dg_dict = {mol_name: a["pred_dg"] for mol_name, a in aux.items()}
    loss_args.pred_dg_dict.update(pred_dg_dict)

    return float(v), np.array(g, dtype=np.float64)


def generate_samples_for_mols(
    mols: Mols, prefactor_dict: PrefactorDict, new_ff: Forcefield
) -> Dict[str, EndpointSamples]:
    """
    Generete new endstate samples for the mols, under a new ff.
    """
    new_samples = {}
    for mol in mols:
        prefactor = prefactor_dict[get_mol_name(mol)]
        new_samples_0 = absolute_hydration.generate_samples_for_smc_parameter_changes(
            mol,
            prefactor.ff,
            new_ff,
            prefactor.samples.samples_0,
            is_vacuum=False,
        )
        new_samples_1 = absolute_hydration.generate_samples_for_smc_parameter_changes(
            mol,
            prefactor.ff,
            new_ff,
            prefactor.samples.samples_1,
            is_vacuum=True,
        )
        new_samples[get_mol_name(mol)] = EndpointSamples(new_samples_0, new_samples_1)
    return new_samples


def generate_samples_for_mols_parallel(
    num_workers: int, mols: Mols, prefactor_dict: PrefactorDict, new_ff: Forcefield
) -> Dict[str, EndpointSamples]:
    """
    Generete new endstate samples for the mols, under a new ff.

    Parameters
    ----------
    num_workers:
        Number of workers to use for parallel generation.
    prefactor_dict:
        Prefactor containing the origianl ff and generated samples.
    new_ff:
        Generate samples using these parameters.

    Returns
    -------
    Dict of mol names to the corresponding prefactors.
    """
    print("Generate", len(mols), "new samples using", num_workers, "gpus")
    client = get_client(num_workers)
    futures = []
    for mol_subset in batch_list(mols, num_workers):
        prefactors_subset = {get_mol_name(mol): prefactor_dict[get_mol_name(mol)] for mol in mol_subset}
        futures.append(client.submit(generate_samples_for_mols, mol_subset, prefactors_subset, new_ff))
    batched_results = [fut.result() for fut in futures]

    # batched_results is a list of dict, where the keys are the mol names
    # combine into a single dict
    merged_results = {}
    for batched_result in batched_results:
        merged_results.update(batched_result)
    return merged_results


def ligand_only_traj(xvbs: List[CoordsVelBox], num_lig_atoms: int) -> List[CoordsVelBox]:
    """
    Return the trajectory with only ligand atoms. This assumes the ligand is
    the last molecule.
    """
    return [CoordsVelBox(xvb.coords[-num_lig_atoms:], xvb.velocities[-num_lig_atoms:], xvb.box) for xvb in xvbs]


def get_water_charges() -> List[float]:
    # Return the O, H, H atomic partial charges for water.
    water_system, water_coords, water_box, water_top = build_water_system(0.5)
    water_bps, water_masses = openmm_deserializer.deserialize_system(water_system, cutoff=1.2)
    water_top = topology.HostGuestTopology(water_bps, None)
    water_charges = water_top.host_nonbonded.params[:, 0][:3]
    return water_charges


def compute_prefactors(
    q_params: Array, ff: Forcefield, mols: Mols, endpoint_samples: EndpointSamples, ref_delta_fs: List[float]
) -> PrefactorDict:
    """
    Compute the prefactors used for the quick electrostatic interaction calculation.

    Parameters
    ----------
    q_params:
    ff:
        Forcefield used to generate the samples.
    mols:
        Compute prefactors for these molecules.
    endpoint_samples:
        Samples generated at each lambda end state.
        List order corresponds to the mols.
    ref_delta_fs:
        List of reference free energies corresponding to the perturbation
        from the one endstate to the other under the given ff.
        List order corresponds to the mols.

    Returns
    -------
    prefactor_dict:
        Dict of molecule names to the corresponding prefactors.
    """
    prefactors = {}
    water_charges = list(get_water_charges())

    for mol, endpoint_sample, ref_delta_f in zip(mols, endpoint_samples, ref_delta_fs):

        traj = np.array([s.coords for s in endpoint_sample.samples_0])
        boxes = np.array([s.box for s in endpoint_sample.samples_0])

        num_lig_atoms = mol.GetNumAtoms()
        num_water_atoms = traj[0].shape[0] - num_lig_atoms
        np.testing.assert_allclose(ff.q_handle.params, q_params, err_msg=f"q_params do not match {get_mol_name(mol)}")
        ligand_charges0 = list(compute_ligand_charges(ff.q_handle.params, mol, ff))

        # generate environment charges
        env_charges = water_charges * num_water_atoms
        charges = np.array(env_charges + ligand_charges0)
        env_idx = np.array(list(range(num_water_atoms)))
        ligand_idx = np.arange(num_lig_atoms) + num_water_atoms

        # prefactor data
        data = coulomb_prefactors_on_traj(traj, boxes, charges, ligand_idx, env_idx, cutoff=1.2)

        # samples
        samples = EndpointSamples(endpoint_sample.samples_0, ligand_only_traj(endpoint_sample.samples_1, num_lig_atoms))

        # exclusions for intramolecular term
        exclusion_idxs, scale_factors = nonbonded.generate_exclusion_idxs(
            mol, scale12=topology._SCALE_12, scale13=topology._SCALE_13, scale14=topology._SCALE_14
        )
        scale_factors = np.stack([scale_factors, scale_factors], axis=1)
        charge_rescale_mask, _ = convert_exclusions_to_rescale_masks(exclusion_idxs, scale_factors, num_lig_atoms)

        # prefactor
        prefactor = Prefactor(
            data,
            charge_rescale_mask,
            np.array(ligand_charges0),
            ff,
            samples,
            ref_delta_f,
        )
        prefactors[get_mol_name(mol)] = prefactor
    return prefactors


def intramol_energy(ligand_charges: Array, traj: Samples, charge_rescale_mask: Array, beta=2.0, cutoff=1.2) -> Array:
    """
    Compute the intramolecular energy for the given molecule.

    Parameters
    ----------
    ligand_charges:
        Charges of the ligand.
    traj:
        Trajectory to compute energies over. This may contain other molecules,
        but the ligand should be the last molecule.
    charge_rescale_mask:
        Generated using `timemachine.potentials.nonbonded.convert_exclusions_to_rescale_masks`
    beta:
        Electrostatics beta parameter.
    cutoff:
        Maximum interaction distance.

    Returns
    -------
    Energy value for each sample in traj.
    """
    N = len(ligand_charges)
    ligand_traj = jnp.array([x.coords for x in ligand_only_traj(traj, N)])
    ligand_charges = jnp.expand_dims(ligand_charges, axis=0)

    # LJ terms are ignored
    params = jnp.concatenate(
        [ligand_charges, jnp.ones(ligand_charges.shape), jnp.zeros(ligand_charges.shape)], axis=0
    ).T

    lambda_plane_idxs = np.zeros(N, dtype=np.int32)
    lambda_offset_idxs = np.zeros(N, dtype=np.int32)

    def calc(coords):
        return nonbonded_v3(
            coords,
            params,
            None,
            0.0,
            charge_rescale_mask,
            jnp.zeros(charge_rescale_mask.shape),
            beta=beta,
            cutoff=cutoff,
            lambda_plane_idxs=lambda_plane_idxs,
            lambda_offset_idxs=lambda_offset_idxs,
            runtime_validate=False,
        )

    return jit(vmap(calc))(ligand_traj)


def compute_delta_f_and_resample_value(
    ligand_charges: Array, mol: Chem.rdchem.Mol, prefactor: Prefactor
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute the predicted free energy and the resample value (effective sample size).

    Parameters
    ----------
    ligand_charges:
        Array of current ligand charges.
    mol:
        Molecule to compute the values for.
    prefactor:
        Contains the samples generated using the original ligand charge parameters.

    Returns
    -------
    pred_delta_f:
        Predicted change in free energy with the adjusted ligand charges
    (ess0, ess1):
        Effective sample size for the lambda=0 and lambda=1 end states.
        If this value is too low, the pred_delta_f may be unreliable and
        new samples should be generated.
    """

    def endpoint_correction_and_ess0(ligand_charges: Array) -> Tuple[float, float]:
        def U(ligand_charges: Array) -> Array:
            u = coulomb_interaction_group_energy(ligand_charges, prefactor.prefactor)
            u += intramol_energy(
                ligand_charges,
                prefactor.samples.samples_0,
                prefactor.charge_rescale_mask,
            )
            return u / DEFAULT_KT

        new_u_0 = U(ligand_charges)
        ref_u_0 = U(prefactor.ligand_charges)
        delta_us = new_u_0 - ref_u_0
        return one_sided_exp(delta_us), effective_sample_size(-delta_us)

    def endpoint_correction_and_ess1(ligand_charges: Array) -> Tuple[float, float]:
        def U(ligand_charges: Array) -> Array:
            return (
                intramol_energy(ligand_charges, prefactor.samples.samples_1, prefactor.charge_rescale_mask) / DEFAULT_KT
            )

        new_u_1 = U(ligand_charges)
        ref_u_1 = U(prefactor.ligand_charges)
        delta_us = new_u_1 - ref_u_1
        return one_sided_exp(delta_us), effective_sample_size(-delta_us)

    endpoint_correction_0, ess0 = endpoint_correction_and_ess0(ligand_charges)
    endpoint_correction_1, ess1 = endpoint_correction_and_ess1(ligand_charges)
    return prefactor.ref_delta_f + endpoint_correction_0 - endpoint_correction_1, (ess0, ess1)


def update_delta_fs(mols: Mols, prefactors: PrefactorDict, new_prefactors: PrefactorDict):
    """
    Update the ref_delta_f attribute using the new samples in
    new_prefactors.

    Parameters
    ----------
    mols:
        List of molecules to update
    prefactors:
        Dict of mol names to the corresponding prefactors from the original samples.
    new_prefactors:
        Dict of mol names to the corresponding prefactors for the new samples.
        The ref_delta_f attribute is updated in place.
    """

    def bar_correction_0(prefactor, new_prefactor) -> float:
        def U(ligand_charges, prefactor_data, samples):
            # prefactor data should match the samples (derived from environment coords)
            u = coulomb_interaction_group_energy(ligand_charges, prefactor_data)
            u += intramol_energy(
                ligand_charges,
                samples,
                prefactor.charge_rescale_mask,
            )
            return u / DEFAULT_KT

        # orig charges orig samples
        ref_u_f = U(prefactor.ligand_charges, prefactor.prefactor, prefactor.samples.samples_0)

        # new charges orig samples
        new_u_f = U(new_prefactor.ligand_charges, prefactor.prefactor, prefactor.samples.samples_0)

        # orig charges new samples
        ref_u_r = U(prefactor.ligand_charges, new_prefactor.prefactor, new_prefactor.samples.samples_0)

        # new charges new samples
        new_u_r = U(new_prefactor.ligand_charges, new_prefactor.prefactor, new_prefactor.samples.samples_0)

        u_fwd = new_u_f - ref_u_f
        u_rev = ref_u_r - new_u_r
        return pymbar.BAR(u_fwd, u_rev, compute_uncertainty=False)

    def bar_correction_1(prefactor, new_prefactor) -> float:
        def U(ligand_charges, samples):
            return intramol_energy(ligand_charges, samples, prefactor.charge_rescale_mask) / DEFAULT_KT

        # orig charges orig samples
        ref_u_f = U(prefactor.ligand_charges, prefactor.samples.samples_1)

        # new charges orig samples
        new_u_f = U(new_prefactor.ligand_charges, prefactor.samples.samples_1)

        # orig charges new samples
        ref_u_r = U(prefactor.ligand_charges, new_prefactor.samples.samples_1)

        # new charges new samples
        new_u_r = U(new_prefactor.ligand_charges, new_prefactor.samples.samples_1)

        u_fwd = new_u_f - ref_u_f
        u_rev = ref_u_r - new_u_r
        return pymbar.BAR(u_fwd, u_rev, compute_uncertainty=False)

    for mol in mols:
        prefactor = prefactors[get_mol_name(mol)]
        new_prefactor = new_prefactors[get_mol_name(mol)]
        bc0 = bar_correction_0(prefactor, new_prefactor)
        bc1 = bar_correction_1(prefactor, new_prefactor)
        delta_f = prefactor.ref_delta_f + bc0 - bc1
        new_prefactor.ref_delta_f = delta_f


def main():
    cmd_args = parse_options()
    result_path = Path(cmd_args.result_path)
    np.random.seed(cmd_args.seed)

    all_mols = fetch_freesolv(n_mols=cmd_args.n_mols, exclude_mols=cmd_args.exclude_mols)
    keep_mol_names = [get_mol_name(m) for m in all_mols]
    print("fit on", len(keep_mol_names), "mols", keep_mol_names)

    # Compute charge prefactors for all molecules
    ff = Forcefield.load_from_file(cmd_args.ff)
    mols, samples, ref_delta_fs = load_smc_results(result_path, keep_mols=keep_mol_names)
    prefactor_dict = compute_prefactors(ff.q_handle.params, ff, mols, samples, ref_delta_fs)

    ff_refit = Forcefield.load_from_file(cmd_args.ff_refit) if cmd_args.ff_refit else ff
    active_q_types = get_all_active_types(all_mols, ff_refit)
    initial_q_params = jnp.array(ff_refit.q_handle.params)
    compress, expand = get_compress_expand_fxns(initial_q_params, active_q_types)
    active_params = compress(initial_q_params)

    opt_traj = []

    def callback(x):
        # write out current ff
        opt_traj.append(x)
        fit_q_params = expand(np.array(x))
        new_ff = Forcefield.load_from_file(cmd_args.ff)
        new_ff.q_handle.params = np.array(fit_q_params)
        ff_str = serialize_handlers(new_ff.get_ordered_handles())
        Path(f"fit_ffld_all_iter_{len(opt_traj)}.py").write_text(ff_str)

    # Used to pass the predictions back
    pred_dg_dict = {}

    def parallel_loss_func(x):
        loss_args = LossArgs(
            mols=mols,
            ff=ff,
            initial_q_params=initial_q_params,
            active_q_types=active_q_types,
            prefactor_dict=prefactor_dict,
            n_gpus=cmd_args.n_gpus,
            pred_dg_dict=pred_dg_dict,
        )
        return loss(x, loss_args)

    if cmd_args.loss_only:
        loss_value, _ = parallel_loss_func(active_params)
        loss_value = (loss_value / len(mols)) ** 0.5
        print("loss_value", loss_value)
    else:
        result = minimize(
            parallel_loss_func,
            active_params,
            method="L-BFGS-B",
            jac=True,
            callback=callback,
            options={"maxiter": 1000, "gtol": 1e-2, "ftol": 1e-2},
        )
        print("result", result)
        loss_value = (result.fun / len(mols)) ** 0.5
        print("loss_value", loss_value)

        # write out final ff
        fit_q_params = expand(result.x)
        new_ff = Forcefield.load_from_file(cmd_args.ff)
        new_ff.q_handle.params = np.array(fit_q_params)
        ff_str = serialize_handlers(new_ff.get_ordered_handles())
        Path("fit_ffld_all_final.py").write_text(ff_str)

    # write out pred csv
    with open("fit_pred_dg.csv", "w") as f:
        fields = ["mol_name", "pred_dg (kJ/mol)", "exp_dg (kJ/mol)"]
        w = DictWriter(f, fields)
        w.writeheader()
        exp_dg_dict = {get_mol_name(mol): get_exp_dg(mol) for mol in all_mols}
        for mol_name, pred_dg in pred_dg_dict.items():
            exp_dg = exp_dg_dict[mol_name]
            w.writerow({fields[0]: mol_name, fields[1]: f"{pred_dg * DEFAULT_KT:.2f}", fields[2]: f"{exp_dg:.2f}"})

    # optionally write out new samples
    if cmd_args.prefactor_path:
        prefactor_path = Path(cmd_args.prefactor_path)
        prefactor_path.mkdir(exist_ok=True)
        for mol_name, prefactor in prefactor_dict.items():
            with (prefactor_path / f"prefactor_{mol_name}.pkl").open("wb") as f:
                pickle.dump(prefactor, f)


if __name__ == "__main__":
    main()
