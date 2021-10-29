"""
1. Solvates a protein, minimizes w.r.t guest_A, equilibrates & spins off switching jobs
   (deleting guest_A while inserting guest_B), calculates work
2. Does the same thing in solvent instead of protein
3. Repeats 1 & 2 for the opposite direction (guest_B --> guest_A)
"""
import os
import time
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdFMCS

from md import builders, minimizer
from fe import free_energy, topology
from fe.atom_mapping import (
    get_core_by_geometry,
    get_core_by_mcs,
    get_core_by_smarts,
    mcs_map,
)
from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers
from timemachine.lib import custom_ops, LangevinIntegrator

from docking import report


MAX_LAMBDA = 1.0
MIN_LAMBDA = 0.0


def do_relative_docking(host_pdbfile, mol_a, mol_b, core, num_switches, transition_steps):
    """Runs non-equilibrium switching jobs:
    1. Solvates a protein, minimizes w.r.t guest_A, equilibrates & spins off switching jobs
       (deleting guest_A while inserting guest_B) every 1000th step, calculates work.
    2. Does the same thing in solvent instead of protein
    Does num_switches switching jobs per leg.

    Parameters
    ----------

    host_pdbfile (str): path to host pdb file
    mol_a (rdkit mol): the starting ligand to swap from
    mol_b (rdkit mol): the ending ligand to swap to
    core (np.array[[int, int], [int, int], ...]): the common core atoms between mol_a and mol_b
    num_switches (int): number of switching trajectories to run per compound pair per leg
    transition_stpes (int): length of each switching trajectory

    Returns
    -------

    {str: float}: map of leg label to work values of switching mol_a to mol_b in that leg,
                  {'protein': [work values], 'solvent': [work_values]}

    Output
    ------

    stdout noting the step number, lambda value, and energy at various steps
    stdout noting the work of transition, if applicable
    stdout noting how long it took to run

    Note
    ----
    The work will not be calculated if any norm of force per atom exceeds 20000 kJ/(mol*nm)
       [MAX_NORM_FORCE defined in docking/report.py]
    The simulations won't run if the atom maps are not factorizable
    """

    # Prepare host
    # TODO: handle extra (non-transitioning) guests?
    print("Solvating host...")
    (
        solvated_host_system,
        solvated_host_coords,
        _,
        _,
        host_box,
        solvated_topology,
    ) = builders.build_protein_system(host_pdbfile)

    # Prepare water box
    print("Generating water box...")
    # TODO: water box probably doesn't need to be this big
    box_lengths = host_box[np.diag_indices(3)]
    water_box_width = min(box_lengths)
    (
        water_system,
        water_coords,
        water_box,
        water_topology,
    ) = builders.build_water_system(water_box_width)

    # it's okay if the water box here and the solvated protein box don't align -- they have PBCs

    # Run the procedure
    start_time = time.time()
    guest_name_a = mol_a.GetProp("_Name")
    guest_name_b = mol_b.GetProp("_Name")
    combined_name = guest_name_a + "-->" + guest_name_b

    guest_conformer_a = mol_a.GetConformer(0)
    orig_guest_coords_a = np.array(guest_conformer_a.GetPositions(), dtype=np.float64)
    orig_guest_coords_a = orig_guest_coords_a / 10  # convert to md_units

    guest_ff_handlers = deserialize_handlers(
        open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "ff/params/smirnoff_1_1_0_ccc.py",
            )
        ).read()
    )
    ff = Forcefield(guest_ff_handlers)

    all_works = {}
    for system, coords, box, label in zip(
        [solvated_host_system, water_system],
        [solvated_host_coords, water_coords],
        [host_box, water_box],
        ["protein", "solvent"],
    ):
        # minimize w.r.t. both mol_a and mol_b?
        min_coords = minimizer.minimize_host_4d([mol_a], system, coords, ff, box)

        try:
            single_topology = topology.SingleTopology(mol_a, mol_b, core, ff)
            rfe = free_energy.RelativeFreeEnergy(single_topology)
            ups, sys_params, combined_masses, combined_coords = rfe.prepare_host_edge(
                ff.get_ordered_params(), system, min_coords
            )
        except topology.AtomMappingError as e:
            print(f"NON-FACTORIZABLE PAIR: {combined_name}")
            print(e)
            return {}

        combined_bps = []
        for up, sp in zip(ups, sys_params):
            combined_bps.append(up.bind(sp))
        all_works[label] = run_leg(
            combined_coords,
            combined_bps,
            combined_masses,
            box,
            combined_name,
            label,
            num_switches,
            transition_steps,
        )
        end_time = time.time()
        print(
            f"{combined_name} {label} leg time:",
            "%.2f" % (end_time - start_time),
            "seconds",
        )
    return all_works


def run_leg(
    combined_coords,
    combined_bps,
    combined_masses,
    host_box,
    guest_name,
    leg_type,
    num_switches,
    transition_steps,
):
    x0 = combined_coords
    v0 = np.zeros_like(x0)
    print(
        f"{leg_type.upper()}_SYSTEM",
        f"guest_name: {guest_name}",
        f"num_atoms: {len(x0)}",
    )

    seed = 2021
    intg = LangevinIntegrator(300.0, 1.5e-3, 1.0, combined_masses, seed).impl()

    u_impls = []
    for bp in combined_bps:
        bp_impl = bp.bound_impl(precision=np.float32)
        u_impls.append(bp_impl)

    ctxt = custom_ops.Context(x0, v0, host_box, intg, u_impls)

    # TODO: pre-equilibrate?

    # equilibrate & shoot off switching jobs
    steps_per_batch = 1001

    works = []
    for b in range(num_switches):
        equil2_lambda_schedule = np.ones(steps_per_batch) * MIN_LAMBDA
        ctxt.multiple_steps(equil2_lambda_schedule, 0)
        lamb = equil2_lambda_schedule[-1]
        step = len(equil2_lambda_schedule) - 1
        report.report_step(
            ctxt,
            (b + 1) * step,
            lamb,
            host_box,
            combined_bps,
            u_impls,
            guest_name,
            num_switches * steps_per_batch,
            f"{leg_type.upper()}_EQUILIBRATION_2",
        )

        if report.too_much_force(ctxt, MIN_LAMBDA, host_box, combined_bps, u_impls):
            return

        work = do_switch(
            ctxt.get_x_t(),
            ctxt.get_v_t(),
            combined_bps,
            combined_masses,
            host_box,
            guest_name,
            leg_type,
            u_impls,
            transition_steps,
        )
        works.append(work)

    return works


def do_switch(
    x0,
    v0,
    combined_bps,
    combined_masses,
    box,
    guest_name,
    leg_type,
    u_impls,
    transition_steps,
):
    seed = 2021
    intg = LangevinIntegrator(300.0, 1.5e-3, 1.0, combined_masses, seed).impl()

    ctxt = custom_ops.Context(x0, v0, box, intg, u_impls)

    switching_lambda_schedule = np.linspace(MIN_LAMBDA, MAX_LAMBDA, transition_steps)

    subsample_interval = 1
    full_du_dls, _, _ = ctxt.multiple_steps(switching_lambda_schedule, subsample_interval)

    step = len(switching_lambda_schedule) - 1
    lamb = switching_lambda_schedule[-1]
    ctxt.step(lamb)
    report.report_step(
        ctxt,
        step,
        lamb,
        box,
        combined_bps,
        u_impls,
        guest_name,
        transition_steps,
        f"{leg_type.upper()}_SWITCH",
    )

    if report.too_much_force(ctxt, lamb, box, combined_bps, u_impls):
        return

    work = np.trapz(full_du_dls, switching_lambda_schedule[::subsample_interval])
    print(f"guest_name: {guest_name}\t{leg_type}_work: {work:.2f}")
    return work


def get_core_by_permissive_mcs(mol_a, mol_b):
    # copied from timemachine/examples/hif2a/generate_star_map.py
    mcs_result = rdFMCS.FindMCS(
        [mol_a, mol_b],
        timeout=30,
        threshold=1.0,
        atomCompare=rdFMCS.AtomCompare.CompareAny,
        completeRingsOnly=True,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        matchValences=True,
    )
    query = mcs_result.queryMol

    # fails distance assertions
    # return get_core_by_mcs(mol_a, mol_b, query)

    inds_a = mol_a.GetSubstructMatches(query)[0]
    inds_b = mol_b.GetSubstructMatches(query)[0]
    core = np.array([inds_a, inds_b]).T

    return core


def _get_match(mol, query):
    # copied from timemachine/examples/hif2a/generate_star_map.py
    matches = mol.GetSubstructMatches(query)
    return matches[0]


def _get_core_by_smarts_wo_checking_uniqueness(mol_a, mol_b, core_smarts):
    # copied from timemachine/examples/hif2a/generate_star_map.py
    """no atom mapping errors with this one, but the core size is smaller"""
    query = Chem.MolFromSmarts(core_smarts)

    return np.array([_get_match(mol_a, query), _get_match(mol_b, query)]).T


def get_core(strategy, mol_a, mol_b, smarts=None):
    # adapted from timemachine/examples/hif2a/generate_star_map.py
    core_strategies = {
        "custom_mcs": lambda a, b, s: get_core_by_mcs(a, b, mcs_map(a, b).queryMol),
        "any_mcs": lambda a, b, s: get_core_by_permissive_mcs(a, b),
        "geometry": lambda a, b, s: get_core_by_geometry(a, b, threshold=0.5),
        "smarts": lambda a, b, s: get_core_by_smarts(a, b, core_smarts=s),
        "smarts_wo_uniqueness": lambda a, b, s: _get_core_by_smarts_wo_checking_uniqueness(a, b, core_smarts=s),
    }
    f = core_strategies[strategy]
    core = f(mol_a, mol_b, smarts)
    return core


def main():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-p",
        "--host_pdbfile",
        default="tests/data/hif2a_nowater_min.pdb",
        help="host to dock into",
    )
    parser.add_argument(
        "-s",
        "--guests_sdfile",
        default="tests/data/ligands_40.sdf",
        help="guests to pose",
    )
    parser.add_argument(
        "-e",
        "--edges_file",
        default="tests/data/ligands_40__edges.txt",
        help="edges file. Each line should be two ligand indices separated by a comma, e.g. '1,2'",
    )
    parser.add_argument(
        "-c",
        "--core_strategy",
        default="geometry",
        help="algorithm to use to find common core. can be 'geometry', 'custom_mcs', 'any_mcs', 'smarts', or 'smarts_wo_uniqueness'",
    )
    parser.add_argument(
        "--smarts",
        default=None,
        help="core smarts pattern, to be used with core_strategy 'smarts' or 'smarts_wo_uniqueness'",
    )
    parser.add_argument(
        "--num_switches",
        default=15,
        type=int,
        help="number of A-->B transitions to do for each compound pair",
    )
    parser.add_argument(
        "--transition_steps",
        default=1001,
        type=int,
        help="how many steps to transition from A-->B over",
    )

    args = parser.parse_args()

    print(
        f"""
    MAX_LAMBDA = {MAX_LAMBDA}
    MIN_LAMBDA = {MIN_LAMBDA}
    """
    )
    print(args)

    suppl = Chem.SDMolSupplier(args.guests_sdfile, removeHs=False)
    mols = [x for x in suppl]

    with open(args.edges_file, "r") as rfile:
        lines = rfile.readlines()
    for line in lines:
        i, j = [int(x) for x in line.strip().split(",")]
        mol_a = mols[i]
        mol_b = mols[j]

        core = get_core(args.core_strategy, mol_a, mol_b, args.smarts)
        print(core)

        # A --> B
        all_works = do_relative_docking(
            args.host_pdbfile,
            mol_a,
            mol_b,
            core,
            args.num_switches,
            args.transition_steps,
        )
        print(all_works)


if __name__ == "__main__":
    main()
