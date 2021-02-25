"""
1. Solvates a protein, minimizes w.r.t guest_A, equilibrates & spins off switching jobs
   (deleting guest_A while inserting guest_B), calculates work
2. Does the same thing in solvent instead of protein
3. Repeats 1 & 2 for the opposite direction (guest_B --> guest_A)
"""
import os
import time
import numpy as np

from md import builders, minimizer
from fe import free_energy, topology
from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers
from timemachine.lib import custom_ops, LangevinIntegrator

from testsystems.relative import hif2a_ligand_pair

from docking import report

MAX_LAMBDA = 1.0
MIN_LAMBDA = 0.0
TRANSITION_STEPS = 501
NUM_SWITCHES = 10


def do_relative_docking(
    host_pdbfile,
    mol_a,
    mol_b,
    core,
):
    """Runs non-equilibrium switching jobs:
    1. Solvates a protein, minimizes w.r.t guest_A, equilibrates & spins off switching jobs
       (deleting guest_A while inserting guest_B) every 1000th step, calculates work.
    2. Does the same thing in solvent instead of protein
    3. Repeats 1 & 2 for the opposite direction (guest_B --> guest_A)
    Does 10 switching jobs per leg.

    Parameters
    ----------

    host_pdbfile (str): path to host pdb file
    mol_a (rdkit mol): the starting ligand to swap from
    mol_b (rdkit mol): the ending ligand to swap to
    core (np.array[[int, int], [int, int], ...]): the common core atoms between mol_a and mol_b

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
    """

    print(
        f"""
    HOST_PDBFILE = {host_pdbfile}
    MAX_LAMBDA = {MAX_LAMBDA}
    MIN_LAMBDA = {MIN_LAMBDA}
    TRANSITION_STEPS = {TRANSITION_STEPS}
    NUM_SWITCHES = {NUM_SWITCHES}
    """
    )

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

    # sometimes water boxes are sad. Should be minimized first; this is a workaround
    host_box += np.eye(3) * 0.1

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

    # sometimes water boxes are sad. should be minimized first; this is a workaround
    water_box += np.eye(3) * 0.1
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
        single_topology = topology.SingleTopology(mol_a, mol_b, core, ff)
        rfe = free_energy.RelativeFreeEnergy(single_topology)
        ups, sys_params, combined_masses, combined_coords = rfe.prepare_host_edge(
            ff.get_ordered_params(), system, min_coords
        )
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
    for b in range(NUM_SWITCHES):
        equil2_lambda_schedule = np.ones(steps_per_batch) * MIN_LAMBDA
        ctxt.multiple_steps(equil2_lambda_schedule, 0)
        lamb = equil2_lambda_schedule[-1]
        step = len(equil2_lambda_schedule) - 1
        report.report_step(
            ctxt,
            (b+1) * step,
            lamb,
            host_box,
            combined_bps,
            u_impls,
            guest_name,
            NUM_SWITCHES * steps_per_batch,
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
        )
        works.append(work)

    return works


def do_switch(
    x0, v0, combined_bps, combined_masses, box, guest_name, leg_type, u_impls
):
    seed = 2021
    intg = LangevinIntegrator(300.0, 1.5e-3, 1.0, combined_masses, seed).impl()

    ctxt = custom_ops.Context(x0, v0, box, intg, u_impls)

    switching_lambda_schedule = np.linspace(
        MIN_LAMBDA, MAX_LAMBDA, TRANSITION_STEPS
    )

    subsample_freq = 1
    full_du_dls = ctxt.multiple_steps(switching_lambda_schedule, subsample_freq)

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
        TRANSITION_STEPS,
        f"{leg_type.upper()}_SWITCH",
    )

    if report.too_much_force(ctxt, lamb, box, combined_bps, u_impls):
        return

    work = np.trapz(full_du_dls, switching_lambda_schedule[::subsample_freq])
    print(f"guest_name: {guest_name}\t{leg_type}_work: {work:.2f}")
    return work


def main():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-p",
        "--host_pdbfile",
        default="tests/data/hif2a_nowater_min.pdb",
        help="host to dock into",
    )
    parser.add_argument(
        "-s",
        "--guests_sdfile",
        default="tests/data/ligands_40__first-two-ligs.sdf",
        help="guests to pose",
    )
    parser.add_argument(
        "-e",
        "--edges_file",
        default="tests/data/ligands_40__first-two-ligs_edges.txt",
        help="edges file. Each line should be two ligand indices separated by a comma, e.g. '1,2'",
    )
    args = parser.parse_args()

    # TODO: use guests_sdfile & edges_file instead of this test system

    # fetch mol_a, mol_b, core, forcefield from testsystem
    mol_a, mol_b, core = (
        hif2a_ligand_pair.mol_a,
        hif2a_ligand_pair.mol_b,
        hif2a_ligand_pair.core,
    )
    core_rev = core[:, ::-1]

    # A --> B
    do_relative_docking(
        args.host_pdbfile,
        mol_a,
        mol_b,
        core,
        args.outdir,
        args.fewer_outfiles,
        args.no_outfiles,
    )

    # B --> A
    do_relative_docking(
        args.host_pdbfile,
        mol_b,
        mol_a,
        core_rev,
        args.outdir,
        args.fewer_outfiles,
        args.no_outfiles,
    )


if __name__ == "__main__":
    main()
