"""
1. Solvates a host, inserts guest(s) into solvated host, equilibrates, spins off deletion jobs, calculates work
2. Creates a water box, inserts guest(s) into water box, equilibrates, spins off deletion jobs, calculates work
"""
import os
import time
import tempfile
from collections import defaultdict
import numpy as np

from rdkit import Chem

from timemachine.md import builders, minimizer
from timemachine.fe import pdb_writer, free_energy
from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers
from timemachine.lib import custom_ops, LangevinIntegrator

from docking import report

DELETION_MAX_LAMBDA = 1.0
MIN_LAMBDA = 0.0


def calculate_rigorous_work(
    host_pdbfile,
    guests_sdfile,
    outdir,
    num_deletions,
    deletion_steps,
    insertion_max_lambda=0.5,
    insertion_steps=501,
    eq1_steps=5001,
    fewer_outfiles=False,
    no_outfiles=False,
):
    """Runs non-equilibrium deletion jobs:
    1. Solvates a protein, inserts guest, equilibrates, equilibrates more & spins off
       deletion jobs every 1000th step, calculates work.
    2. Does the same thing in solvent instead of protein.
    Does num_deletions deletion jobs per leg per compound.

    Parameters
    ----------

    host_pdbfile (str): path to host pdb file
    guests_sdfile (str): path to guests sdf file
    outdir (str): path to directory to which to write output
    num_deletions (int): number of deletion trajectories to run per leg per compound
    deletion_steps (int): length of each deletion trajectory
    insertion_max_lambda (float): how far away to insert from (0.0-1.0)
    insertion_steps (int): how long to insert over
    eq1_steps (int): how long to equilibrate after insertion and before starting the deletions
    fewer_outfiles (bool): only save the starting frame of each deletion trajectory
    no_outfiles (bool): don't keep any output files

    Returns
    -------

    {str: {str: float}}: map of compound to leg label to work values
                         {'guest_1': {'protein': [work values], 'solvent': [work_values]}, ...}

    Output
    ------

    A pdb & sdf file for each guest's final insertion step
      (outdir/<guest_name>_pd_<step>_host.pdb & outdir/<guest_name>_pd_<step>_guest.sdf)
      (unless fewer_outfiles or no_outfiles is True)
    A pdb & sdf file for each guest's final eq1 step
      (outdir/<guest_name>_pd_<step>_host.pdb & outdir/<guest_name>_pd_<step>_guest.sdf)
      (unless fewer_outfiles or no_outfiles is True)
    A pdb & sdf file for each deletion job's first step
      (outdir/<guest_name>_pd_<step>_host.pdb & outdir/<guest_name>_pd_<step>_guest.sdf)
      (unless no_outfiles is True)
    stdout corresponding to the files written noting the lambda value and energy
    stdout noting the work of deletion, if applicable
    stdout noting how long each leg took to run

    Note
    ----
    The work will not be calculated if the du_dl endpoints are not close to 0 or if any norm of
    force per atom exceeds 20000 kJ/(mol*nm) [MAX_NORM_FORCE defined in docking/report.py]
    """

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print(
        f"""
    HOST_PDBFILE = {host_pdbfile}
    GUESTS_SDFILE = {guests_sdfile}
    OUTDIR = {outdir}

    DELETION_MAX_LAMBDA = {DELETION_MAX_LAMBDA}
    MIN_LAMBDA = {MIN_LAMBDA}
    insertion_max_lambda = {insertion_max_lambda}
    insertion_steps = {insertion_steps}
    eq1_steps = {eq1_steps}
    num_deletions = {num_deletions}
    deletion_steps = {deletion_steps}
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

    _, solvated_host_pdb = tempfile.mkstemp(suffix=".pdb", text=True)
    writer = pdb_writer.PDBWriter([solvated_topology], solvated_host_pdb)
    writer.write_frame(solvated_host_coords)
    writer.close()
    solvated_host_mol = Chem.MolFromPDBFile(solvated_host_pdb, removeHs=False)
    os.remove(solvated_host_pdb)

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
    _, water_pdb = tempfile.mkstemp(suffix=".pdb", text=True)
    writer = pdb_writer.PDBWriter([water_topology], water_pdb)
    writer.write_frame(water_coords)
    writer.close()
    water_mol = Chem.MolFromPDBFile(water_pdb, removeHs=False)
    os.remove(water_pdb)

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

    # Run the procedure
    all_works = defaultdict(dict)
    print("Getting guests...")
    suppl = Chem.SDMolSupplier(guests_sdfile, removeHs=False)
    for guest_mol in suppl:
        start_time = time.time()
        guest_name = guest_mol.GetProp("_Name")
        guest_conformer = guest_mol.GetConformer(0)
        orig_guest_coords = np.array(guest_conformer.GetPositions(), dtype=np.float64)
        orig_guest_coords = orig_guest_coords / 10  # convert to md_units

        for system, coords, host_mol, box, label in zip(
            [solvated_host_system, water_system],
            [solvated_host_coords, water_coords],
            [solvated_host_mol, water_mol],
            [host_box, water_box],
            ["protein", "solvent"],
        ):
            minimized_coords = minimizer.minimize_host_4d([guest_mol], system, coords, ff, box)

            afe = free_energy.AbsoluteFreeEnergy(guest_mol, ff)
            ups, sys_params, combined_masses, combined_coords = afe.prepare_host_edge(
                ff.get_ordered_params(), system, minimized_coords
            )

            combined_bps = []
            for up, sp in zip(ups, sys_params):
                combined_bps.append(up.bind(sp))

            works = run_leg(
                minimized_coords,
                orig_guest_coords,
                combined_bps,
                combined_masses,
                box,
                guest_name,
                label,
                host_mol,
                guest_mol,
                outdir,
                num_deletions,
                deletion_steps,
                insertion_max_lambda,
                insertion_steps,
                eq1_steps,
                fewer_outfiles,
                no_outfiles,
            )
            all_works[guest_name][label] = works
            end_time = time.time()
            print(
                f"{guest_name} {label} leg time:",
                "%.2f" % (end_time - start_time),
                "seconds",
            )
    return all_works


def run_leg(
    orig_host_coords,
    orig_guest_coords,
    combined_bps,
    combined_masses,
    host_box,
    guest_name,
    leg_type,
    host_mol,
    guest_mol,
    outdir,
    num_deletions,
    deletion_steps,
    insertion_max_lambda,
    insertion_steps,
    eq1_steps,
    fewer_outfiles=False,
    no_outfiles=False,
):
    x0 = np.concatenate([orig_host_coords, orig_guest_coords])
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

    # insert guest
    insertion_lambda_schedule = np.linspace(insertion_max_lambda, MIN_LAMBDA, insertion_steps)

    ctxt.multiple_steps(insertion_lambda_schedule, 0)  # do not collect du_dls

    lamb = insertion_lambda_schedule[-1]
    step = len(insertion_lambda_schedule) - 1

    report.report_step(
        ctxt,
        step,
        lamb,
        host_box,
        combined_bps,
        u_impls,
        guest_name,
        insertion_steps,
        f"{leg_type.upper()}_INSERTION",
    )
    if not fewer_outfiles and not no_outfiles:
        host_coords = ctxt.get_x_t()[: len(orig_host_coords)] * 10
        guest_coords = ctxt.get_x_t()[len(orig_host_coords) :] * 10
        report.write_frame(
            host_coords,
            host_mol,
            guest_coords,
            guest_mol,
            guest_name,
            outdir,
            str(step).zfill(len(str(insertion_steps))),
            f"{leg_type}-ins",
        )
    if report.too_much_force(ctxt, lamb, host_box, combined_bps, u_impls):
        return []

    # equilibrate
    equil_lambda_schedule = np.ones(eq1_steps) * MIN_LAMBDA
    lamb = equil_lambda_schedule[-1]
    step = len(equil_lambda_schedule) - 1
    ctxt.multiple_steps(equil_lambda_schedule, 0)
    report.report_step(
        ctxt,
        step,
        MIN_LAMBDA,
        host_box,
        combined_bps,
        u_impls,
        guest_name,
        eq1_steps,
        f"{leg_type.upper()}_EQUILIBRATION_1",
    )
    if not fewer_outfiles and not no_outfiles:
        host_coords = ctxt.get_x_t()[: len(orig_host_coords)] * 10
        guest_coords = ctxt.get_x_t()[len(orig_host_coords) :] * 10
        report.write_frame(
            host_coords,
            host_mol,
            guest_coords,
            guest_mol,
            guest_name,
            outdir,
            str(step).zfill(len(str(eq1_steps))),
            f"{leg_type}-eq1",
        )
    if report.too_much_force(ctxt, MIN_LAMBDA, host_box, combined_bps, u_impls):
        print("Too much force")
        return []

    # equilibrate more & shoot off deletion jobs
    steps_per_batch = 1001
    works = []
    for b in range(num_deletions):
        deletion_lambda_schedule = np.ones(steps_per_batch) * MIN_LAMBDA

        ctxt.multiple_steps(deletion_lambda_schedule, 0)
        lamb = deletion_lambda_schedule[-1]
        step = len(deletion_lambda_schedule) - 1
        report.report_step(
            ctxt,
            (b + 1) * step,
            MIN_LAMBDA,
            host_box,
            combined_bps,
            u_impls,
            guest_name,
            num_deletions * steps_per_batch,
            f"{leg_type.upper()}_EQUILIBRATION_2",
        )

        # TODO: if guest has undocked, stop simulation
        if not no_outfiles:
            host_coords = ctxt.get_x_t()[: len(orig_host_coords)] * 10
            guest_coords = ctxt.get_x_t()[len(orig_host_coords) :] * 10
            report.write_frame(
                host_coords,
                host_mol,
                guest_coords,
                guest_mol,
                guest_name,
                outdir,
                str((b + 1) * step).zfill(len(str(num_deletions * steps_per_batch))),
                f"{leg_type}-eq2",
            )
        if report.too_much_force(ctxt, MIN_LAMBDA, host_box, combined_bps, u_impls):
            print("Too much force")
            return works

        work = do_deletion(
            ctxt.get_x_t(),
            ctxt.get_v_t(),
            combined_bps,
            combined_masses,
            host_box,
            guest_name,
            leg_type,
            u_impls,
            deletion_steps,
        )
        works.append(work)

    return works


def do_deletion(
    x0,
    v0,
    combined_bps,
    combined_masses,
    box,
    guest_name,
    leg_type,
    u_impls,
    deletion_steps,
):
    seed = 2021
    intg = LangevinIntegrator(300.0, 1.5e-3, 1.0, combined_masses, seed).impl()

    ctxt = custom_ops.Context(x0, v0, box, intg, u_impls)

    # du_dl_obs = custom_ops.FullPartialUPartialLambda(u_impls, subsample_freq)
    # ctxt.add_observable(du_dl_obs)

    deletion_lambda_schedule = np.linspace(MIN_LAMBDA, DELETION_MAX_LAMBDA, deletion_steps)

    subsample_freq = 1
    full_du_dls, _, _ = ctxt.multiple_steps(deletion_lambda_schedule, subsample_freq)

    step = len(deletion_lambda_schedule) - 1
    lamb = deletion_lambda_schedule[-1]
    ctxt.step(lamb)
    report.report_step(
        ctxt,
        step,
        lamb,
        box,
        combined_bps,
        u_impls,
        guest_name,
        deletion_steps,
        f"{leg_type.upper()}_DELETION",
    )

    if report.too_much_force(ctxt, lamb, box, combined_bps, u_impls):
        print("Not calculating work (too much force)")
        return None

    # Note: this condition only applies for ABFE, not RBFE
    if abs(full_du_dls[0]) > 0.001 or abs(full_du_dls[-1]) > 0.001:
        print("Not calculating work (du_dl endpoints are not ~0)")
        return None

    work = np.trapz(full_du_dls, deletion_lambda_schedule[::subsample_freq])
    print(f"guest_name: {guest_name}\t{leg_type}_work: {work:.2f}")
    return work


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
        default="tests/data/ligands_40__first-two-ligs.sdf",
        help="guests to pose",
    )
    parser.add_argument("-o", "--outdir", default="rigorous_work_outdir", help="where to write output")
    parser.add_argument("--num_deletions", type=int, default=10, help="number of deletions to do")
    parser.add_argument("--deletion_steps", type=int, default=501, help="how many steps to delete over")
    parser.add_argument(
        "--insertion_max_lambda",
        type=float,
        default=0.5,
        help="how far away to insert from (0.0-1.0)",
    )
    parser.add_argument("--insertion_steps", type=int, default=501, help="how long to insert over")
    parser.add_argument(
        "--eq1_steps",
        type=int,
        default=5001,
        help="how long to equilibrate after insertion and before starting the deletions",
    )
    parser.add_argument("--fewer_outfiles", action="store_true", help="write fewer output pdb/sdf files")
    parser.add_argument("--no_outfiles", action="store_true", help="write no output pdb/sdf files")
    args = parser.parse_args()

    calculate_rigorous_work(
        args.host_pdbfile,
        args.guests_sdfile,
        args.outdir,
        args.num_deletions,
        args.deletion_steps,
        args.insertion_max_lambda,
        args.insertion_steps,
        args.eq1_steps,
        args.fewer_outfiles,
        args.no_outfiles,
    )


if __name__ == "__main__":
    main()
