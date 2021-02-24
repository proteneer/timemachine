"""
1. Solvates a host, inserts guest(s) into solvated host, equilibrates, spins off deletion jobs, calculates work
2. Creates a water box, inserts guest(s) into water box, equilibrates, spins off deletion jobs, calculates work
"""
import os
import time

import numpy as np

from rdkit import Chem
from rdkit.Chem.rdmolfiles import PDBWriter, SDWriter
from rdkit.Geometry import Point3D

from md import builders
from fe import pdb_writer, free_energy
from ff import Forcefield
from ff.handlers import openmm_deserializer
from ff.handlers.deserialize import deserialize_handlers
from timemachine.lib import potentials, custom_ops, LangevinIntegrator

from testsystems.relative import hif2a_ligand_pair

import report

INSERTION_MAX_LAMBDA = 0.5
DELETION_MAX_LAMBDA = 1.0
MIN_LAMBDA = 0.0
TRANSITION_STEPS = 1001
EQ1_STEPS = 5001
NUM_DELETIONS = 10
# EQ2_STEPS = 10001


def calculate_rigorous_work(
        host_pdbfile, mol_a, mol_b, core, outdir, fewer_outfiles=False, no_outfiles=False
):
    """
    """

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print(
        f"""
    HOST_PDBFILE = {host_pdbfile}
    OUTDIR = {outdir}

    INSERTION_MAX_LAMBDA = {INSERTION_MAX_LAMBDA}
    DELETION_MAX_LAMBDA = {DELETION_MAX_LAMBDA}
    MIN_LAMBDA = {MIN_LAMBDA}
    TRANSITION_STEPS = {TRANSITION_STEPS}
    EQ1_STEPS = {EQ1_STEPS}
    NUM_DELETIONS = {NUM_DELETIONS}
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
    print("host box", host_box)

    solvated_host_pdb = os.path.join(outdir, "solvated_host.pdb")
    writer = pdb_writer.PDBWriter([solvated_topology], solvated_host_pdb)
    writer.write_frame(solvated_host_coords)
    writer.close()
    solvated_host_mol = Chem.MolFromPDBFile(solvated_host_pdb, removeHs=False)
    if no_outfiles:
        os.remove(solvated_host_pdb)

    # Prepare water box
    print("Generating water box...")
    # TODO: water box probably doesn't need to be this big
    box_lengths = host_box[np.diag_indices(3)]
    water_box_width = min(box_lengths)
    (
        water_system,
        orig_water_coords,
        water_box,
        water_topology,
    ) = builders.build_water_system(water_box_width)

    # sometimes water boxes are sad. should be minimized first; this is a workaround
    water_box += np.eye(3) * 0.1
    print("water box", water_box)

    # it's okay if the water box here and the solvated protein box don't align -- they have PBCs
    water_pdb = os.path.join(outdir, "water_box.pdb")
    writer = pdb_writer.PDBWriter([water_topology], water_pdb)
    writer.write_frame(orig_water_coords)
    writer.close()
    water_mol = Chem.MolFromPDBFile(water_pdb, removeHs=False)
    if no_outfiles:
        os.remove(water_pdb)

    # Run the procedure
    print("Getting guests...")

    start_time = time.time()
    guest_name_a = mol_a.GetProp("_Name")
    guest_name_b = mol_b.GetProp("_Name")
    combined_name = guest_name_a +'_'+guest_name_b

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

    from md import minimizer
    from fe import topology
    # minimize w.r.t. both mol_a and mol_b?
    solvated_host_coords = minimizer.minimize_host_4d([mol_a], solvated_host_system, solvated_host_coords, ff, host_box)

    single_topology = topology.SingleTopology(mol_a, mol_b, core, ff)
    rfe = free_energy.RelativeFreeEnergy(single_topology)

    ups, sys_params, combined_masses, combined_coords = rfe.prepare_host_edge(
        ff.get_ordered_params(), solvated_host_system, solvated_host_coords
    )

    combined_bps = []
    for up, sp in zip(ups, sys_params):
        combined_bps.append(up.bind(sp))

    works = run_leg(
        combined_coords,
        combined_bps,
        combined_masses,
        host_box,
        combined_name,
        "host",
        outdir,
        fewer_outfiles,
        no_outfiles,
    )
    end_time = time.time()
    print(
        f"{combined_name} host leg time:", "%.2f" % (end_time - start_time), "seconds"
    )



def run_leg(
    combined_coords,
    combined_bps,
    combined_masses,
    host_box,
    guest_name,
    leg_type,
    outdir,
    fewer_outfiles=False,
    no_outfiles=False,
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

    # equilibrate more & shoot off deletion jobs
    steps_per_batch = 1000

    works = []
    for b in range(NUM_DELETIONS):
        equil2_lambda_schedule = np.ones(steps_per_batch)*MIN_LAMBDA
        ctxt.multiple_steps(equil2_lambda_schedule, 0)
        lamb = equil2_lambda_schedule[-1]
        step = len(equil2_lambda_schedule)-1
        report.report_step(
            ctxt,
            step,
            MIN_LAMBDA,
            host_box,
            combined_bps,
            u_impls,
            guest_name,
            NUM_DELETIONS*steps_per_batch,
            f"{leg_type.upper()}_EQUILIBRATION_2",
        )

        if report.too_much_force(ctxt, MIN_LAMBDA, host_box, combined_bps, u_impls):
            return

        work = do_deletion(
            ctxt.get_x_t(),
            ctxt.get_v_t(),
            combined_bps,
            combined_masses,
            host_box,
            guest_name,
            leg_type,
        )
        works.append(work)

    print(works)
    return works


def do_deletion(x0, v0, combined_bps, combined_masses, box, guest_name, leg_type):
    seed = 2021
    intg = LangevinIntegrator(300.0, 1.5e-3, 1.0, combined_masses, seed).impl()

    u_impls = []
    for bp in combined_bps:
        bp_impl = bp.bound_impl(precision=np.float32)
        u_impls.append(bp_impl)

    ctxt = custom_ops.Context(x0, v0, box, intg, u_impls)

    # du_dl_obs = custom_ops.FullPartialUPartialLambda(u_impls, subsample_freq)
    # ctxt.add_observable(du_dl_obs)

    deletion_lambda_schedule = np.linspace(
        MIN_LAMBDA, DELETION_MAX_LAMBDA, TRANSITION_STEPS
    )

    calc_work = True

    subsample_freq = 1
    full_du_dls = ctxt.multiple_steps(deletion_lambda_schedule, subsample_freq)

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
        TRANSITION_STEPS,
        f"{leg_type.upper()}_DELETION",
    )

    if report.too_much_force(ctxt, lamb, box, combined_bps, u_impls):
        calc_work = False
        return

    if calc_work:
        work = np.trapz(
            full_du_dls,
            deletion_lambda_schedule[::subsample_freq]
        )
        print(f"guest_name: {guest_name}\t{leg_type}_work: {work:.2f}")
        return work
    return


def main():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-s",
        "--guests_sdfile",
        default="tests/data/ligands_40.sdf",
        help="guests to pose",
    )
    parser.add_argument(
        "-p",
        "--host_pdbfile",
        default="tests/data/hif2a_nowater_min.pdb",
        help="host to dock into",
    )
    parser.add_argument(
        "-o", "--outdir", default="rigorous_work_outdir", help="where to write output"
    )
    parser.add_argument(
        "--fewer_outfiles", action="store_true", help="write fewer output pdb/sdf files"
    )
    parser.add_argument(
        "--no_outfiles", action="store_true", help="write no output pdb/sdf files"
    )
    args = parser.parse_args()

    # fetch mol_a, mol_b, core, forcefield from testsystem
    mol_a, mol_b, core = hif2a_ligand_pair.mol_a, hif2a_ligand_pair.mol_b, hif2a_ligand_pair.core

    print(core)
    print(core.shape)
    core_rev = core[:,::-1]
    print(core_rev)

    calculate_rigorous_work(
        args.host_pdbfile,
        mol_a,
        mol_b,
        core,
        args.outdir,
        args.fewer_outfiles,
        args.no_outfiles,
    )

    calculate_rigorous_work(
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
