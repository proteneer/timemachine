"""Solvates a host, inserts guest(s) into solvated host, equilibrates
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

import report


def dock_and_equilibrate(
    host_pdbfile,
    guests_sdfile,
    max_lambda,
    insertion_steps,
    eq_steps,
    outdir,
    fewer_outfiles=False,
    constant_atoms=[]
):
    """Solvates a host, inserts guest(s) into solvated host, equilibrates

    Parameters
    ----------

    host_pdbfile: path to host pdb file to dock into
    guests_sdfile: path to input sdf with guests to pose/dock
    max_lambda: lambda value the guest should insert from or delete to
        (recommended: 1.0 for work calulation, 0.25 to stay close to original pose)
        (must be =1 for work calculation to be applicable)
    insertion_steps: how many steps to insert the guest over (recommended: 501)
    eq_steps: how many steps of equilibration to do after insertion (recommended: 15001)
    outdir: where to write output (will be created if it does not already exist)
    fewer_outfiles: if True, will only write frames for the equilibration, not insertion
    constant_atoms: atom numbers from the host_pdbfile to hold mostly fixed across the simulation
        (1-indexed, like PDB files)

    Output
    ------

    A pdb & sdf file every 100 steps of insertion (outdir/<guest_name>/<guest_name>_<step>.[pdb/sdf])
    A pdb & sdf file every 1000 steps of equilibration (outdir/<guest_name>/<guest_name>_<step>.[pdb/sdf])
    stdout every 100(0) steps noting the step number, lambda value, and energy
    stdout for each guest noting the work of transition
    stdout for each guest noting how long it took to run

    Note
    ----
    If any norm of force per atom exceeds 20000 kJ/(mol*nm) [MAX_NORM_FORCE defined in docking/report.py],
    the simulation for that guest will stop and the work will not be calculated.
    """

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print(
        f"""
    HOST_PDBFILE = {host_pdbfile}
    GUESTS_SDFILE = {guests_sdfile}
    OUTDIR = {outdir}
    MAX_LAMBDA = {max_lambda}
    INSERTION_STEPS = {insertion_steps}
    EQ_STEPS = {eq_steps}
    """
    )

    # Prepare host
    # TODO: handle extra (non-transitioning) guests?
    print("Solvating host...")
    # TODO: return topology from builders.build_protein_system
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
    os.remove(solvated_host_pdb)

    # Run the procedure
    print("Getting guests...")
    suppl = Chem.SDMolSupplier(guests_sdfile, removeHs=False)
    for guest_mol in suppl:
        start_time = time.time()
        guest_name = guest_mol.GetProp("_Name")
        guest_conformer = guest_mol.GetConformer(0)
        orig_guest_coords = np.array(guest_conformer.GetPositions(), dtype=np.float64)
        orig_guest_coords = orig_guest_coords / 10  # convert to md_units
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

        afe = free_energy.AbsoluteFreeEnergy(guest_mol, ff)

        ups, sys_params, combined_masses, _ = afe.prepare_host_edge(ff.get_ordered_params(), solvated_host_system, solvated_host_coords)

        combined_bps = []
        for up, sp in zip(ups, sys_params):
            combined_bps.append(up.bind(sp))

        x0 = np.concatenate([solvated_host_coords, orig_guest_coords])
        v0 = np.zeros_like(x0)
        print(
            f"SYSTEM", f"guest_name: {guest_name}", f"num_atoms: {len(x0)}",
        )

        for atom_num in constant_atoms:
            combined_masses[atom_num - 1] += 50000

        seed = 2021
        intg = LangevinIntegrator(300.0, 1.5e-3, 1.0, combined_masses, seed).impl()

        u_impls = []
        for bp in combined_bps:
            bp_impl = bp.bound_impl(precision=np.float32)
            u_impls.append(bp_impl)

        ctxt = custom_ops.Context(x0, v0, host_box, intg, u_impls)

        # insert guest
        insertion_lambda_schedule = np.linspace(
            max_lambda, 0.0, insertion_steps
        )
        calc_work = True


        # collect a du_dl calculation once every other step
        subsample_freq = 1

        full_du_dls = ctxt.multiple_steps(insertion_lambda_schedule, subsample_freq)
        step = len(insertion_lambda_schedule) - 1
        lamb = insertion_lambda_schedule[-1]
        ctxt.step(lamb)

        report.report_step(ctxt, step, lamb, host_box, combined_bps, u_impls, guest_name, insertion_steps, "INSERTION")
        if not fewer_outfiles:
            host_coords = ctxt.get_x_t()[: len(solvated_host_coords)] * 10
            guest_coords = ctxt.get_x_t()[len(solvated_host_coords) :] * 10
            report.write_frame(
                host_coords,
                solvated_host_mol,
                guest_coords,
                guest_mol,
                guest_name,
                outdir,
                str(step).zfill(len(str(insertion_steps))),
                f"ins",
            )

        if report.too_much_force(ctxt, lamb, host_box, combined_bps, u_impls):
            calc_work = False
            break

        # Note: this condition only applies for ABFE, not RBFE
        if (
            abs(full_du_dls[0]) > 0.001
            or abs(full_du_dls[-1]) > 0.001
        ):
            print("Error: du_dl endpoints are not ~0")
            calc_work = False

        if calc_work:
            work = np.trapz(
                full_du_dls, insertion_lambda_schedule[::subsample_freq]
            )
            print(f"guest_name: {guest_name}\tinsertion_work: {work:.2f}")

        # equilibrate
        for step in range(eq_steps):
            ctxt.step(0.00)
            if step % 1000 == 0:
                report.report_step(ctxt, step, 0.00, host_box, combined_bps, u_impls, guest_name, eq_steps, 'EQUILIBRATION')
                host_coords = ctxt.get_x_t()[: len(solvated_host_coords)] * 10
                guest_coords = ctxt.get_x_t()[len(solvated_host_coords) :] * 10
                report.write_frame(
                    host_coords,
                    solvated_host_mol,
                    guest_coords,
                    guest_mol,
                    guest_name,
                    outdir,
                    str(step).zfill(len(str(eq_steps))),
                    f"eq",
                )
            if step in (0, int(eq_steps/2), eq_steps-1):
                if report.too_much_force(ctxt, 0.00, host_box, combined_bps, u_impls):
                    break

        end_time = time.time()
        print(f"{guest_name} took {(end_time - start_time):.2f} seconds")


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
        "-c",
        "--constant_atoms_file",
        help="file containing comma-separated atom numbers to hold ~fixed",
    )
    parser.add_argument(
        "--max_lambda",
        type=float,
        default=1.0,
        help=(
            "lambda value the guest should insert from or delete to "
            "(must be =1 for the work calculation to be applicable)"
        ),
    )
    parser.add_argument(
        "--eq_steps",
        type=int,
        default=15001,
        help="equilibration length (1 step = 1.5 femtoseconds)",
    )
    parser.add_argument(
        "--insertion_steps",
        type=int,
        default=501,
        help="how many steps to take while phasing in each guest",
    )
    parser.add_argument(
        "-o", "--outdir", default="dock_equil_out", help="where to write output"
    )
    parser.add_argument(
        "--fewer_outfiles", action="store_true", help="write fewer output pdb/sdf files"
    )
    args = parser.parse_args()

    constant_atoms_list = []
    if args.constant_atoms_file:
        with open(args.constant_atoms_file, "r") as rfile:
            for line in rfile.readlines():
                atoms = [int(x.strip()) for x in line.strip().split(",")]
                constant_atoms_list += atoms

    dock_and_equilibrate(
        args.host_pdbfile,
        args.guests_sdfile,
        args.max_lambda,
        args.insertion_steps,
        args.eq_steps,
        args.outdir,
        args.fewer_outfiles,
        constant_atoms_list,
    )


if __name__ == "__main__":
    main()
