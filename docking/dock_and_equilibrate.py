"""Solvates a host, inserts guest(s) into solvated host, equilibrates
"""
import os
import time

import numpy as np

from rdkit import Chem
from rdkit.Chem.rdmolfiles import PDBWriter, SDWriter
from rdkit.Geometry import Point3D

from md import builders, Recipe
from fe import pdb_writer
from timemachine.lib import potentials, custom_ops, LangevinIntegrator
from ff.handlers.deserialize import deserialize_handlers


MAX_NORM_FORCE = 20000


def dock_and_equilibrate(
    host_pdbfile,
    guests_sdfile,
    max_lambda,
    insertion_steps,
    eq_steps,
    outdir,
    fewer_outfiles=False,
    constant_atoms=[],
    skip_errors=False,
):
    """Solvates a host, inserts guest(s) into solvated host, equilibrates

    Parameters
    ----------

    host_pdbfile: path to host pdb file to dock into
    guests_sdfile: path to input sdf with guests to pose/dock
    max_lambda: lambda value the guest should insert from or delete to
        (recommended: 1.1 for work calulation, 0.25 to stay close to original pose)
        (must be >1 for work calculation to be applicable)
    insertion_steps: how many steps to insert the guest over (recommended: 501)
    eq_steps: how many steps of equilibration to do after insertion (recommended: 15001)
    outdir: where to write output (will be created if it does not already exist)
    fewer_outfiles: if True, will only write frames for the equilibration, not insertion
    constant_atoms: atom numbers from the host_pdbfile to hold mostly fixed across the simulation
        (1-indexed, like PDB files)
    skip_errors: if True, will report errors to stdout and continue on to the next guest.
        If False, will halt upon errors.

    Output
    ------

    A pdb & sdf file every 100 steps of insertion (outdir/<guest_name>/<guest_name>_<step>.[pdb/sdf])
    A pdb & sdf file every 1000 steps of equilibration (outdir/<guest_name>/<guest_name>_<step>.[pdb/sdf])
    stdout every 100(0) steps noting the step number, lambda value, and energy
    stdout for each guest noting the work of transition
    stdout for each guest noting how long it took to run

    Note
    ----
    If any norm of force per atom exceeds 20000 kJ/(mol*nm), the simulation for that
    guest will stop and the work will not be calculated.
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
    MAX_NORM_FORCE = {MAX_NORM_FORCE}
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
    host_recipe = Recipe.from_openmm(solvated_host_system)

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
        try:
            guest_recipe = Recipe.from_rdkit(guest_mol, guest_ff_handlers)
        except TypeError as e:
            if skip_errors:
                print(e)
                continue
            else:
                raise e

        # let guest be affected by lambda
        for bp in guest_recipe.bound_potentials:
            if isinstance(bp, potentials.Nonbonded):
                array = bp.get_lambda_offset_idxs()
                array[:] = 1

        host_guest_coords = np.concatenate([solvated_host_coords, orig_guest_coords])
        host_guest_recipe = host_recipe.combine(guest_recipe)
        x0 = host_guest_coords
        v0 = np.zeros_like(x0)
        print(
            f"SYSTEM", f"guest_name: {guest_name}", f"num_atoms: {len(x0)}",
        )

        masses = host_guest_recipe.masses
        for atom_num in constant_atoms:
            masses[atom_num - 1] += 50000

        seed = 2020
        intg = LangevinIntegrator(300.0, 1.5e-3, 1.0, masses, seed).impl()

        u_impls = []
        for bp in host_guest_recipe.bound_potentials:
            bp_impl = bp.bound_impl(precision=np.float32)
            u_impls.append(bp_impl)

        ctxt = custom_ops.Context(x0, v0, host_box, intg, u_impls)

        # collect a du_dl calculation once every other step
        subsample_freq = 2
        du_dl_obs = custom_ops.FullPartialUPartialLambda(u_impls, subsample_freq)
        ctxt.add_observable(du_dl_obs)

        # insert guest
        insertion_lambda_schedule = np.linspace(
            max_lambda, 0.0, insertion_steps
        )
        calc_work = True
        for step, lamb in enumerate(insertion_lambda_schedule):
            ctxt.step(lamb)
            if step % 100 == 0:
                print(
                    f"INSERTION\t"
                    f"guest_name: {guest_name}\t"
                    f"step: {str(step).zfill(len(str(insertion_steps)))}\t"
                    f"lambda: {lamb:.2f}\t"
                    f"energy: {ctxt.get_u_t():.2f}"
                )
                if not fewer_outfiles:
                    host_coords = ctxt.get_x_t()[: len(solvated_host_coords)] * 10
                    guest_coords = ctxt.get_x_t()[len(solvated_host_coords) :] * 10
                    write_frame(
                        host_coords,
                        solvated_host_mol,
                        guest_coords,
                        guest_mol,
                        outdir,
                        guest_name,
                        str(step).zfill(len(str(insertion_steps))),
                        f"ins",
                    )
            if too_much_force(ctxt, host_guest_recipe, host_box, u_impls, lamb):
                calc_work = False
                break

        if (
            abs(du_dl_obs.full_du_dl()[0]) > 0.001
            or abs(du_dl_obs.full_du_dl()[-1]) > 0.001
        ):
            print("Error: du_dl endpoints are not ~0")
            calc_work = False

        if calc_work:
            work = np.trapz(
                du_dl_obs.full_du_dl(), insertion_lambda_schedule[::subsample_freq]
            )
            print(f"guest_name: {guest_name}\tinsertion_work: {work:.2f}")

        # equilibrate
        for step in range(eq_steps):
            ctxt.step(0.00)
            if step % 1000 == 0:
                print(
                    f"EQUILIBRATION\t"
                    f"guest_name: {guest_name}\t"
                    f"step: {str(step).zfill(len(str(eq_steps)))}\t"
                    f"lambda: 0.00\t"
                    f"energy: {ctxt.get_u_t():.2f}"
                )
                host_coords = ctxt.get_x_t()[: len(solvated_host_coords)] * 10
                guest_coords = ctxt.get_x_t()[len(solvated_host_coords) :] * 10
                write_frame(
                    host_coords,
                    solvated_host_mol,
                    guest_coords,
                    guest_mol,
                    outdir,
                    guest_name,
                    str(step).zfill(len(str(eq_steps))),
                    f"eq",
                )
            if too_much_force(ctxt, host_guest_recipe, host_box, u_impls, 0.0):
                break

        end_time = time.time()
        print(f"{guest_name} took {(end_time - start_time):.2f} seconds")


def too_much_force(ctxt, recipe, box, u_impls, lamb):
    forces = ctxt.get_du_dx_t()
    norm_forces = np.linalg.norm(forces, axis=-1)
    if np.any(norm_forces > MAX_NORM_FORCE):
        print("Error: at least one force is too large to continue")
        print("max norm force", np.amax(norm_forces))
        for bp, u in zip(recipe.bound_potentials, u_impls):
            du_dx, _, _ = u.execute(ctxt.get_x_t(), box, lamb)
            print(
                bp,
                "atom",
                np.argmax(np.linalg.norm(du_dx, axis=-1)),
                "max norm force",
                np.amax(np.linalg.norm(du_dx, axis=-1)),
            )
        return True
    return False


def write_frame(
    host_coords, host_mol, guest_coords, guest_mol, outdir, guest_name, step, sim_info
):
    if not os.path.exists(os.path.join(outdir, guest_name)):
        os.mkdir(os.path.join(outdir, guest_name))

    host_frame = host_mol.GetConformer()
    for i in range(host_mol.GetNumAtoms()):
        x, y, z = host_coords[i]
        host_frame.SetAtomPosition(i, Point3D(x, y, z))
    conf_id = host_mol.AddConformer(host_frame)
    writer = PDBWriter(
        os.path.join(outdir, guest_name, f"{guest_name}_{sim_info}_{step}_host.pdb",)
    )
    writer.write(host_mol, conf_id)
    writer.close()
    host_mol.RemoveConformer(conf_id)

    guest_frame = guest_mol.GetConformer()
    for i in range(guest_mol.GetNumAtoms()):
        x, y, z = guest_coords[i]
        guest_frame.SetAtomPosition(i, Point3D(x, y, z))
    conf_id = guest_mol.AddConformer(guest_frame)
    guest_mol.SetProp("_Name", f"{guest_name}_{sim_info}_{step}_guest")
    writer = SDWriter(
        os.path.join(outdir, guest_name, f"{guest_name}_{sim_info}_{step}_guest.sdf",)
    )
    writer.write(guest_mol, conf_id)
    writer.close()
    guest_mol.RemoveConformer(conf_id)


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
        default=0.25,
        help=(
            "lambda value the guest should insert from or delete to "
            "(must be >1 for the work calculation to be applicable)"
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
    parser.add_argument(
        "--skip_errors",
        action="store_true",
        help=(
            "Report errors to stdout and continue on to the next guest. "
            "Otherwise, will halt upon errors."
        ),
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
        args.skip_errors,
    )


if __name__ == "__main__":
    main()
