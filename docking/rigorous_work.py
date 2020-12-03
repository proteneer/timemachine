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

from md import builders, Recipe
from fe import pdb_writer
from timemachine.lib import potentials, custom_ops, LangevinIntegrator
from ff.handlers.deserialize import deserialize_handlers


INSERTION_MAX_LAMBDA = 0.5
DELETION_MAX_LAMBDA = 1.1
MIN_LAMBDA = 0.0
TRANSITION_STEPS = 501
EQ1_STEPS = 5001
EQ2_STEPS = 10001
MAX_NORM_FORCE = 20000


def calculate_rigorous_work(host_pdbfile, guests_sdfile, outdir, fewer_outfiles=False, no_outfiles=False):
    """
    """

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print(
        f"""
    HOST_PDBFILE = {host_pdbfile}
    GUESTS_SDFILE = {guests_sdfile}
    OUTDIR = {outdir}

    INSERTION_MAX_LAMBDA = {INSERTION_MAX_LAMBDA}
    DELETION_MAX_LAMBDA = {DELETION_MAX_LAMBDA}
    MIN_LAMBDA = {MIN_LAMBDA}
    TRANSITION_STEPS = {TRANSITION_STEPS}
    EQ1_STEPS = {EQ1_STEPS}
    EQ2_STEPS = {EQ2_STEPS}
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
    if no_outfiles:
        os.remove(solvated_host_pdb)
    host_recipe = Recipe.from_openmm(solvated_host_system)

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
    water_recipe = Recipe.from_openmm(water_system)

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
        guest_recipe = Recipe.from_rdkit(guest_mol, guest_ff_handlers)

        # let guest be affected by lambda
        for bp in guest_recipe.bound_potentials:
            if isinstance(bp, potentials.Nonbonded):
                array = bp.get_lambda_offset_idxs()
                array[:] = 1

        run_leg(
            solvated_host_coords,
            orig_guest_coords,
            host_recipe,
            guest_recipe,
            host_box,
            guest_name,
            "host",
            solvated_host_mol,
            guest_mol,
            outdir,
            fewer_outfiles,
            no_outfiles
        )
        end_time = time.time()
        print(
            f"{guest_name} host leg time:", "%.2f" % (end_time - start_time), "seconds"
        )

        start_time = time.time()
        run_leg(
            orig_water_coords,
            orig_guest_coords,
            water_recipe,
            guest_recipe,
            water_box,
            guest_name,
            "water",
            water_mol,
            guest_mol,
            outdir,
            fewer_outfiles,
            no_outfiles
        )
        end_time = time.time()
        print(
            f"{guest_name} water leg time:", "%.2f" % (end_time - start_time), "seconds"
        )


def run_leg(
    orig_host_coords,
    orig_guest_coords,
    host_recipe,
    guest_recipe,
    host_box,
    guest_name,
    leg_type,
    host_mol,
    guest_mol,
    outdir,
    fewer_outfiles=False,
    no_outfiles=False
):
    host_guest_coords = np.concatenate([orig_host_coords, orig_guest_coords])
    host_guest_recipe = host_recipe.combine(guest_recipe)

    x0 = host_guest_coords
    v0 = np.zeros_like(x0)
    print(
        f"{leg_type.upper()}_SYSTEM",
        f"guest_name: {guest_name}",
        f"num_atoms: {len(x0)}",
    )

    seed = 2020
    intg = LangevinIntegrator(300.0, 1.5e-3, 1.0, host_guest_recipe.masses, seed).impl()

    u_impls = []
    for bp in host_guest_recipe.bound_potentials:
        bp_impl = bp.bound_impl(precision=np.float32)
        u_impls.append(bp_impl)

    ctxt = custom_ops.Context(x0, v0, host_box, intg, u_impls)

    # insert guest
    insertion_lambda_schedule = np.linspace(
        INSERTION_MAX_LAMBDA, MIN_LAMBDA, TRANSITION_STEPS
    )
    for step, lamb in enumerate(insertion_lambda_schedule):
        ctxt.step(lamb)
        if step % 100 == 0:
            print(
                f"{leg_type.upper()}_INSERTION\t"
                f"guest_name: {guest_name}\t"
                f"step: {str(step).zfill(len(str(TRANSITION_STEPS)))}\t"
                f"lambda: {lamb:.2f}\t"
                f"energy: {ctxt.get_u_t():.2f}"
            )
            if not fewer_outfiles and not no_outfiles:
                host_coords = ctxt.get_x_t()[: len(orig_host_coords)] * 10
                guest_coords = ctxt.get_x_t()[len(orig_host_coords) :] * 10
                write_frame(
                    host_coords,
                    host_mol,
                    guest_coords,
                    guest_mol,
                    outdir,
                    guest_name,
                    str(step).zfill(len(str(TRANSITION_STEPS))),
                    f"{leg_type}-ins",
                )
        if too_much_force(ctxt, host_guest_recipe, host_box, u_impls, lamb):
            return

    # equilibrate
    for step in range(EQ1_STEPS):
        ctxt.step(MIN_LAMBDA)
        if step % 100 == 0:
            print(
                f"{leg_type.upper()}_EQUILIBRATION_1\t"
                f"guest_name: {guest_name}\t"
                f"step: {str(step).zfill(len(str(EQ1_STEPS)))}\t"
                f"lambda: {MIN_LAMBDA:.2f}\t"
                f"energy: {ctxt.get_u_t():.2f}"
            )
            if step % 1000 == 0:
                if not fewer_outfiles and not no_outfiles:
                    host_coords = ctxt.get_x_t()[: len(orig_host_coords)] * 10
                    guest_coords = ctxt.get_x_t()[len(orig_host_coords) :] * 10
                    write_frame(
                        host_coords,
                        host_mol,
                        guest_coords,
                        guest_mol,
                        outdir,
                        guest_name,
                        str(step).zfill(len(str(EQ1_STEPS))),
                        f"{leg_type}-eq1",
                    )
        if too_much_force(ctxt, host_guest_recipe, host_box, u_impls, MIN_LAMBDA):
            return

    # equilibrate more & shoot off deletion jobs
    for step in range(EQ2_STEPS):
        ctxt.step(MIN_LAMBDA)
        if step % 100 == 0:
            print(
                f"{leg_type.upper()}_EQUILIBRATION_2\t"
                f"guest_name: {guest_name}\t"
                f"step: {str(step).zfill(len(str(EQ2_STEPS)))}\t"
                f"lambda: {MIN_LAMBDA:.2f}\t"
                f"energy: {ctxt.get_u_t():.2f}"
            )
        if too_much_force(ctxt, host_guest_recipe, host_box, u_impls, MIN_LAMBDA):
            return

        if step % 1000 == 0:
            # TODO: if guest has undocked, stop simulation
            if not no_outfiles:
                host_coords = ctxt.get_x_t()[: len(orig_host_coords)] * 10
                guest_coords = ctxt.get_x_t()[len(orig_host_coords) :] * 10
                write_frame(
                    host_coords,
                    host_mol,
                    guest_coords,
                    guest_mol,
                    outdir,
                    guest_name,
                    str(step).zfill(len(str(EQ2_STEPS))),
                    f"{leg_type}-eq2",
                )

            do_deletion(
                ctxt.get_x_t(),
                ctxt.get_v_t(),
                host_guest_recipe,
                host_box,
                guest_name,
                leg_type,
            )


def do_deletion(x0, v0, combined_recipe, box, guest_name, leg_type):
    seed = 2020
    intg = LangevinIntegrator(300.0, 1.5e-3, 1.0, combined_recipe.masses, seed).impl()

    u_impls = []
    for bp in combined_recipe.bound_potentials:
        bp_impl = bp.bound_impl(precision=np.float32)
        u_impls.append(bp_impl)

    ctxt = custom_ops.Context(x0, v0, box, intg, u_impls)

    subsample_freq = 2
    du_dl_obs = custom_ops.FullPartialUPartialLambda(u_impls, subsample_freq)
    ctxt.add_observable(du_dl_obs)

    deletion_lambda_schedule = np.linspace(
        MIN_LAMBDA, DELETION_MAX_LAMBDA, TRANSITION_STEPS
    )
    for step, lamb in enumerate(deletion_lambda_schedule):
        ctxt.step(lamb)
        if step % 100 == 0:
            print(
                f"{leg_type.upper()}_DELETION\t"
                f"guest_name: {guest_name}\t"
                f"step: {str(step).zfill(len(str(TRANSITION_STEPS)))}\t"
                f"lambda: {lamb:.2f}\t"
                f"energy: {ctxt.get_u_t():.2f}"
            )
        if too_much_force(ctxt, combined_recipe, box, u_impls, lamb):
            return

    calc_work = True
    if (
        abs(du_dl_obs.full_du_dl()[0]) > 0.001
        or abs(du_dl_obs.full_du_dl()[-1]) > 0.001
    ):
        print("Error: du_dl endpoints are not ~0")
        calc_work = False

    if calc_work:
        work = np.trapz(
            du_dl_obs.full_du_dl(), deletion_lambda_schedule[::subsample_freq]
        )
        print(f"guest_name: {guest_name}\t{leg_type}_work: {work:.2f}")


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
        "-o", "--outdir", default="rigorous_work_outdir", help="where to write output"
    )
    parser.add_argument("--fewer_outfiles", action="store_true", help="write fewer output pdb/sdf files")
    parser.add_argument("--no_outfiles", action="store_true", help="write no output pdb/sdf files")
    args = parser.parse_args()

    calculate_rigorous_work(args.host_pdbfile, args.guests_sdfile, args.outdir, args.fewer_outfiles, args.no_outfiles)


if __name__ == "__main__":
    main()
