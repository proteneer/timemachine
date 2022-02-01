import os
import time

import numpy as np

from simtk.openmm import app
from simtk.openmm.app import PDBFile

from rdkit import Chem

from timemachine.fe.utils import to_md_units
from timemachine.fe import free_energy
from ff.handlers.deserialize import deserialize_handlers
from ff import Forcefield
from timemachine.lib import LangevinIntegrator
from timemachine.lib import custom_ops

from docking import report


def pose_dock(
    host_pdbfile,
    guests_sdfile,
    transition_type,
    n_steps,
    transition_steps,
    max_lambda,
    outdir,
    random_rotation=False,
    constant_atoms=[],
):
    """Runs short simulations in which the guests phase in or out over time

    Parameters
    ----------

    host_pdbfile: path to host pdb file to dock into
    guests_sdfile: path to input sdf with guests to pose/dock
    transition_type: "insertion" or "deletion"
    n_steps: how many total steps of simulation to do (recommended: <= 1000)
    transition_steps: how many steps to insert/delete the guest over (recommended: <= 500)
        (must be <= n_steps)
    max_lambda: lambda value the guest should insert from or delete to
        (recommended: 1.0 for work calulation, 0.25 to stay close to original pose)
        (must be =1 for work calculation to be applicable)
    outdir: where to write output (will be created if it does not already exist)
    random_rotation: whether to apply a random rotation to each guest before inserting
    constant_atoms: atom numbers from the host_pdbfile to hold mostly fixed across the simulation
        (1-indexed, like PDB files)

    Output
    ------

    A pdb & sdf file for each guest's final step
      (outdir/<guest_name>_pd_<step>_host.pdb & outdir/<guest_name>_pd_<step>_guest.sdf)
    stdout for each guest noting the step number, lambda value, and energy for the last step
    stdout for each guest noting the work of transition, if applicable
    stdout for each guest noting how long it took to run

    Note
    ----
    The work will not be calculated if the du_dl endpoints are not close to 0 or if any norm of
    force per atom exceeds 20000 kJ/(mol*nm) [MAX_NORM_FORCE defined in docking/report.py]
    """
    assert transition_steps <= n_steps
    assert transition_type in ("insertion", "deletion")
    if random_rotation:
        assert transition_type == "insertion"

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    host_mol = Chem.MolFromPDBFile(host_pdbfile, removeHs=False)
    amber_ff = app.ForceField("amber99sbildn.xml", "tip3p.xml")
    host_file = PDBFile(host_pdbfile)
    host_system = amber_ff.createSystem(
        host_file.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False,
    )
    host_conf = []
    for x, y, z in host_file.positions:
        host_conf.append([to_md_units(x), to_md_units(y), to_md_units(z)])
    host_conf = np.array(host_conf)

    # TODO (ytz): we should really fix this later on. This padding was done to
    # address the particles that are too close to the boundary.
    padding = 0.1
    box_lengths = np.amax(host_conf, axis=0) - np.amin(host_conf, axis=0)
    box_lengths = box_lengths + padding
    box = np.eye(3, dtype=np.float64) * box_lengths

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

    suppl = Chem.SDMolSupplier(guests_sdfile, removeHs=False)
    for guest_mol in suppl:
        start_time = time.time()
        guest_name = guest_mol.GetProp("_Name")

        afe = free_energy.AbsoluteFreeEnergy(guest_mol, ff)

        ups, sys_params, masses, _ = afe.prepare_host_edge(ff.get_ordered_params(), host_system, host_conf)

        bps = []
        for up, sp in zip(ups, sys_params):
            bps.append(up.bind(sp))

        for atom_num in constant_atoms:
            masses[atom_num - 1] += 50000

        conformer = guest_mol.GetConformer(0)
        mol_conf = np.array(conformer.GetPositions(), dtype=np.float64)
        mol_conf = mol_conf / 10  # convert to md_units

        if random_rotation:
            center = np.mean(mol_conf, axis=0)
            mol_conf -= center
            from scipy.stats import special_ortho_group

            mol_conf = np.matmul(mol_conf, special_ortho_group.rvs(3))
            mol_conf += center

        x0 = np.concatenate([host_conf, mol_conf])  # combined geometry
        v0 = np.zeros_like(x0)

        seed = 2021
        intg = LangevinIntegrator(300, 1.5e-3, 1.0, masses, seed).impl()

        impls = []
        precision = np.float32
        for b in bps:
            p_impl = b.bound_impl(precision)
            impls.append(p_impl)

        ctxt = custom_ops.Context(x0, v0, box, intg, impls)

        if transition_type == "insertion":
            new_lambda_schedule = np.concatenate(
                [
                    np.linspace(max_lambda, 0.0, transition_steps),
                    np.zeros(n_steps - transition_steps),
                ]
            )
        elif transition_type == "deletion":
            new_lambda_schedule = np.concatenate(
                [
                    np.linspace(0.0, max_lambda, transition_steps),
                    np.ones(n_steps - transition_steps) * max_lambda,
                ]
            )
        else:
            raise (RuntimeError('invalid `transition_type` (must be one of ["insertion", "deletion"])'))

        calc_work = True

        # (ytz): we gotta figure out how to batch this code, tbd: batch this
        # collect a du_dl calculation every step
        subsample_du_dl_interval = 1

        full_du_dls, _, _ = ctxt.multiple_steps(new_lambda_schedule, subsample_du_dl_interval)

        step = len(new_lambda_schedule) - 1
        final_lamb = new_lambda_schedule[-1]
        report.report_step(ctxt, step, final_lamb, box, bps, impls, guest_name, n_steps, "pose_dock")
        host_coords = ctxt.get_x_t()[: len(host_conf)] * 10
        guest_coords = ctxt.get_x_t()[len(host_conf) :] * 10
        report.write_frame(
            host_coords,
            host_mol,
            guest_coords,
            guest_mol,
            guest_name,
            outdir,
            step,
            "pd",
        )

        if report.too_much_force(ctxt, final_lamb, box, bps, impls):
            print("Not calculating work (too much force)")
            calc_work = False
            break

        # Note: this condition only applies for ABFE, not RBFE
        if abs(full_du_dls[0]) > 0.001 or abs(full_du_dls[-1]) > 0.001:
            print("Not calculating work (du_dl endpoints are not ~0)")
            calc_work = False

        if calc_work:
            work = np.trapz(full_du_dls, new_lambda_schedule[::subsample_du_dl_interval])
            print(f"guest_name: {guest_name}\twork: {work:.2f}")
        end_time = time.time()
        print(f"{guest_name} took {(end_time - start_time):.2f} seconds")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Poses guests into a host by running short simulations in which the guests phase in over time",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        default="tests/data/ligands_40.sdf",
        help="guests to pose",
    )
    parser.add_argument("-t", "--transition_type", help="'insertion' or 'deletion'", default="insertion")
    parser.add_argument(
        "--n_steps",
        type=int,
        default=1001,
        help="total simulation length (1 step = 1.5 femtoseconds)",
    )
    parser.add_argument(
        "--transition_steps",
        type=int,
        default=500,
        help="how many steps to take while phasing in or out the guest (must be <= n_steps)",
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
    parser.add_argument("-o", "--outdir", default="pose_dock_outdir", help="where to write output")
    parser.add_argument(
        "--random_rotation",
        action="store_true",
        help="apply a random rotation to each guest before inserting",
    )
    parser.add_argument(
        "-c",
        "--constant_atoms_file",
        help="file containing comma-separated atom numbers to hold ~fixed",
    )
    args = parser.parse_args()
    print(args)

    constant_atoms_list = []

    if args.constant_atoms_file:
        with open(args.constant_atoms_file, "r") as rfile:
            for line in rfile.readlines():
                atoms = [int(x.strip()) for x in line.strip().split(",")]
                constant_atoms_list += atoms

    pose_dock(
        args.host_pdbfile,
        args.guests_sdfile,
        args.transition_type,
        args.n_steps,
        args.transition_steps,
        args.max_lambda,
        args.outdir,
        random_rotation=args.random_rotation,
        constant_atoms=constant_atoms_list,
    )
