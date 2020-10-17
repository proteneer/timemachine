import os
import numpy as np

from rdkit import Chem
import simtk.openmm
from simtk.openmm import app
from simtk.openmm.app import PDBFile
from fe.pdb_writer import PDBWriter

from docking import dock_setup
from ff.handlers.deserialize import deserialize_handlers
from timemachine.lib import LangevinIntegrator
from timemachine.lib import custom_ops
from io import StringIO

from fe import system
from fe.utils import to_md_units

from matplotlib import pyplot as plt


def pose_dock(
    guests_sdfile,
    host_pdbfile,
    outdir,
    n_steps,
    lowering_steps,
    start_lambda,
    random_rotation=False,
    constant_atoms=[],
):
    """Poses guests into a host by running short simulations in which the guests phase in over time

    Parameters
    ----------

    guests_sdfile: path to input sdf with guests to pose/dock
    host_pdbfile: path to host pdb file to dock into
    outdir: where to write output (will be created if it does not already exist)
    n_steps: how many total steps of simulation to do (recommended: <= 20000)
    lowering_steps: how many steps to lower the guest over (recommended: <= 10000)
        (should be <= n_steps)
    start_lambda: what lambda value the guest should start out at (recommended: 0.25)
    random_rotation: whether to apply a random rotation to each guest before beginning the simulation
    constant_atoms: atom numbers from the host_pdbfile to hold mostly fixed across the simulation
        (1-indexed, like PDB files)

    Output
    ------

    A pdb file every 1000 steps (outdir/<guest_name>_<step>.pdb)
    stdout every 1000 steps noting the step number, lambda value, and energy
    """
    assert lowering_steps <= n_steps

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    suppl = Chem.SDMolSupplier(guests_sdfile, removeHs=False)
    for guest_mol in suppl:
        guest_name = guest_mol.GetProp("_Name")
        host_mol = Chem.MolFromPDBFile(host_pdbfile, removeHs=False)
        combined_pdb = Chem.CombineMols(host_mol, guest_mol)

        guest_ff_handlers = deserialize_handlers(
            open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "..",
                    "ff/params/smirnoff_1_1_0_ccc.py",
                )
            ).read()
        )
        amber_ff = app.ForceField("amber99sbildn.xml", "tip3p.xml")
        host_file = PDBFile(host_pdbfile)
        host_system = amber_ff.createSystem(
            host_file.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=None,
            rigidWater=False,
        )

        bps, masses = dock_setup.combine_potentials(
            guest_ff_handlers, guest_mol, host_system, np.float32
        )
        for atom_num in constant_atoms:
            masses[atom_num - 1] += 50000

        host_conf = []
        for x, y, z in host_file.positions:
            host_conf.append([to_md_units(x), to_md_units(y), to_md_units(z)])
        host_conf = np.array(host_conf)
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

        seed = 2020
        intg = LangevinIntegrator(300, 1.5e-3, 1.0, masses, seed).impl()

        box = np.eye(3) * 100
        v0 = np.zeros_like(x0)

        impls = []
        for b in bps:
            p_impl = b.bound_impl()
            impls.append(p_impl)

        ctxt = custom_ops.Context(x0, v0, box, intg, impls)

        new_lambda_schedule = np.concatenate(
            [
                np.linspace(start_lambda, 0.0, lowering_steps),
                np.zeros(n_steps - lowering_steps),
            ]
        )

        for step, lamb in enumerate(new_lambda_schedule):
            ctxt.step(lamb)
            if step % 1000 == 0:
                print("step", step, "lamb", lamb, "nrg", ctxt.get_u_t())
                combined_pdb_str = StringIO(Chem.MolToPDBBlock(combined_pdb))
                pdb_writer = PDBWriter(
                    combined_pdb_str,
                    os.path.join(
                        outdir, f"{guest_name}_{str(step).zfill(len(str(n_steps)))}.pdb"
                    ),
                )
                pdb_writer.write_header()
                pdb_writer.write(ctxt.get_x_t() * 10)
                pdb_writer.close()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Poses guests into a host by running short simulations in which the guests phase in over time",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--guests_sdfile", default="tests/data/ligands_40.sdf", help="guests to pose"
    )
    parser.add_argument(
        "--host_pdbfile",
        default="tests/data/hif2a_nowater_min.pdb",
        help="host to dock into",
    )
    parser.add_argument(
        "--outdir", default="pose_dock_outdir", help="where to write output"
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=20000,
        help="simulation length (1 step = 1.5 femtoseconds",
    )
    parser.add_argument(
        "--lowering_steps",
        type=int,
        default=10000,
        help="how many steps to take while phasing in the guest",
    )
    parser.add_argument(
        "--start_lambda",
        type=float,
        default=0.25,
        help="lambda value to start the guest at",
    )
    parser.add_argument(
        "--random_rotation",
        action="store_true",
        help="apply a random rotation to each guest before docking",
    )
    parser.add_argument(
        "--constant_atoms_file",
        help="file containing comma-separated atom numbers to hold ~fixed",
    )
    args = parser.parse_args()

    constant_atoms = []

    if args.constant_atoms_file:
        with open(args.constant_atoms_file, "r") as rfile:
            for line in rfile.readlines():
                atoms = [int(x.strip()) for x in line.strip().split(",")]
                constant_atoms += atoms

    pose_dock(
        args.guests_sdfile,
        args.host_pdbfile,
        args.outdir,
        args.nsteps,
        args.lowering_steps,
        args.start_lambda,
        random_rotation=args.random_rotation,
        constant_atoms=constant_atoms,
    )
