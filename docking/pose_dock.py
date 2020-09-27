import numpy as np

from rdkit import Chem
import simtk.openmm
from simtk.openmm.app import PDBFile
from fe.pdb_writer import PDBWriter

from docking import dock_setup
from ff.handlers.deserialize import deserialize
from timemachine.lib import ops, custom_ops
from io import StringIO

from fe import system

from matplotlib import pyplot as plt

if __name__ == "__main__":

    suppl = Chem.SDMolSupplier("tests/data/ligands_40.sdf", removeHs=False)
    for guest_mol in suppl:
        break

    guest_masses = np.array([a.GetMass() for a in guest_mol.GetAtoms()], dtype=np.float64)

    host_pdbfile = "tests/data/hif2a_nowater_min.pdb"

    host_pdb = PDBFile(host_pdbfile)

    combined_pdb = Chem.CombineMols(
        Chem.MolFromPDBFile(host_pdbfile, removeHs=False),
        guest_mol
    )

    ff_handlers = deserialize(open('ff/params/smirnoff_1_1_0_ccc.py').read())

    x0, combined_masses, final_gradients = dock_setup.create_system(
        guest_mol,
        host_pdb,
        ff_handlers
    )

    gradients = []

    for grad_name, grad_args in final_gradients:
        op_fn = getattr(ops, grad_name)
        grad = op_fn(*grad_args, precision=np.float32)
        gradients.append(grad)

    print(gradients)

    n_steps = 20000

    integrator = system.Integrator(
        steps=n_steps,
        dt=1.5e-3,
        temperature=300.0,
        friction=50,
        masses=combined_masses,
        lamb=np.zeros(n_steps),
        seed=42
    )

    lowering_steps = 10000

    new_lambda_schedule = np.concatenate([
        np.linspace(1.0, 0.0, lowering_steps),
        np.zeros(n_steps - lowering_steps)
    ])


    stepper = custom_ops.AlchemicalStepper_f64(
        gradients,
        new_lambda_schedule
        # integrator.lambs
    )

    v0 = np.zeros_like(x0)

    ctxt = custom_ops.ReversibleContext_f64(
        stepper,
        x0,
        v0,
        integrator.cas,
        integrator.cbs,
        integrator.ccs,
        integrator.dts,
        integrator.seed
    )

    combined_pdb_str = StringIO(Chem.MolToPDBBlock(combined_pdb))
    out_file = "pose_dock.pdb"

    pdb_writer = PDBWriter(combined_pdb_str, out_file)

    # pdb_writer.write_header()

    # frames = ctxt.get_all_coords()
    # for frame_idx, x in enumerate(frames):
    #     # if frame_idx % 100 == 0:
    #     pdb_writer.write(x*10)
    #     break
    # pdb_writer.close()

    # assert 0

    ctxt.forward_mode()

    energies = stepper.get_energies()

    print(energies)

    frames = ctxt.get_all_coords()

    pdb_writer.write_header()
    for frame_idx, x in enumerate(frames):
        if frame_idx % 100 == 0:
            pdb_writer.write(x*10)
    pdb_writer.close()

    plt.plot(energies)
    plt.show()