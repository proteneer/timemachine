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


# from timemachine.lib import LangevinIntegrator


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

    ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_ccc.py').read())

    amber_ff = app.ForceField('amber99sbildn.xml', 'tip3p.xml')
    host_system = amber_ff.createSystem(
        host_pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False
    )

    bps, masses = dock_setup.combine_potentials(
        ff_handlers,
        guest_mol,
        host_system,
        np.float32
    )


    host_conf = []
    for x,y,z in host_pdb.positions:
        host_conf.append([to_md_units(x),to_md_units(y),to_md_units(z)])
    host_conf = np.array(host_conf)
    conformer = guest_mol.GetConformer(0)
    mol_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    mol_conf = mol_conf/10 # convert to md_units

    # for applying a random rotation
    # center = np.mean(mol_conf, axis=0)
    # mol_conf -= center
    # from scipy.stats import special_ortho_group
    # mol_conf = np.matmul(mol_conf, special_ortho_group.rvs(3))
    # mol_conf += center

    x0 = np.concatenate([host_conf, mol_conf]) # combined geometry
    v0 = np.zeros_like(x0)

    # print(potentia)

    n_steps = 20000

    # integrator = system.Integrator(
    #     steps=n_steps,
    #     dt=1.5e-3,
    #     temperature=300.0,
    #     friction=50,
    #     masses=combined_masses,
    #     lamb=np.zeros(n_steps),
    #     seed=42
    # )
    seed = 2020
    intg = LangevinIntegrator(
        300,
        1.5e-3,
        1.0,
        masses,
        seed
    ).impl()

    lowering_steps = 10000

    new_lambda_schedule = np.concatenate([
        np.linspace(1.0, 0.0, lowering_steps),
        np.zeros(n_steps - lowering_steps)
    ])


    # stepper = custom_ops.AlchemicalStepper_f64(
    #     gradients,
    #     new_lambda_schedule
    #     # integrator.lambs
    # )

    box = np.eye(3) * 100

    v0 = np.zeros_like(x0)


    impls = []
    for b in bps:
        p_impl = b.bound_impl()
        impls.append(p_impl)


    ctxt = custom_ops.Context(
        x0,
        v0,
        box,
        intg,
        impls
    )

    combined_pdb_str = StringIO(Chem.MolToPDBBlock(combined_pdb))
    out_file = "pose_dock.pdb"

    pdb_writer = PDBWriter(combined_pdb_str, out_file)

    pdb_writer.write_header()
    for step, lamb in enumerate(np.linspace(1.0, 0.0, 1000)):
        ctxt.step(lamb)
        if step % 100 == 0:
            print("step", step, "nrg", ctxt.get_u_t())
            pdb_writer.write(ctxt.get_x_t()*10)
    pdb_writer.close()
