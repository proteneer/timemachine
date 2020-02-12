import argparse
import time
import numpy as np
from io import StringIO

import rdkit
from rdkit import Chem

from simtk.openmm import app
from simtk.openmm.app import forcefield as ff
from simtk.openmm.app import PDBFile

from timemachine.lib import custom_ops, ops
from fe.utils import to_md_units, write
from system import serialize, forcefield

from openforcefield.typing.engines import smirnoff

def write_coords(frames, pdb_path, romol, outfile, num_frames=100):
    combined_pdb = Chem.CombineMols(Chem.MolFromPDBFile(pdb_path, removeHs=False), mol)
    combined_pdb_str = StringIO(Chem.MolToPDBBlock(combined_pdb))
    cpdb = app.PDBFile(combined_pdb_str)
    PDBFile.writeHeader(cpdb.topology, outfile)

    interval = max(1, frames.shape[0]//num_frames)
    # interval = 1

    # print(interval, frames.shape[0], num_frames)

    for frame_idx, x in enumerate(frames):
        if frame_idx % interval == 0:
            PDBFile.writeModel(cpdb.topology, x*10, outfile, frame_idx)

    PDBFile.writeFooter(pdb.topology, outfile)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Quick Test')
    parser.add_argument('--out_pdb', type=str)
    parser.add_argument('--protein_pdb', type=str)
    parser.add_argument('--ligand_sdf', type=str)
    args = parser.parse_args()

    # parameterize the protein
    pdb = app.PDBFile(args.protein_pdb)
    amber_ff = app.ForceField('amber99sb.xml', 'tip3p.xml')
    system = amber_ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False)

    host_potentials, (host_params, host_param_groups), host_masses = serialize.deserialize_system(system, 3)
    host_conf = []
    for x,y,z in pdb.positions:
        host_conf.append([to_md_units(x),to_md_units(y),to_md_units(z)])
    host_conf = np.array(host_conf, dtype=np.float64)


    # parameterize the small molecule
    suppl = Chem.SDMolSupplier(args.ligand_sdf, removeHs=False)
    for mol in suppl:
        break

    off = smirnoff.ForceField("test_forcefields/smirnoff99Frosst.offxml")
    guest_potentials, (guest_params, guest_param_groups), guest_conf, guest_masses = forcefield.parameterize(mol, off)

    print("Host Shape", host_conf.shape, "Guest Shape", guest_conf.shape)

    combined_potentials, combined_params, combined_param_groups, combined_conf, combined_masses = forcefield.combiner(
        host_potentials, guest_potentials,
        host_params, guest_params,
        host_param_groups, guest_param_groups,
        host_conf, guest_conf,
        host_masses, guest_masses)

    x0 = combined_conf
    v0 = np.zeros_like(x0)

    gradients = []
    for fn, fn_args in combined_potentials:
        gradients.append(fn(*fn_args))

    for g in gradients:
        # if g != ops.HarmonicBond:
            # continue
        du_dx = g.execute(combined_conf, combined_params)
        for atom_idx, xyz in enumerate(np.abs(du_dx)):
            if np.amax(xyz) > 10000:
                print(atom_idx, xyz)

    # assert 0

    stepper = custom_ops.BasicStepper_f64(gradients)

    # should be negative
    cbs = -np.ones(combined_conf.shape[0])*0.001
    dt = 0.002
    T = 10000
    step_sizes = np.ones(T)*dt
    cas = np.ones(T)*0.95


    ctxt = custom_ops.ReversibleContext_f64_3d(
        stepper,
        len(combined_masses),
        x0.reshape(-1).tolist(),
        v0.reshape(-1).tolist(),
        cas.tolist(),
        cbs.tolist(),
        step_sizes.tolist(),
        combined_params.reshape(-1).tolist(),
    )

    out_file = open(args.out_pdb, "w")
    PDBFile.writeHeader(pdb.topology, out_file)

    start = time.time()
    ctxt.forward_mode()
    print("forward time", time.time()-start)

    coords = ctxt.get_all_coords()

    write_coords(coords, args.protein_pdb, mol, out_file)

    test_adjoint = np.random.rand(x0.shape[0], x0.shape[0])/10
    ctxt.set_x_t_adjoint(test_adjoint)
    start = time.time()
    ctxt.backward_mode()
    print("backward time", time.time()-start)

    # compute the parameter derivatives
    dL_dp = ctxt.get_param_adjoint_accum()

    print(dL_dp.shape)