"""Solvates a host, inserts guest(s) into solvated host, equilibrates
"""
import os
import numpy as np
from rdkit.Chem.rdmolfiles import PDBWriter, SDWriter
from rdkit.Geometry import Point3D


MAX_NORM_FORCE = 20000


def report_step(ctxt, step, lamb, box, bps, u_impls, guest_name, n_steps, stage):
    l_energies = []
    names = []
    for name, impl in zip(bps, u_impls):
        _, _, u = impl.execute(ctxt.get_x_t(), box, lamb)
        l_energies.append(u)
        names.append(name)
        energy = sum(l_energies)

    print(
        f"{stage}\t"
        f"guest_name: {guest_name}\t"
        f"step: {str(step).zfill(len(str(n_steps)))}\t"
        f"lambda: {lamb:.2f}\t"
        f"energy: {energy:.2f}"
    )


def too_much_force(ctxt, lamb, box, bps, u_impls):
    """
    Note (ytz): This function should be called sparingly. It is hideously
    expensive due to all the memcpys. Even once every 100 steps might be too often.
    """
    l_forces = []
    names = []
    for name, impl in zip(bps, u_impls):
        du_dx, _, _ = impl.execute(ctxt.get_x_t(), box, lamb)
        l_forces.append(du_dx)
        names.append(name)
    forces = np.sum(l_forces, axis=0)
    norm_forces = np.linalg.norm(forces, axis=-1)
    if np.any(norm_forces > MAX_NORM_FORCE):
        print("Error: at least one force is too large to continue")
        print("max norm force", np.amax(norm_forces))
        for name, force in zip(names, l_forces):
            print(
                name,
                "atom",
                np.argmax(np.linalg.norm(force, axis=-1)),
                "max norm force",
                np.amax(np.linalg.norm(force, axis=-1)),
            )
        return True
    return False


def write_frame(
        host_coords, host_mol, guest_coords, guest_mol, guest_name, outdir, step, stage
):
    if not os.path.exists(os.path.join(outdir, guest_name)):
        os.mkdir(os.path.join(outdir, guest_name))

    host_frame = host_mol.GetConformer()
    for i in range(host_mol.GetNumAtoms()):
        x, y, z = host_coords[i]
        host_frame.SetAtomPosition(i, Point3D(x, y, z))
    conf_id = host_mol.AddConformer(host_frame)
    writer = PDBWriter(
        os.path.join(outdir, guest_name, f"{guest_name}_{stage}_{step}_host.pdb",)
    )
    writer.write(host_mol, conf_id)
    writer.close()
    host_mol.RemoveConformer(conf_id)

    guest_frame = guest_mol.GetConformer()
    for i in range(guest_mol.GetNumAtoms()):
        x, y, z = guest_coords[i]
        guest_frame.SetAtomPosition(i, Point3D(x, y, z))
    conf_id = guest_mol.AddConformer(guest_frame)
    guest_mol.SetProp("_Name", f"{guest_name}_{stage}_{step}_guest")
    writer = SDWriter(
        os.path.join(outdir, guest_name, f"{guest_name}_{stage}_{step}_guest.sdf",)
    )
    writer.write(guest_mol, conf_id)
    writer.close()
    guest_mol.RemoveConformer(conf_id)
