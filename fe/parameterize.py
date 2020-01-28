import argparse

import numpy as np

from simtk.openmm import app
from simtk.openmm.app import forcefield as ff
from simtk.openmm.app import PDBFile

from timemachine.lib import custom_ops, ops
from fe.utils import to_md_units, write
from system import serialize_v2

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Quick Test')
    parser.add_argument('--protein_pdb', type=str)
    args = parser.parse_args()
    pdb = app.PDBFile(args.protein_pdb)
    amber_ff = app.ForceField('amber99sb.xml', 'tip3p.xml')
    system = amber_ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False)

    host_potentials, (host_params, host_param_groups), host_masses = serialize_v2.deserialize_system(system, 3)


    coords = []
    for x,y,z in pdb.positions:
        coords.append([to_md_units(x),to_md_units(y),to_md_units(z)])
    coords = np.array(coords, dtype=np.float64)

    x0 = coords
    v0 = np.zeros_like(x0)

    gradients = []
    for fn, args in host_potentials:
        gradients.append(fn(*args))

    stepper = custom_ops.BasicStepper_f64(gradients)


    cbs = np.ones(coords.shape[0])*0.001
    dt = 0.002
    T = 100
    step_sizes = np.ones(T)*dt
    cas = np.ones(T)*0.95

    ctxt = custom_ops.ReversibleContext_f64_3d(
        stepper,
        len(host_masses),
        x0.reshape(-1).tolist(),
        v0.reshape(-1).tolist(),
        cas.tolist(),
        cbs.tolist(),
        step_sizes.tolist(),
        host_params.reshape(-1).tolist(),
    )

    print(ctxt)
    ctxt.forward_mode()

    ctxt.backward_mode()
    # print(test_potentials)
    # smirnoff = ForceField("test_forcefields/smirnoff99Frosst.offxml")