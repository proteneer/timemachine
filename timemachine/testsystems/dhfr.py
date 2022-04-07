import importlib_resources as resources
import numpy as np
from simtk.openmm import app

from timemachine.ff.handlers import openmm_deserializer


def setup_dhfr():
    with resources.path("timemachine.testsystems.data", "5dfr_solv_equil.pdb") as pdb_path:
        host_pdb = app.PDBFile(str(pdb_path))

    protein_ff = app.ForceField("amber99sbildn.xml", "tip3p.xml")
    host_system = protein_ff.createSystem(
        host_pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False
    )
    host_coords = host_pdb.positions
    box = host_pdb.topology.getPeriodicBoxVectors()
    box = np.asarray(box / box.unit)

    host_fns, host_masses = openmm_deserializer.deserialize_system(host_system, cutoff=1.0)

    return host_fns, host_masses, host_coords, box
