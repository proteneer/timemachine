from importlib import resources

import numpy as np
from openmm import app

from timemachine.ff.handlers import openmm_deserializer
from timemachine.md.builders import strip_units


def setup_dhfr():
    with resources.path("timemachine.testsystems.data", "5dfr_solv_equil.pdb") as pdb_path:
        host_pdb = app.PDBFile(str(pdb_path))

    protein_ff = app.ForceField("amber99sbildn.xml", "tip3p.xml")
    host_system = protein_ff.createSystem(
        host_pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False
    )
    host_coords = strip_units(host_pdb.positions)
    box = host_pdb.topology.getPeriodicBoxVectors()
    box = strip_units(box)

    host_fns, host_masses = openmm_deserializer.deserialize_system(host_system, cutoff=1.0)

    return host_fns, host_masses, np.array(host_coords), np.array(box)


def get_dhfr_system():
    with resources.path("timemachine.testsystems.data", "5dfr_solv_equil.pdb") as pdb_path:
        host_pdb = app.PDBFile(str(pdb_path))

    modeller = app.Modeller(host_pdb.topology, host_pdb.positions)
    protein_ff = app.ForceField("amber99sbildn.xml", "tip3p.xml")
    host_system = protein_ff.createSystem(
        host_pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False
    )
    return host_system, modeller.topology
