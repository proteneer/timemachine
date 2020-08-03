import unittest

from simtk.openmm import app
from ff.handlers import openmm_deserializer

class TestOpenMMDeserializer:

    def test_from_pdb(self):

        host_pdb_file = "tests/hif2a_nowater_min.pdb"

        host_pdb = app.PDBFile(host_pdb_file)
        amber_ff = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')

        host_system = amber_ff.createSystem(
            host_pdb.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=None,
            rigidWater=False
        )
        
        nrg_fns, masses = openmm_deserializer.deserialize_system(host_system)

        assert len(nrg_fns.keys()) == 6
