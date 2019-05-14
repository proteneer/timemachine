
import simtk.unit
from simtk.openmm.app import PDBFile

import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

from openforcefield.utils import get_data_file_path
from openforcefield.topology import Molecule, Topology
from openforcefield.typing.engines.smirnoff import ForceField

(1.5*simtk.unit.angstrom).value_in_unit_system(simtk.unit.md_unit_system)


toluene_pdb_file_path = get_data_file_path('molecules/toluene.pdb')
toluene_pdbfile = PDBFile(toluene_pdb_file_path)
toluene = Molecule.from_smiles('Cc1ccccc1')
off_topology = Topology.from_openmm(
    openmm_topology=toluene_pdbfile.topology,
    unique_molecules=[toluene]
)

# Load the smirnoff99Frosst system from disk.
forcefield = ForceField('smirnoff99Frosst.offxml')

def to_md_units(q):
    return q.value_in_unit_system(simtk.unit.md_unit_system)

for b in forcefield.get_parameter_handler('Bonds').parameters:
    print(b.smirks, to_md_units(b.length), to_md_units(b.k))

for b in forcefield.get_parameter_handler('Angles').parameters:
    print(b.smirks, to_md_units(b.angle), to_md_units(b.k))

for bb in forcefield.get_parameter_handler('ProperTorsions').parameters:
    print(bb.smirks)
    for pe, k, ph in zip(bb.periodicity, bb.k, bb.phase):
        print(pe, to_md_units(k), to_md_units(ph))

assert 0

# Parametrize the toluene molecule.
# toluene_system = force_field.create_openmm_system(off_topology)
labels = force_field.label_molecules(off_topology)

print(labels)
for k, v in labels[0]['Bonds'].items():
    print(k, v)