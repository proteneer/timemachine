from tempfile import NamedTemporaryFile

import numpy as np
from simtk import unit
from simtk.openmm import app, Vec3

from openeye.oechem import (
    oemolistream,
    oemolostream,
    OEGraphMol,
    OEReadMolecule,
    OEWriteMolecule,
    OEAltLocationFactory,
)
from openeye.oespruce import (
    OEStructureMetadata,
    OEMakeDesignUnitOptions,
    OEMakeBioDesignUnits,
)

from ff.handlers.bcc_aromaticity import openeye_log_wrapper


def prepare_protein(path: str, output_path: str):
    """
    Prepares protein

    Performs the following preparation:
    * Add missing hydrogens
    * Remove everything except for the protein

    Parameters
    ----------

    path: string
        Path to a PDB/CIF file that contains an unprepared protein

    output_path: string
        Path to write the prepared protein to

    Relevant Docs:
        https://docs.eyesopen.com/toolkits/python/sprucetk/OESpruceFunctions/OEMakeBioDesignUnits.html
        https://docs.eyesopen.com/toolkits/python/sprucetk/OESpruceClasses/OEMakeDesignUnitOptions.html
    """
    with openeye_log_wrapper() as log_stream:
        ifs = oemolistream()
        if not ifs.open(path):
            raise IOError(f"Unable to open {path}")
        temp_mol = OEGraphMol()
        if not OEReadMolecule(ifs, temp_mol):
            raise IOError(f"Unable to read mol from {path}")
        ifs.close()
        # Its GraphMols all the time with Spruce
        mol = OEGraphMol()
        fact = OEAltLocationFactory(temp_mol)
        fact.MakePrimaryAltMol(mol)
        output_mol = OEGraphMol()
        opts = OEMakeDesignUnitOptions()
        opts.GetPrepOptions().SetProtonate(False)
        build_opts = opts.GetPrepOptions().GetBuildOptions()
        build_opts.SetBuildSidechains(False)
        build_opts.SetBuildLoops(False)
        build_opts.SetCapCTermini(False)
        build_opts.SetCapNTermini(False)
        for i, design_unit in enumerate(OEMakeBioDesignUnits(mol, OEStructureMetadata(), opts)):
            if i > 0:
                raise AssertionError("Got more than one BioUnit")
            design_unit.GetProtein(output_mol)
            with oemolostream(output_path) as ofs:
                OEWriteMolecule(ofs, output_mol)


def strip_units(coords):
    return unit.Quantity(np.array(coords / coords.unit), coords.unit)

def build_protein_system(host_pdbfile: str, prepare: bool = True):
    host_ff = app.ForceField('amber99sbildn.xml', 'tip3p.xml')
    if not prepare:
        host_pdb = app.PDBFile(host_pdbfile)
    else:
        with NamedTemporaryFile(suffix=".pdb") as temp:
            prepare_protein(host_pdbfile, temp.name)
            host_pdb = app.PDBFile(temp.name)

    modeller = app.Modeller(host_pdb.topology, host_pdb.positions)
    host_coords = strip_units(host_pdb.positions)

    padding = 1.0
    box_lengths = np.amax(host_coords, axis=0) - np.amin(host_coords, axis=0)
    box_lengths = box_lengths.value_in_unit_system(unit.md_unit_system)

    box_lengths = box_lengths+padding
    box = np.eye(3, dtype=np.float64)*box_lengths

    modeller.addSolvent(host_ff, boxSize=np.diag(box)*unit.nanometers, neutralize=False)
    solvated_host_coords = strip_units(modeller.positions)

    nha = host_coords.shape[0]
    nwa = solvated_host_coords.shape[0] - nha

    print("building a protein system with", nha, "protein atoms and", nwa, "water atoms")
    solvated_host_system = host_ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False
    )

    return solvated_host_system, solvated_host_coords, nwa, nha, box, modeller.topology

def build_water_system(box_width):
    ff = app.ForceField('tip3p.xml')

    # Create empty topology and coordinates.
    top = app.Topology()
    pos = unit.Quantity((), unit.angstroms)
    m = app.Modeller(top, pos)

    boxSize = Vec3(box_width, box_width, box_width)*unit.nanometers
    m.addSolvent(ff, boxSize=boxSize, model='tip3p')

    system = ff.createSystem(
        m.getTopology(),
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False
    )

    positions = m.getPositions()
    positions = unit.Quantity(np.array(positions / positions.unit), positions.unit)

    assert m.getTopology().getNumAtoms() == positions.shape[0]

    # TODO: minimize the water box (BFGS or scipy.optimize)
    return system, positions, np.eye(3)*box_width, m.getTopology()
