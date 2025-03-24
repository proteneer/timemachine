import os

import numpy as np
from numpy.typing import NDArray
from openmm import app, unit
from rdkit import Chem

from timemachine.fe.free_energy import HostConfig
from timemachine.fe.system import HostSystem
from timemachine.fe.utils import get_romol_conf
from timemachine.ff import sanitize_water_ff
from timemachine.ff.handlers import openmm_deserializer
from timemachine.potentials.jax_utils import idxs_within_cutoff

WATER_RESIDUE_NAME = "HOH"
SODIUM_ION_RESIDUE = "NA"
CHLORINE_ION_RESIDUE = "CL"


def strip_units(coords) -> NDArray[np.float64]:
    return np.array(coords.value_in_unit_system(unit.md_unit_system))


def get_box_from_coords(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    box_lengths = np.max(coords, axis=0) - np.min(coords, axis=0)
    return np.eye(3) * box_lengths


def get_ion_residue_templates(modeller) -> dict[app.Residue, str]:
    """When using amber99sbildn with the amber14 water models the ion templates get duplicated. This forces
    the use of the NA/CL templates (from the amber14 water models) rather than the amber99sbildn template Na+ and CL-
    """
    residue_templates = {}
    residue_templates.update(
        {res: SODIUM_ION_RESIDUE for res in modeller.getTopology().residues() if res.name == SODIUM_ION_RESIDUE}
    )
    residue_templates.update(
        {res: CHLORINE_ION_RESIDUE for res in modeller.getTopology().residues() if res.name == CHLORINE_ION_RESIDUE}
    )
    return residue_templates


def replace_clashy_waters(
    modeller: app.Modeller,
    host_coords: NDArray[np.float64],
    box: NDArray[np.float64],
    water_idxs: NDArray[np.int_],
    mols: list[Chem.Mol],
    host_ff: app.ForceField,
    water_ff: str,
    clash_distance: float = 0.4,
):
    """Replace waters that clash with a set of molecules with waters at the boundaries rather than
    clashing with the molecules. The number of atoms in the system will be identical before and after

    Parameters
    ----------
    modeller: app.Modeller
        Modeller to update in place

    host_coords: NDArray[np.float64]
        Coordinates of host, may be different than modeller.positions

    box: NDArray[np.float64]
        Box to evaluate PBCs under

    water_idxs: NDArray[int]
        The indices of all of the water atoms in the system.

    mols: list[Mol]
        List of molecules to determine which waters are clashy

    host_ff: app.ForceField
        The forcefield used for the host

    water_ff: str
        The water forcefield name (excluding .xml) to parametrize the water with.

    clash_distance: float
        Distance from a ligand atom to a water atom to consider as a clash, in nanometers
    """
    water_coords = host_coords[water_idxs]
    ligand_coords = np.concatenate([get_romol_conf(mol) for mol in mols])
    clashy_idxs = idxs_within_cutoff(water_coords, ligand_coords, box, cutoff=clash_distance)
    if len(clashy_idxs) == 0:
        return
    # Offset the clashy idxs with the first atom idx, else could be pointing at non-water atoms
    clashy_idxs += np.min(water_idxs)

    def get_waters_to_delete():
        all_atoms = list(modeller.topology.atoms())
        waters_to_delete = set()
        for idx in clashy_idxs:
            atom = all_atoms[idx]
            if atom.residue.name == WATER_RESIDUE_NAME:
                waters_to_delete.add(atom.residue)
        return waters_to_delete

    # First add back in the number of waters that are clashy. Then delete the clashy waters.
    # Done in this order so that additional waters being added will be at the boundaries, if added after deleting
    # addSolvent fills the void intended for the mols. Need to end up with the same number of waters as originally
    num_system_atoms = host_coords.shape[0]
    clashy_waters = get_waters_to_delete()
    combined_templates = get_ion_residue_templates(modeller)
    # First add back in the number of waters that are clashy and we know we need to delete
    modeller.addSolvent(
        host_ff,
        numAdded=len(clashy_waters),
        neutralize=False,
        model=sanitize_water_ff(water_ff),
        residueTemplates=combined_templates,
    )
    clashy_waters = get_waters_to_delete()
    modeller.delete(list(clashy_waters))
    assert num_system_atoms == modeller.getTopology().getNumAtoms()


def solvate_modeller(
    modeller: app.Modeller,
    box: NDArray[np.float64],
    ff: app.ForceField,
    water_ff: str,
    mols: list[Chem.Mol] | None,
    ionic_concentration: float = 0.0,
    neutralize: bool = False,
):
    """Utility function to handle the neutralization of adding mols"""
    dummy_chain_id = "DUMMY"
    if neutralize and mols is not None:
        # Since we do not add the ligands to the OpenMM system, we add ions that emulate the net charge
        # of the ligands so that `addSolvent` will neutralize the system correctly.
        charges = [Chem.GetFormalCharge(mol) for mol in mols]
        # If the charges are not the same for all mols, unable to neutralize the system effectively
        # TBD: Decide if we want to weaken this assertion, most likely for charge hopping
        assert all([charge == charges[0] for charge in charges])
        charge = charges[0]
        if charge != 0:
            topology = app.Topology()
            dummy_chain = topology.addChain(dummy_chain_id)
            for _ in range(abs(charge)):
                if charge < 0:
                    res = topology.addResidue(CHLORINE_ION_RESIDUE, dummy_chain)
                    topology.addAtom(CHLORINE_ION_RESIDUE, app.Element.getBySymbol("Cl"), res)
                else:
                    res = topology.addResidue(SODIUM_ION_RESIDUE, dummy_chain)
                    topology.addAtom(SODIUM_ION_RESIDUE, app.Element.getBySymbol("Na"), res)
            coords = np.zeros((topology.getNumAtoms(), 3)) * unit.angstroms
            modeller.add(topology, coords)
    combined_templates = get_ion_residue_templates(modeller)
    modeller.addSolvent(
        ff,
        boxSize=np.diag(box) * unit.nanometers,
        ionicStrength=ionic_concentration * unit.molar,
        model=sanitize_water_ff(water_ff),
        neutralize=neutralize,
        residueTemplates=combined_templates,
    )
    if neutralize and mols is not None:
        current_topo = modeller.getTopology()
        # Remove the chain filled with the dummy ions
        bad_chains = [chain for chain in current_topo.chains() if chain.id == dummy_chain_id]
        modeller.delete(bad_chains)


def build_protein_system(
    host_pdbfile: app.PDBFile | str,
    protein_ff: str,
    water_ff: str,
    mols: list[Chem.Mol] | None = None,
    ionic_concentration: float = 0.0,
    neutralize: bool = False,
) -> HostConfig:
    """
    Build a solvated protein system with a 10A padding.

    Parameters
    ----------
    host_pdbfile: str or app.PDBFile
        PDB of the host structure

    protein_ff: str
        The protein forcefield name (excluding .xml) to parametrize the host_pdbfile with.

    water_ff: str
        The water forcefield name (excluding .xml) to parametrize the water with.

    mols: optional list of mols
        Molecules to be part of the system, will avoid placing water molecules that clash with the mols.
        If water molecules provided in the PDB clash with the mols, will do nothing.

    ionic_concentration: optional float
        Concentration of ions, in molars, to add to the system. Defaults to 0.0, meaning no ions are added.

    neutralize: optional bool
        Whether or not to add ions to the system to ensure the system has a net charge of 0.0. Defaults to False.

    Returns
    -------
    HostConfig
    """

    host_ff = app.ForceField(f"{protein_ff}.xml", f"{water_ff}.xml")
    if isinstance(host_pdbfile, str):
        assert os.path.exists(host_pdbfile)
        host_pdb = app.PDBFile(host_pdbfile)
    elif isinstance(host_pdbfile, app.PDBFile):
        host_pdb = host_pdbfile
    else:
        raise TypeError("host_pdbfile must be a string or an openmm PDBFile object")

    modeller = app.Modeller(host_pdb.topology, host_pdb.positions)
    host_coords = strip_units(host_pdb.positions)

    water_residues_in_pdb = [residue for residue in host_pdb.topology.residues() if residue.name == WATER_RESIDUE_NAME]
    num_host_atoms = host_coords.shape[0]
    if len(water_residues_in_pdb) > 0:
        host_water_atoms = len(water_residues_in_pdb) * 3
        # Only consider non-water atoms as the host, does count excipients as the host
        num_host_atoms = num_host_atoms - host_water_atoms
        water_indices = np.concatenate([[a.index for a in res.atoms()] for res in water_residues_in_pdb])
        expected_water_indices = np.arange(host_water_atoms) + num_host_atoms
        np.testing.assert_equal(
            water_indices, expected_water_indices, err_msg="Waters in PDB must be at the end of the file"
        )

    padding = 1.0
    box = get_box_from_coords(host_coords)
    box += padding

    solvate_modeller(
        modeller, box, host_ff, water_ff, mols=mols, neutralize=neutralize, ionic_concentration=ionic_concentration
    )
    solvated_host_coords = strip_units(modeller.positions)

    if mols is not None:
        # Only look at waters that we have added, ignored the waters provided in the PDB
        water_idxs = np.arange(host_coords.shape[0], solvated_host_coords.shape[0])
        replace_clashy_waters(modeller, solvated_host_coords, box, water_idxs, mols, host_ff, water_ff)
        solvated_host_coords = strip_units(modeller.positions)

    num_water_atoms = solvated_host_coords.shape[0] - num_host_atoms

    assert modeller.getTopology().getNumAtoms() == solvated_host_coords.shape[0]

    print("building a protein system with", num_host_atoms, "protein atoms and", num_water_atoms, "water atoms")
    combined_templates = get_ion_residue_templates(modeller)

    solvated_omm_host_system = host_ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False,
        residueTemplates=combined_templates,
    )

    (bond, angle, proper, improper, nonbonded), masses = openmm_deserializer.deserialize_system(
        solvated_omm_host_system, cutoff=1.2
    )

    solvated_host_system = HostSystem(
        bond=bond,
        angle=angle,
        proper=proper,
        improper=improper,
        nonbonded_all_pairs=nonbonded,
    )

    # Determine box from the system's coordinates
    box = get_box_from_coords(solvated_host_coords)

    assert len(list(modeller.topology.atoms())) == len(solvated_host_coords)

    return HostConfig(
        host_system=solvated_host_system,
        conf=solvated_host_coords,
        box=box,
        num_water_atoms=num_water_atoms,
        omm_topology=modeller.topology,
        masses=masses,
    )


def build_water_system(
    box_width: float,
    water_ff: str,
    mols: list[Chem.Mol] | None = None,
    ionic_concentration: float = 0.0,
    neutralize: bool = False,
) -> HostConfig:
    """
    Build a water system with a cubic box with each side of length box_width.

    Parameters
    ---------
    box_width: float
        The length of each size of the box

    water_ff: str
        The water forcefield name (excluding .xml) to parametrize the water with.

    mols: optional list of mols
        Molecules to be part of the system, will remove water molecules that clash with the mols.

    ionic_concentration: optional float
        Molar concentration of ions to add to the system. Defaults to 0.0, meaning no ions are added.

    neutralize: optional bool
        Whether or not to add ions to the system to ensure the system has a net charge of 0.0. Defaults to False.

    Returns
    -------
    4-Tuple
        OpenMM host system, coordinates, box, OpenMM topology
    """
    if ionic_concentration < 0.0:
        raise ValueError("Ionic concentration must be greater than or equal to 0.0")
    ff = app.ForceField(f"{water_ff}.xml")

    # Create empty topology and coordinates.
    top = app.Topology()
    pos = unit.Quantity((), unit.angstroms)
    modeller = app.Modeller(top, pos)

    box = np.eye(3) * box_width

    solvate_modeller(
        modeller, box, ff, water_ff, mols=mols, neutralize=neutralize, ionic_concentration=ionic_concentration
    )

    def get_host_coords():
        host_coords = strip_units(modeller.positions)
        # If mols provided, center waters such that the center is the mols centroid
        # Done to avoid placing mols at the edges and moves the water coordinates to avoid
        # changing the mol coordinates which are finalized downstream of the builder
        if mols is not None and len(mols) > 0:
            mol_coords = np.concatenate([get_romol_conf(mol) for mol in mols])
            mols_centroid = np.mean(mol_coords, axis=0)
            box_extents = (np.max(host_coords, axis=0) - np.min(host_coords, axis=0)) * 0.5
            box_center = np.min(host_coords, axis=0) + box_extents
            host_coords = host_coords - box_center + mols_centroid
        return host_coords

    if mols is not None:
        solvated_host_coords = get_host_coords()

        water_idxs = np.arange(solvated_host_coords.shape[0])
        replace_clashy_waters(modeller, solvated_host_coords, box.astype(np.float64), water_idxs, mols, ff, water_ff)

    solvated_host_coords = get_host_coords()

    assert modeller.getTopology().getNumAtoms() == solvated_host_coords.shape[0]

    omm_host_system = ff.createSystem(
        modeller.getTopology(), nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False
    )
    # Determine box from the system's coordinates
    box = get_box_from_coords(solvated_host_coords)

    (bond, angle, proper, improper, nonbonded), masses = openmm_deserializer.deserialize_system(
        omm_host_system, cutoff=1.2
    )

    solvated_host_system = HostSystem(
        bond=bond,
        angle=angle,
        proper=proper,
        improper=improper,
        nonbonded_all_pairs=nonbonded,
    )

    # Determine box from the system's coordinates
    box = get_box_from_coords(solvated_host_coords)
    num_water_atoms = len(solvated_host_coords)

    assert len(list(modeller.topology.atoms())) == len(solvated_host_coords)

    return HostConfig(
        host_system=solvated_host_system,
        conf=solvated_host_coords,
        box=box,
        num_water_atoms=num_water_atoms,
        omm_topology=modeller.topology,
        masses=masses,
    )
