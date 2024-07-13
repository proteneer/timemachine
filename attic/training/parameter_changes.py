def get_solvent_phase_system_parameter_changes(mol, ff0, ff1, box_width=3.0, margin=0.5):
    """
    Given a mol and a pair of forcefields return a solvated system.
    The system is set up to determine the relative free energy of
    changing the forcefield parameters.

    Parameters
    ----------
    mol: Chem.Mol

    ff0: Forcefield
        Effective forcefield at lambda = 0.

    ff1: Forcefield
        Effective forcefield at lambda = 1.

    box_width: float
        water box initial width in nm

    margin: Optional, float
        Box margin in nm, default is 0.5 nm.

    minimize_energy: bool
        whether to apply minimize_host_4d
    """

    # construct water box
    if ff0.water_ff != ff1.water_ff:
        raise RuntimeError(f"Can not alchemically change the water model: {ff0.water_ff} != {ff1.water_ff}")
    water_system, water_coords, water_box, water_topology = builders.build_water_system(box_width, ff0.water_ff)

    top = topology.RelativeFreeEnergyForcefield(mol, ff0, ff1)
    afe = free_energy.AbsoluteFreeEnergy(mol, top)
    combined_ff_params = combine_params(ff0.get_params(), ff1.get_params())
    potentials, params, masses = afe.prepare_host_edge(combined_ff_params, water_system)

    # concatenate water_coords and ligand_coords
    ligand_coords = get_romol_conf(mol)
    coords = np.concatenate([water_coords, ligand_coords])

    return potentials, params, np.array(masses), coords, water_box


def get_vacuum_phase_system_parameter_changes(mol, ff0, ff1):
    """
    Given a mol and a pair of forcefields return a vacuum system set up
    to determine the free energy of changing the forcefield params.

    Parameters
    ----------
    mol: Chem.Mol

    ff0: Forcefield
        Effective forcefield at lambda = 0.

    ff1: Forcefield
        Effective forcefield at lambda = 1.
    """
    top = topology.RelativeFreeEnergyForcefield(mol, ff0, ff1)
    afe = free_energy.AbsoluteFreeEnergy(mol, top)
    combined_ff_params = combine_params(ff0.get_params(), ff1.get_params())
    potentials, params, masses = afe.prepare_vacuum_edge(combined_ff_params)
    coords = get_romol_conf(mol)
    return potentials, params, np.array(masses), coords
