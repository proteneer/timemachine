import os

import numpy as np
from openmm import app

from timemachine.constants import AVOGADRO, DEFAULT_PRESSURE, DEFAULT_TEMP
from timemachine.fe.free_energy import AbsoluteFreeEnergy, HostConfig, InitialState
from timemachine.fe.topology import BaseTopology
from timemachine.fe.utils import get_romol_conf
from timemachine.ff import Forcefield
from timemachine.ff.handlers.nonbonded import PrecomputedChargeHandler
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.md.builders import strip_units
from timemachine.md.exchange.exchange_mover import delta_r_np
from timemachine.potentials import HarmonicBond

DEFAULT_BB_RADIUS = 0.46
# uncomment if we want to re-enable minimization
# from timemachine.md.minimizer import minimize_host_4d


def build_system(host_pdbfile: str, water_ff: str, padding: float):
    if isinstance(host_pdbfile, str):
        assert os.path.exists(host_pdbfile)
        host_pdb = app.PDBFile(host_pdbfile)
    elif isinstance(host_pdbfile, app.PDBFile):
        host_pdb = host_pdbfile
    else:
        raise TypeError("host_pdbfile must be a string or an openmm PDBFile object")

    host_coords = strip_units(host_pdb.positions)

    box_lengths = np.amax(host_coords, axis=0) - np.amin(host_coords, axis=0)

    box_lengths = box_lengths + padding
    box = np.eye(3, dtype=np.float64) * box_lengths

    host_ff = app.ForceField(f"{water_ff}.xml")
    nwa = len(host_coords)
    solvated_host_system = host_ff.createSystem(
        host_pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False
    )
    solvated_host_coords = host_coords
    solvated_topology = host_pdb.topology

    return solvated_host_system, solvated_host_coords, box, solvated_topology, nwa


def setup_forcefield():
    # use a precomputed charge handler on the ligand to avoid running AM1 on a buckyball
    ff = Forcefield.load_default()
    q_handle = PrecomputedChargeHandler()
    q_handle_intra = PrecomputedChargeHandler()
    q_handle_solv = PrecomputedChargeHandler()
    return Forcefield(
        ff.hb_handle,
        ff.ha_handle,
        ff.pt_handle,
        ff.it_handle,
        q_handle=q_handle,
        q_handle_solv=q_handle_solv,
        q_handle_intra=q_handle_intra,
        lj_handle=ff.lj_handle,
        lj_handle_solv=ff.lj_handle_solv,
        lj_handle_intra=ff.lj_handle_intra,
        protein_ff=ff.protein_ff,
        water_ff=ff.water_ff,
    )


def compute_density(n_waters, box):
    # returns density of water in kg/m^3
    box_vol = np.prod(np.diag(box))
    numerator = n_waters * 18.01528 * 1e27
    denominator = box_vol * AVOGADRO * 1000
    return numerator / denominator


def compute_occupancy(x_t, box_t, ligand_idxs, threshold):
    # compute the number of waters inside the buckyball ligand
    num_ligand_atoms = len(ligand_idxs)
    num_host_atoms = len(x_t) - num_ligand_atoms
    host_coords = x_t[:num_host_atoms]
    ligand_coords = x_t[num_host_atoms:]
    ligand_centroid = np.mean(ligand_coords, axis=0)
    diffs = delta_r_np(ligand_centroid[None, :], host_coords, box_t)
    dijs = np.linalg.norm(diffs, axis=-1)
    count = np.sum(dijs < threshold)
    return count


def get_initial_state(water_pdb, mol, ff, seed, nb_cutoff, lamb):
    assert nb_cutoff == 1.2  # hardcoded in prepare_host_edge

    # read water system
    solvent_sys, solvent_conf, solvent_box, solvent_topology, num_water_atoms = build_system(
        water_pdb, ff.water_ff, padding=0.1
    )
    solvent_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes

    assert num_water_atoms == len(solvent_conf)

    bt = BaseTopology(mol, ff)
    afe = AbsoluteFreeEnergy(mol, bt)
    host_config = HostConfig(solvent_sys, solvent_conf, solvent_box, num_water_atoms)
    potentials, params, combined_masses = afe.prepare_host_edge(ff.get_params(), host_config, lamb)
    ligand_idxs = np.arange(num_water_atoms, num_water_atoms + mol.GetNumAtoms())
    nb_params = np.array(params[-1])

    final_masses = combined_masses
    final_conf = np.concatenate([solvent_conf, get_romol_conf(mol)], axis=0)

    component_dim = 4  # q,s,e,w
    num_atoms = len(final_conf)

    ligand_water_flat_idxs = np.s_[component_dim * num_atoms : 2 * component_dim * num_atoms]  # water
    ligand_water_params = nb_params[ligand_water_flat_idxs].reshape(-1, 4)
    # Forms a hole on the "flat" side of the buckyball, 1-indexed, over the selected atoms
    # previously failed attempts:
    # 1) uniformly decoupling half of the buckyball
    # 2) uniformly decoupling entire buckyball
    # 3) uniformly shrinking vdw radii of every buckyball atom
    # 4) uniformly shrinking vdw radii of half of the buckyball atoms
    lambda_coupled_ligand_atoms = num_water_atoms + np.array([10, 11, 16, 17, 21, 22, 12, 14, 13, 25, 24, 23, 20]) - 1
    fully_coupled_ligand_atoms = np.setdiff1d(ligand_idxs, lambda_coupled_ligand_atoms)

    ligand_water_params[fully_coupled_ligand_atoms, -1] = 0
    nb_params[ligand_water_flat_idxs] = ligand_water_params.reshape(-1)

    # override last potential with the updated nb_params
    host_bps = []
    for p, ubp in zip(params[:-1], potentials[:-1]):
        host_bps.append(ubp.bind(p))

    host_bps.append(potentials[-1].bind(nb_params))

    # (YTZ): This is disabled because the initial ligand and starting waters are pre-minimized
    # print("Minimizing the host system...", end="", flush=True)
    # host_conf = minimize_host_4d([mol], host_config, ff)
    # print("Done", flush=True)

    temperature = DEFAULT_TEMP
    dt = 1.0e-3
    barostat_interval = 25
    integrator = LangevinIntegrator(temperature, dt, 1.0, final_masses, seed)
    bond_list = get_bond_list(next(bp for bp in host_bps if isinstance(bp.potential, HarmonicBond)).potential)
    group_idxs = get_group_indices(bond_list, len(final_masses))
    barostat = MonteCarloBarostat(
        len(final_masses), DEFAULT_PRESSURE, temperature, group_idxs, barostat_interval, seed + 1
    )

    initial_state = InitialState(
        host_bps, integrator, barostat, final_conf, np.zeros_like(final_conf), solvent_box, lamb, ligand_idxs
    )

    assert num_water_atoms % 3 == 0
    num_water_mols = num_water_atoms // 3

    return initial_state, num_water_mols, solvent_topology
