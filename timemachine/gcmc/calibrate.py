# This script determines two parameters used in the Adams formulation
# of the Grand Canonical Monte Carlo move (GCMC).
# 1) the chemical potential
# 2) the standard volume of water
# the latter is computed using a staged AHFE calculation
# the former is computed using the end-state corresponding to a fully interacting water

import functools

import numpy as np

from timemachine.constants import DEFAULT_PRESSURE, DEFAULT_TEMP
from timemachine.fe.free_energy import InitialState, MDParams, run_sims_with_greedy_bisection
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.gcmc.mover_ref import compute_density
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat
from timemachine.md import builders
from timemachine.md.barostat.utils import get_bond_list, get_group_indices


def _get_initial_state(lamb, water_ff, box_width, seed, nb_cutoff):
    solvent_sys, solvent_conf, solvent_box, _ = builders.build_water_system(box_width, water_ff)

    solvent_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
    host_bps, host_masses = openmm_deserializer.deserialize_system(solvent_sys, cutoff=nb_cutoff)

    w = lamb * nb_cutoff

    n_atoms = len(solvent_conf)
    ligand_idxs = [n_atoms - 3, n_atoms - 2, n_atoms - 1]
    host_bps[-1].params[ligand_idxs, -1] = w
    temperature = DEFAULT_TEMP
    dt = 1.5e-3

    integrator = LangevinIntegrator(temperature, dt, 1.0, host_masses, seed)

    bond_list = get_bond_list(host_bps[0].potential)
    group_idxs = get_group_indices(bond_list, len(host_masses))
    barostat_interval = 5

    barostat = MonteCarloBarostat(
        len(host_masses), DEFAULT_PRESSURE, temperature, group_idxs, barostat_interval, seed + 1
    )

    initial_state = InitialState(
        host_bps, integrator, barostat, solvent_conf, np.zeros_like(solvent_conf), solvent_box, lamb, ligand_idxs
    )

    return initial_state


def calibrate_gcmc(water_ff, box_width, seed, nb_cutoff=1.2):
    mdp = MDParams(n_frames=1000, n_eq_steps=100000, steps_per_frame=1000, seed=seed)

    partial_get_initial_state_fn = functools.partial(
        _get_initial_state, water_ff=water_ff, box_width=box_width, seed=seed, nb_cutoff=nb_cutoff
    )

    _get_initial_state(lamb=0, water_ff=water_ff, box_width=box_width, seed=seed, nb_cutoff=nb_cutoff)

    n_bisections = 24
    initial_lambdas = [0.0, 1.0]
    results, frames, boxes = run_sims_with_greedy_bisection(
        initial_lambdas, partial_get_initial_state_fn, mdp, n_bisections, DEFAULT_TEMP, verbose=True
    )

    final_result = results[-1]

    print("Chemical Potential", np.sum(final_result.dGs), "kJ/mol")

    # use the fully interacting state to compute standard state volumes
    # n_windows x n_frames x n_atoms x n_dims
    num_waters = len(frames[0][0]) // 3
    water_vols_0 = []
    water_densities_0 = []
    for b in boxes[0]:
        vol = np.product(np.diag(b))
        water_vols_0.append(vol / num_waters)
        water_densities_0.append(compute_density(num_waters, b))

    print("Average Standard Volume (E0)", np.mean(water_vols_0), "nm^3")
    print("Average Density (E0)", np.mean(water_densities_0), "kg/m^3")
    # self consistency check on standard volume,
    # now we use the non-interacting state
    water_vols_1 = []
    water_densities_1 = []
    for b in boxes[-1]:
        vol = np.product(np.diag(b))
        water_vols_1.append(vol / (num_waters - 1))  # one less water in the non-interacting state
        water_densities_1.append(compute_density(num_waters, b))

    # if we use NPT, hopefully the two end-states converge to the same
    print("Average Standard Volume (E1)", np.mean(water_vols_1), "nm^3")
    print("Average Density (E1)", np.mean(water_densities_1), "kg/m^3")


if __name__ == "__main__":
    ff = Forcefield.load_default()
    calibrate_gcmc(ff.water_ff, box_width=5.0, seed=2023, nb_cutoff=1.2)
