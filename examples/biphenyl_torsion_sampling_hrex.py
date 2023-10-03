import argparse
from dataclasses import replace
from typing import Iterable, List, Optional, Tuple, cast

import jax
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from timemachine.constants import DEFAULT_TEMP
from timemachine.fe import model_utils
from timemachine.fe.free_energy import HostConfig, HREXParams, InitialState, MDParams, run_sims_bisection, run_sims_hrex
from timemachine.fe.plots import plot_hrex_replica_state_distribution_heatmap, plot_hrex_transition_matrix
from timemachine.fe.topology import BaseTopology, HostGuestTopology
from timemachine.fe.utils import get_mol_masses, get_romol_conf
from timemachine.ff import Forcefield, ForcefieldParams
from timemachine.ff.handlers import openmm_deserializer
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat
from timemachine.md import builders, minimizer
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.md.hrex import HREXDiagnostics
from timemachine.potentials import BoundPotential, NonbondedPairListPrecomputed, SummedPotential
from timemachine.potentials.bonded import signed_torsion_angle
from timemachine.potentials.potentials import HarmonicBond, Nonbonded, NonbondedInteractionGroup, PeriodicTorsion
from timemachine.testsystems.ligands import get_biphenyl


def get_potentials(
    top: BaseTopology | HostGuestTopology,
    ff_params: ForcefieldParams,
    decouple_atom_idxs: NDArray[np.int_],
    lamb: float,
) -> List[BoundPotential]:
    assert 0.0 <= lamb <= 1.0

    if isinstance(top, BaseTopology):
        return get_potentials_vacuum(top, ff_params, decouple_atom_idxs, lamb)
    elif isinstance(top, HostGuestTopology):
        return get_potentials_solvent(top, ff_params, decouple_atom_idxs, lamb)
    else:
        assert False


def get_potentials_vacuum(
    top: BaseTopology,
    ff_params: ForcefieldParams,
    decouple_atom_idxs: NDArray[np.int_],
    lamb: float,
) -> List[BoundPotential]:
    hb_params, hb_pot = top.parameterize_harmonic_bond(ff_params.hb_params)
    ha_params, ha_pot = top.parameterize_harmonic_angle(ff_params.ha_params)

    ppt_params, ppt_pot = top.parameterize_proper_torsion(ff_params.pt_params)
    ipt_params, ipt_pot = top.parameterize_improper_torsion(ff_params.it_params)
    pt_params = np.concatenate([ppt_params, ipt_params])
    pt_pot = PeriodicTorsion(np.concatenate([ppt_pot.idxs, ipt_pot.idxs]))
    nb_params, nb_pot = top.parameterize_nonbonded_pairlist(
        ff_params.q_params,
        ff_params.q_params_intra,
        ff_params.lj_params,
        ff_params.lj_params_intra,
    )

    decouple_atom = np.isin(nb_pot.idxs, decouple_atom_idxs)
    decouple_pair = np.logical_xor(decouple_atom[:, 0], decouple_atom[:, 1])
    nb_params[decouple_pair, 3] = lamb

    return [
        hb_pot.bind(hb_params),
        ha_pot.bind(ha_params),
        pt_pot.bind(pt_params),
        nb_pot.bind(nb_params),
    ]


def get_potentials_solvent(
    top: HostGuestTopology,
    ff_params: ForcefieldParams,
    decouple_atom_idxs: NDArray[np.int_],
    lamb: float,
) -> List[BoundPotential]:
    hb_params, hb_pot = top.parameterize_harmonic_bond(ff_params.hb_params)
    ha_params, ha_pot = top.parameterize_harmonic_angle(ff_params.ha_params)

    pt_params, pt_pot = top.parameterize_periodic_torsion(ff_params.pt_params, ff_params.it_params)
    nb_params, nb_pot = top.parameterize_nonbonded(
        ff_params.q_params,
        ff_params.q_params_intra,
        ff_params.q_params_solv,
        ff_params.lj_params,
        ff_params.lj_params_intra,
        ff_params.lj_params_solv,
        lamb,
    )

    # for host-guest topology, need to modify parameters for ligand-water and ligand-ligand nonbonded potentials
    # separately
    assert isinstance(nb_pot, SummedPotential)
    ww_pot, lw_pot, ll_pot = nb_pot.potentials
    assert isinstance(ww_pot, Nonbonded)
    assert isinstance(lw_pot, NonbondedInteractionGroup)
    assert isinstance(ll_pot, NonbondedPairListPrecomputed)

    ww_params, lw_params, ll_params = nb_pot.unflatten_params(nb_params)
    lw_params = cast(jax.Array, lw_params)
    ll_params = cast(jax.Array, ll_params)

    lw_params = lw_params.at[decouple_atom_idxs, 3].set(lamb)

    decouple_atom = np.isin(ll_pot.idxs, decouple_atom_idxs)
    decouple_pair = np.logical_xor(decouple_atom[:, 0], decouple_atom[:, 1])
    ll_params.at[decouple_pair, 3].set(lamb)

    # decouple a selection of atoms from both ligand and environment
    nb_params = np.concatenate([ww_params.flatten(), lw_params.flatten(), ll_params.flatten()])

    return [
        hb_pot.bind(hb_params),
        ha_pot.bind(ha_params),
        pt_pot.bind(pt_params),
        nb_pot.bind(nb_params),
    ]


def sample_biphenyl_hrex(
    solvent: bool, min_overlap: float, n_frames_bisection: int, n_frames: int
) -> Tuple[List[NDArray], List[NDArray], List[float], HREXDiagnostics]:
    seed = 2023

    md_params = MDParams(
        n_frames=n_frames,
        n_eq_steps=100_000 if solvent else 10_000,
        steps_per_frame=400,
        seed=seed,
        hrex_params=HREXParams(n_frames_bisection=n_frames_bisection, n_frames_per_iter=1),
    )
    assert md_params.hrex_params

    temperature = DEFAULT_TEMP

    mol, torsion_idxs = get_biphenyl()
    ff = Forcefield.load_from_file("smirnoff_2_0_0_ccc.py")

    bt = BaseTopology(mol, ff)
    ligand_masses = get_mol_masses(mol)
    decouple_atom_idxs = np.array([12, 16], dtype=np.int_)

    x0_ligand = get_romol_conf(mol)
    baro: Optional[MonteCarloBarostat] = None

    if solvent:
        # construct water box
        water_system, water_coords, water_box, _ = builders.build_water_system(box_width=3.0, water_ff=ff.water_ff)
        water_box += 0.5 * np.eye(3)  # add a small margin around the box for stability
        num_water_atoms = water_coords.shape[0]
        host_config = HostConfig(water_system, water_coords, water_box, num_water_atoms)
        host_bps, host_masses = openmm_deserializer.deserialize_system(water_system, cutoff=1.2)

        top = HostGuestTopology(host_bps, bt, num_water_atoms)

        # translate ligand indices to system indices
        decouple_atom_idxs += num_water_atoms
        torsion_idxs += num_water_atoms

        combined_masses = np.concatenate([host_masses, ligand_masses])
        integrator = LangevinIntegrator(temperature, dt=2.5e-3, friction=1.0, masses=combined_masses, seed=seed)

        bps = get_potentials_solvent(top, ff.get_params(), decouple_atom_idxs, 0.0)
        bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential
        hmr_masses = model_utils.apply_hmr(combined_masses, bond_pot.idxs)
        group_idxs = get_group_indices(get_bond_list(bond_pot), len(combined_masses))
        baro = MonteCarloBarostat(len(hmr_masses), 1.0, temperature, group_idxs, 15, seed)

        x0_env = minimizer.minimize_host_4d([mol], host_config, ff)
        x0 = np.concatenate([x0_env, x0_ligand])
        box0 = water_box

        ligand_idxs = np.arange(num_water_atoms, top.get_num_atoms())
    else:
        top = bt
        integrator = LangevinIntegrator(temperature, dt=2.5e-3, friction=1.0, masses=ligand_masses, seed=seed)
        x0 = x0_ligand
        box0 = np.eye(3) * 10.0
        ligand_idxs = np.arange(x0_ligand.shape[0])

    v0 = np.zeros_like(x0)

    def make_initial_state(lamb: float) -> InitialState:
        bps = get_potentials(top, ff.get_params(), decouple_atom_idxs, lamb)
        return InitialState(bps, integrator, baro, x0, v0, box0, lamb, ligand_idxs)

    results, trajectories_by_state = run_sims_bisection(
        [0.0, 1.0],
        make_initial_state,
        replace(md_params, n_frames=md_params.hrex_params.n_frames_bisection),
        n_bisections=30,  # maximum number if min_overlap not achieved
        min_overlap=min_overlap,
        temperature=temperature,
    )

    initial_states = results[-1].initial_states

    initial_states_hrex = [
        replace(initial_state, x0=traj.frames[-1], v0=traj.final_velocities, box0=traj.boxes[-1])
        for initial_state, traj in zip(initial_states, trajectories_by_state)
    ]

    _, trajectories_by_state_hrex, diagnostics = run_sims_hrex(
        initial_states_hrex,
        replace(md_params, n_eq_steps=0),  # using pre-equilibrated samples
        n_frames_per_iter=md_params.hrex_params.n_frames_per_iter,
    )

    def get_torsion_angle_traj(frames: NDArray) -> NDArray:
        traj = np.concatenate([np.array(frame)[torsion_idxs] for frame in frames])  # (n_frames, 4, 3)
        phi_traj = np.asarray(signed_torsion_angle(*traj.swapaxes(0, 1)))
        return phi_traj

    phi_traj_by_state = [get_torsion_angle_traj(np.array(traj.frames)) for traj in trajectories_by_state]
    phi_traj_by_state_hrex = [get_torsion_angle_traj(np.array(traj.frames)) for traj in trajectories_by_state_hrex]

    lambdas = [s.lamb for s in initial_states]

    return phi_traj_by_state, phi_traj_by_state_hrex, lambdas, diagnostics


def plot_occupancy(phi_traj_by_state: Iterable[NDArray], lambda_by_state: Iterable[float], window_size: int = 1):
    _, ax = plt.subplots()
    for lamb, phi_traj in zip(lambda_by_state, phi_traj_by_state):
        is_left = phi_traj < 0.0
        fraction_left_rolling = np.convolve(is_left, np.ones(window_size) / window_size, mode="valid")
        lw = 2 if lamb == 0.0 else 1  # bold line for lambda = 0
        ax.plot(fraction_left_rolling, label=f"{lamb:.3g}", lw=lw)

    ax.axhline(0.5, color="gray", ls="--", lw=2)
    ax.set_xlabel("frame")
    ax.set_ylabel(r"cumulative fraction $\varphi < 0$")
    ax.legend(title=r"$\lambda$")


def plot_torsion_angle_trajectories(phi_traj_by_state: Iterable[NDArray], lambda_by_state: Iterable[float]):
    _, ax = plt.subplots()
    for lamb, phi_traj in zip(lambda_by_state, phi_traj_by_state):
        ax.plot(phi_traj, label=f"{lamb:.3g}")

    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    ax.set_xlabel("frame")
    ax.set_ylabel(r"$\varphi$")
    ax.legend(title=r"$\lambda$")


def main():
    parser = argparse.ArgumentParser(description="Example sampling sterically-hindered biphenyl using HREX")

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")

    run_parser.add_argument("--solvent", action="store_true", help="Run in solvent")
    run_parser.add_argument(
        "--min_overlap",
        type=float,
        default=0.6,
        help="Minimum overlap between adjacent windows",
    )
    run_parser.add_argument(
        "--n_frames_bisection",
        type=int,
        default=1_000,
        help="Number of frames (picoseconds) for bisection to determine the lambda schedule",
    )
    run_parser.add_argument(
        "--n_frames", type=int, default=4_000, help="Number of frames (picoseconds) to sample using HREX"
    )

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("--save", action="store_true")

    args = parser.parse_args()

    if args.command == "run":
        phi_traj_by_state, phi_traj_by_state_hrex, lambdas, diagnostics = sample_biphenyl_hrex(
            args.solvent or False, args.min_overlap, args.n_frames_bisection, args.n_frames
        )
        np.savez(
            "hrex_biphenyl_data.npz",
            lambdas=lambdas,
            phi_traj_by_state=phi_traj_by_state,
            phi_traj_by_state_hrex=phi_traj_by_state_hrex,
            replica_idx_by_state_by_iter=diagnostics.replica_idx_by_state_by_iter,
            fraction_accepted_by_pair_by_iter=diagnostics.fraction_accepted_by_pair_by_iter,
        )
    else:
        data = np.load("hrex_biphenyl_data.npz")
        lambdas = data["lambdas"]
        phi_traj_by_state = data["phi_traj_by_state"]
        phi_traj_by_state_hrex = data["phi_traj_by_state_hrex"]
        diagnostics = HREXDiagnostics(data["replica_idx_by_state_by_iter"], data["fraction_accepted_by_pair_by_iter"])

        def savefig(name):
            if args.save:
                plt.savefig(name, bbox_inches="tight")

        plot_hrex_transition_matrix(diagnostics.transition_matrix)
        savefig("hrex_biphenyl_transition_matrix.png")

        plot_hrex_replica_state_distribution_heatmap(diagnostics.cumulative_replica_state_counts)
        savefig("hrex_biphenyl_replica_state_distribution.png")

        def plot(phi_traj_by_state, title_suffix, name_suffix=""):
            plot_occupancy(phi_traj_by_state, lambdas, window_size=400)
            plt.title(f"Occupancy {title_suffix}")
            savefig(f"hrex_biphenyl_cumulative_fraction_left_by_state{name_suffix}.png")

            plot_torsion_angle_trajectories(phi_traj_by_state[:1], lambdas[:1])
            plt.title(f"Torsion angle trajectory {title_suffix}")
            savefig(f"hrex_biphenyl_torsion_angle_trajectory_by_state{name_suffix}.png")

        plot(phi_traj_by_state, "(no HREX)")
        plot(phi_traj_by_state_hrex, "(HREX)", name_suffix="_hrex")

        if not args.save:
            plt.show()


if __name__ == "__main__":
    main()
