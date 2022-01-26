# absolute hydration free energy

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

import timemachine
from timemachine.constants import BOLTZ


from ff import Forcefield
from ff.handlers import openmm_deserializer
from ff.handlers.deserialize import deserialize_handlers

from fe import topology
from fe.utils import get_romol_conf
from fe.free_energy import construct_lambda_schedule

from md import builders, minimizer
from md.moves import NPTMove
from md.states import CoordsVelBox

from pymbar import BAR
from typing import Dict, Tuple, List
from numpy.typing import ArrayLike

from pathlib import Path

from tqdm import tqdm


# force field
tm_path = Path(timemachine.__path__[0]).parent
path_to_ff = tm_path / "ff/params/smirnoff_1_1_0_ccc.py"
with open(path_to_ff, "r") as f:
    ff_handlers = deserialize_handlers(f.read())
ff = Forcefield(ff_handlers)

# ligand
romol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O"))
ligand_masses = [a.GetMass() for a in romol.GetAtoms()]
AllChem.EmbedMolecule(romol)
ligand_coords = get_romol_conf(romol)

# water box
host_system, host_coords, box, omm_topology = builders.build_water_system(4.0)
host_bps, host_masses = openmm_deserializer.deserialize_system(host_system, cutoff=1.2)
num_host_atoms = host_coords.shape[0]

# minimize
min_host_coords = minimizer.minimize_host_4d([romol], host_system, host_coords, ff, box)
# water box + ligand
masses = np.concatenate([host_masses, ligand_masses])
coords = np.concatenate([min_host_coords, ligand_coords])

# alchemical topologies / potentials
final_potentials = []

gbt = topology.BaseTopology(romol, ff)
hgt = topology.HostGuestTopology(host_bps, gbt)

fn_handle_tuples = [
    [hgt.parameterize_harmonic_bond, [ff.hb_handle]],
    [hgt.parameterize_harmonic_angle, [ff.ha_handle]],
    [hgt.parameterize_periodic_torsion, [ff.pt_handle, ff.it_handle]],
    [hgt.parameterize_nonbonded, [ff.q_handle, ff.lj_handle]],
]

for fn, handles in fn_handle_tuples:
    params, potential = fn(*[h.params for h in handles])
    final_potentials.append(potential.bind(params))

bound_impls = [p.bound_impl(np.float32) for p in final_potentials]

temperature = 300
pressure = 1.0


def u(x: CoordsVelBox, lam: float) -> float:
    # TODO: replace with SummedPotential
    U_s = np.array([bp.execute(x.coords, x.box, lam)[-1] for bp in bound_impls])
    u_s = U_s / (BOLTZ * temperature)
    return np.sum(u_s)


# MD
n_steps_per_move = 50
n_prod_samples = 100
n_equil_moves = 50

dt = 1.5e-3

seed = 2022


def construct_npt_move(lam):
    return NPTMove(final_potentials, lam, masses, temperature, pressure, n_steps_per_move, seed)


# collect samples from p(x | lam) for lam in lambda_schedule
lambda_schedule = construct_lambda_schedule(32)

trajs = []

for lam in lambda_schedule:
    print("lam = ", lam)

    x_equil = CoordsVelBox(coords, np.zeros_like(coords), box)

    npt = construct_npt_move(lam)
    assert npt.lamb == lam
    trange = tqdm(range(n_equil_moves), desc="equilibration")
    for _ in trange:
        x_equil = npt.move(x_equil)
    print("u(x_equil)", u(x_equil, lam))

    # production
    traj = [x_equil]
    trange = tqdm(range(n_prod_samples - 1), desc="production")
    for _ in trange:
        traj.append(npt.move(traj[-1]))
    trajs.append(traj)

# analyze
# TODO: move this into fe module, if it's not there already
Array = ArrayLike
Traj = List[CoordsVelBox]
Diagnostics = Dict


def vec_u(xs: Traj, lam: float) -> Array:
    return np.array([u(x, lam) for x in xs])


def work(xs: Traj, from_lam: float, to_lam: float) -> Array:
    return vec_u(xs, to_lam) - vec_u(xs, from_lam)


def pair_bar(trajs: List[Traj], lambda_schedule: Array) -> Tuple[float, Diagnostics]:
    assert len(trajs) == len(lambda_schedule)

    works = []
    estimates = []
    for i in range(len(trajs) - 1):
        w_F = work(trajs[i], lambda_schedule[i], lambda_schedule[i + 1])
        w_R = work(trajs[i + 1], lambda_schedule[i + 1], lambda_schedule[i])

        works.append((w_F, w_R))
        df, ddf = BAR(w_F, w_R)
        estimates.append((df, ddf))

    delta_f = sum([df for (df, _) in estimates])

    return delta_f, dict(works=works, estimates=estimates)


delta_f_decouple, diagnostics = pair_bar(trajs, lambda_schedule)
delta_f_hydration = -delta_f_decouple  # reduced
delta_F_hydration = delta_f_hydration * (BOLTZ * temperature)  # in kJ/mol

print(f"estimated absolute hydration free energy: {delta_F_hydration:.3f} kJ/mol")
print("(delta_f, err_est) (in k_B T) for each lambda increment")
for (df, ddf) in diagnostics["estimates"]:
    print(f"{df:.3f} +/- {ddf:.3f}")
