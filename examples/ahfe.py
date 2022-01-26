# absolute hydration free energy

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

import timemachine

from ff import Forcefield
from ff.handlers import openmm_deserializer
from ff.handlers.deserialize import deserialize_handlers

from fe import topology
from fe.utils import get_romol_conf

from md import builders
from md.moves import NPTMove
from md.states import CoordsVelBox

from pathlib import Path

from tqdm import tqdm

# force field
tm_path = Path(timemachine.__path__[0]).parent
path_to_ff = tm_path / "ff/params/smirnoff_1_1_0_ccc.py"
with open(path_to_ff, "r") as f:
    ff_handlers = deserialize_handlers(f.read())

# ligand
romol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O"))
ligand_masses = [a.GetMass() for a in romol.GetAtoms()]
AllChem.EmbedMolecule(romol)
ligand_coords = get_romol_conf(romol)

# water box
system, host_coords, box, omm_topology = builders.build_water_system(4.0)
host_bps, host_masses = openmm_deserializer.deserialize_system(system, cutoff=1.2)
num_host_atoms = host_coords.shape[0]

# water box + ligand
masses = np.concatenate([host_masses, ligand_masses])
coords = np.concatenate([host_coords, ligand_coords])

# alchemical topologies / potentials
final_potentials = []
ff = Forcefield(ff_handlers)
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

# MD
n_steps_per_move = 50
n_prod_samples = 100
n_equil_moves = 50

temperature = 300
dt = 1.5e-3
friction = 10
pressure = 1.0
seed = 2022


def construct_npt_move(lam, friction=1.0):
    return NPTMove(final_potentials, lam, masses, temperature, pressure, n_steps_per_move, seed, friction=friction)


# collect samples from p(x | lam) for lam in lambda_schedule
lambda_schedule = np.linspace(0, 1, 8)
trajs = []

for lam in lambda_schedule:
    print("lam = ", lam)

    x0 = CoordsVelBox(coords, np.zeros_like(coords), box)

    # initial insertion step to remove clashes
    # TODO: this is currently very slow, since NPTMove doesn't allow lambda to be set on the fly
    #   --> modify NPTMove so that this can be fast
    x_relaxed = x0
    declash_schedule = np.linspace(1.0, lam, 2)
    trange = tqdm(declash_schedule, desc="resolving clashes")
    for noneq_lam in trange:
        npt = construct_npt_move(noneq_lam, friction=np.inf)
        x_relaxed = npt.move(x_relaxed)

    # equilibration
    x_equil = x_relaxed
    assert npt.lamb == lam
    trange = tqdm(range(n_equil_moves), desc="equilibration")
    for _ in trange:
        x_equil = npt.move(x_equil)

    # production
    traj = [x_equil]
    trange = tqdm(range(n_prod_samples - 1), desc="production")
    for _ in trange:
        traj.append(npt.move(traj[-1]))
    trajs.append(traj)

# analyze
# TODO: define u(x, lam)
# TODO: define work(xs, from_lam, to_lam)
# TODO: \sum_i BAR(w_F=work(xs[i], lam[i], lam[i+1]), w_R=work(xs[i+1], lam[i+1], lam[i]))
