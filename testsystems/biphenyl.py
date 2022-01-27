from typing import Tuple
from timemachine.lib import custom_ops
from md.barostat.utils import get_group_indices, get_bond_list
from timemachine import lib

# copied from https://raw.githubusercontent.com/proteneer/timemachine/fa751c8e3c6ff51601d4ea1a58d4c6b7e35c4f19/tests/test_smc.py
from jax.config import config
import pickle
import time

from md.states import CoordsVelBox

config.update("jax_enable_x64", True)

from md import enhanced

from rdkit import Chem
import numpy as np

import jax
from timemachine.potentials import bonded

from fe import functional
from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers

from fe import free_energy
from md import builders
import os

# (ytz): useful for visualization, so please leave this comment here!
import asciiplotlib as apl

from timemachine.constants import BOLTZ

temperature = 300.0
pressure = 1.0

kBT = BOLTZ * temperature


class MonteCarloMove:
    n_proposed: int = 0
    n_accepted: int = 0

    def propose(self, x: CoordsVelBox, lam: float) -> Tuple[CoordsVelBox, float]:
        """return proposed state and log acceptance probability"""
        raise NotImplementedError

    def move(self, x: CoordsVelBox, lam: float) -> CoordsVelBox:
        proposal, log_acceptance_probability = self.propose(x, lam)
        self.n_proposed += 1

        alpha = np.random.rand()
        acceptance_probability = np.exp(log_acceptance_probability)
        if alpha < acceptance_probability:
            self.n_accepted += 1
            return proposal
        else:
            return x

    @property
    def acceptance_fraction(self):
        if self.n_proposed > 0:
            return self.n_accepted / self.n_proposed
        else:
            return 0.0


class MoveImpl:
    def __init__(self, bound_impls, barostat_impl, integrator_impl):
        self.bound_impls = bound_impls
        self.barostat_impl = barostat_impl
        self.integrator_impl = integrator_impl


class NPTMove(MonteCarloMove):
    def __init__(
            self,
            ubps,
            masses,
            temperature,
            pressure,
            n_steps,
            seed,
            dt=1.5e-3,
            friction=1.0,
            barostat_interval=5,
    ):
        print('constructing a new mover!')

        self.ubps = ubps
        self.masses = masses
        self.temperature = temperature
        self.pressure = pressure
        self.seed = seed
        self.dt = dt
        self.friction = friction
        self.barostat_interval = barostat_interval

        # intg = lib.LangevinIntegrator(temperature, dt, friction, masses, seed)
        # self.integrator_impl = intg.impl()
        # all_impls = [bp.bound_impl(np.float32) for bp in ubps]

        bond_list = get_bond_list(ubps[0])
        self.group_idxs = get_group_indices(bond_list)

        # barostat = lib.MonteCarloBarostat(len(masses), pressure, temperature, group_idxs, barostat_interval, seed + 1)
        # barostat_impl = barostat.impl(all_impls)

        # self.bound_impls = all_impls
        # self.barostat_impl = barostat_impl

        self.integrator_impl = None
        self.barostat_impl = None
        self.move_impl = None
        self.n_steps = n_steps

    def initialize_once(self):
        if self.move_impl is None:

            if "CUDA_VISIBLE_DEVICES" in os.environ:
                print("Initializing on:", os.environ["CUDA_VISIBLE_DEVICES"])
            else:
                print("initialize_once() called serially")

            bound_impls = [bp.bound_impl(np.float32) for bp in self.ubps]
            intg_impl = lib.LangevinIntegrator(self.temperature, self.dt, self.friction, self.masses, self.seed).impl()
            barostat_impl = lib.MonteCarloBarostat(
                len(self.masses), pressure, self.temperature, self.group_idxs, self.barostat_interval, self.seed + 1
            ).impl(bound_impls)
            self.move_impl = MoveImpl(bound_impls, barostat_impl, intg_impl)

        # else do nothing

    def propose(self, x: CoordsVelBox, lam: float):

        self.initialize_once()
        # note: context creation overhead here is actually very small!

        # print('impl', self.move_impl)
        ctxt = custom_ops.Context(
            x.coords,
            x.velocities,
            x.box,
            self.move_impl.integrator_impl,
            self.move_impl.bound_impls,
            self.move_impl.barostat_impl,
        )

        # arguments: lambda_schedule, du_dl_interval, x_interval
        _ = ctxt.multiple_steps(lam * np.ones(self.n_steps), 0, 0)
        x_t = ctxt.get_x_t()
        v_t = ctxt.get_v_t()
        box = ctxt.get_box()

        after_npt = CoordsVelBox(x_t, v_t, box)
        log_accept_prob = 0.0  # always accept

        return after_npt, log_accept_prob


temperature = 300
kBT = BOLTZ * temperature


class PotentialEnergyFunction:
    def __init__(self, ubps, params):
        self.ubps = ubps
        self.params = params
        self.U_fn = None

    def initialize_once(self):
        if self.U_fn is None:
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                print("Initializing on:", os.environ["CUDA_VISIBLE_DEVICES"])
            else:
                print("initialize_once() called serially")
            self.U_fn = functional.construct_differentiable_interface_fast(self.ubps, self.params)

    def u(self, xvb, lam):
        self.initialize_once()
        return self.U_fn(xvb.coords, self.params, xvb.box, lam) / kBT


def get_biphenyl():
    MOL_SDF = """
  Mrv2118 11122115063D

 22 23  0  0  0  0            999 V2000
   -0.5376   -2.1603   -1.0521 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2440   -3.3774   -1.0519 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.1258   -3.6819    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.3029   -2.7660    1.0519 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.6021   -1.5457    1.0521 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7097   -1.2292    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0003   -0.0005    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6954    1.2325    0.0063 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0098    2.4503    0.0035 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.4158    2.4522    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.1171    1.2336   -0.0035 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.4151    0.0140   -0.0063 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0951   -1.9564   -1.8304 F   0  0  0  0  0  0  0  0  0  0  0  0
   -1.1164   -4.0414   -1.8186 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.6370   -4.5674    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.9418   -2.9876    1.8186 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.7418   -0.8958    1.8304 F   0  0  0  0  0  0  0  0  0  0  0  0
   -1.7187    1.2541    0.0033 F   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5012    3.3357    0.0034 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.9270    3.3377    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.1394    1.2338   -0.0034 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.9454   -0.8614   -0.0033 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0  0  0  0
  2  3  1  0  0  0  0
  3  4  2  0  0  0  0
  4  5  1  0  0  0  0
  5  6  2  0  0  0  0
  6  1  1  0  0  0  0
  7  8  2  0  0  0  0
  8  9  1  0  0  0  0
  9 10  2  0  0  0  0
 10 11  1  0  0  0  0
 11 12  2  0  0  0  0
 12  7  1  0  0  0  0
  6  7  1  0  0  0  0
  2 14  1  0  0  0  0
  3 15  1  0  0  0  0
  4 16  1  0  0  0  0
  9 19  1  0  0  0  0
 10 20  1  0  0  0  0
 11 21  1  0  0  0  0
 12 22  1  0  0  0  0
  5 17  1  0  0  0  0
  1 13  1  0  0  0  0
  8 18  1  0  0  0  0
M  END
$$$$"""
    mol = Chem.MolFromMolBlock(MOL_SDF, removeHs=False)
    torsion_idxs = np.array([[4, 5, 6, 7]])
    return mol, torsion_idxs


def get_ff_am1cc():
    # old: "/home/jfass/Documents/GitHub/timemachine/ff/params/smirnoff_1_1_0_ccc.py"
    ff_handlers = deserialize_handlers(
        open("/home/jfass/Documents/GitHub/timemachine/ff/params/smirnoff_1_1_0_ccc.py").read())
    ff = Forcefield(ff_handlers)
    return ff


def generate_solvent_samples(coords, box, masses, ubps, params, temperature, pressure, seed, n_samples):
    num_equil_steps = 50000  # bump to 50k to be safe/production
    xvb0 = enhanced.equilibrate_solvent_phase(
        ubps, params, masses, coords, box, temperature, pressure, num_equil_steps, seed
    )

    md_steps_per_move = 1000  # probably good enough?
    lamb = 1.0  # non-interacting state
    npt_mover = NPTMove(ubps, masses, temperature, pressure, n_steps=md_steps_per_move, seed=seed)
    xvbs = []
    xvb_t = xvb0
    for _ in range(n_samples):
        xvb_t = npt_mover.move(xvb_t, lamb)
        xvbs.append(xvb_t)
    return xvbs


def generate_ligand_samples(num_batches, mol, ff, temperature, seed):
    state = enhanced.VacuumState(mol, ff)
    proposal_U = state.U_full
    vacuum_samples, vacuum_log_weights = enhanced.generate_log_weighted_samples(
        mol, temperature, state.U_easy, proposal_U, num_batches=num_batches, seed=seed
    )

    return vacuum_samples, vacuum_log_weights


def generate_endstate_samples(num_samples, solvent_samples, ligand_samples, ligand_log_weights, num_ligand_atoms):
    """solvent + (noninteracting ligand) sample --> solvent + (vacuum ligand) sample

    Assumptions:
    ------------
    * ligand indices: last num_ligand_atoms"""
    all_xvbs = []
    for _ in range(num_samples):
        choice_idx = np.random.choice(np.arange(len(solvent_samples)))
        solvent_x = solvent_samples[choice_idx].coords
        solvent_v = solvent_samples[choice_idx].velocities
        ligand_xv = enhanced.sample_from_log_weights(ligand_samples, ligand_log_weights, size=1)[0]
        ligand_x = ligand_xv[0]
        ligand_v = ligand_xv[1]
        combined_x = np.concatenate([solvent_x[:-num_ligand_atoms], ligand_x], axis=0)
        combined_v = np.concatenate([solvent_v[:-num_ligand_atoms], ligand_v], axis=0)
        combined_box = solvent_samples[choice_idx].box
        all_xvbs.append(CoordsVelBox(combined_x, combined_v, combined_box))
    return all_xvbs


mol, torsion_idxs = get_biphenyl()


@jax.jit
def get_torsion(x_l):
    ci = x_l[torsion_idxs[:, 0]]
    cj = x_l[torsion_idxs[:, 1]]
    ck = x_l[torsion_idxs[:, 2]]
    cl = x_l[torsion_idxs[:, 3]]
    # last [0] is used to return from a length-1 array
    return bonded.signed_torsion_angle(ci, cj, ck, cl)[0]


ligand_masses = np.array([a.GetMass() for a in mol.GetAtoms()])
num_ligand_atoms = len(ligand_masses)


def mini_get_solvent_phase_system(mol, ff):
    masses = np.array([a.GetMass() for a in mol.GetAtoms()])
    water_system, water_coords, water_box, water_topology = builders.build_water_system(3.0)
    water_box = water_box + np.eye(3) * 0.5  # add a small margin around the box for stability
    afe = free_energy.AbsoluteFreeEnergy(mol, ff)
    ff_params = ff.get_ordered_params()
    ubps, params, masses, coords = afe.prepare_host_edge(ff_params, water_system, water_coords)

    # commented out minimization

    return ubps, params, masses, coords, water_box


def construct_biphenyl_test_system(n_steps=1000):
    """
    Generate samples from the equilibrium distribution at lambda=1

    Return:
    * potential_energy_fxn : classy for parallelism-reasons
    * mover : classy for parallelism reasons
    * initial_samples
    """

    seed = 2021
    np.random.seed(seed)

    ff = get_ff_am1cc()

    cache_path = "test_smc_cache.pkl"
    if not os.path.exists(cache_path):

        print("Generating cache")
        ubps, params, masses, coords, box = enhanced.get_solvent_phase_system(mol, ff)

        n_solvent_samples = 1000  # should be maybe 1000
        print(f"generating {n_solvent_samples} solvent samples")
        solvent_xvbs = generate_solvent_samples(
            coords, box, masses, ubps, params, temperature, pressure, seed, n_solvent_samples
        )

        n_ligand_batches = os.cpu_count() * 2000  # should be 30k # 24 == os.cpu_count()
        print(f"generating ligand samples")
        ligand_samples, ligand_log_weights = generate_ligand_samples(n_ligand_batches, mol, ff, temperature, seed)

        with open(cache_path, "wb") as fh:
            pickle.dump([solvent_xvbs, ligand_samples, ligand_log_weights], fh)
    else:
        # elide minimize_host_4d
        ubps, params, masses, coords, box = mini_get_solvent_phase_system(mol, ff)

    with open(cache_path, "rb") as fh:
        print("Loading cache")
        solvent_xvbs, ligand_samples, ligand_log_weights = pickle.load(fh)

    n_endstate_samples = 5000
    all_xvbs = generate_endstate_samples(
        n_endstate_samples, solvent_xvbs, ligand_samples, ligand_log_weights, num_ligand_atoms
    )

    # plot torsions at the end-states
    end_state_torsions = []
    for xvb in all_xvbs:
        x = xvb.coords
        x_l = x[-num_ligand_atoms:]
        end_state_torsions.append(get_torsion(x_l))

    # plot histogram using asciiplotlib
    print("torsion distribution in sample cache")
    fig = apl.figure()
    fig.hist(*np.histogram(end_state_torsions, bins=25, range=(-np.pi, np.pi)))
    fig.show()

    for u, p in zip(ubps, params):
        u.bind(p)

    # propagate with NPTMove
    seed = int(time.time())

    mover = NPTMove(ubps, masses, temperature, pressure, n_steps, seed)
    potential_energy_fxn = PotentialEnergyFunction(ubps, params)

    return potential_energy_fxn, mover, all_xvbs