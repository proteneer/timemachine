# adapted from https://raw.githubusercontent.com/proteneer/timemachine/fa751c8e3c6ff51601d4ea1a58d4c6b7e35c4f19/tests/test_smc.py

import jax

jax.config.update("jax_enable_x64", True)

import os
import pickle
import time
from pathlib import Path

from rdkit import Chem
import numpy as np

import timemachine
from timemachine.constants import BOLTZ
from timemachine.potentials import bonded

from fe import functional, free_energy
from fe.absolute_hydration import generate_solvent_samples, generate_ligand_samples, generate_endstate_samples

from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers

from md import builders, enhanced
from md.noneq import NPTMove

## (ytz): useful for visualization, so please leave this comment here!
# import asciiplotlib as apl

temperature = 300.0
pressure = 1.0

kBT = BOLTZ * temperature

temperature = 300
kBT = BOLTZ * temperature


# TODO: where to move this definition?
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


def get_ff_am1ccc():
    tm_path = Path(timemachine.__path__[0]).parent
    path_to_ff = tm_path / "ff/params/smirnoff_1_1_0_ccc.py"
    with open(path_to_ff, "r") as f:
        ff_handlers = deserialize_handlers(f.read())
    ff = Forcefield(ff_handlers)
    return ff


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


def get_solvent_phase_system(mol, ff):
    water_system, water_coords, water_box, water_topology = builders.build_water_system(3.0)
    water_box = water_box + np.eye(3) * 0.5  # add a small margin around the box for stability
    afe = free_energy.AbsoluteFreeEnergy(mol, ff)
    ff_params = ff.get_ordered_params()
    ubps, params, masses, coords = afe.prepare_host_edge(ff_params, water_system, water_coords)
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

    ff = get_ff_am1ccc()

    cache_path = "test_smc_cache.pkl"  # TODO: store this somewhere safer!
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
        ubps, params, masses, coords, box = get_solvent_phase_system(mol, ff)

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

    # # plot histogram using asciiplotlib
    # print("torsion distribution in sample cache")
    # fig = apl.figure()
    # fig.hist(*np.histogram(end_state_torsions, bins=25, range=(-np.pi, np.pi)))
    # fig.show()

    for u, p in zip(ubps, params):
        u.bind(p)

    # propagate with NPTMove
    seed = int(time.time())

    mover = NPTMove(ubps, masses, temperature, pressure, n_steps, seed)
    potential_energy_fxn = PotentialEnergyFunction(ubps, params)

    return potential_energy_fxn, mover, all_xvbs
