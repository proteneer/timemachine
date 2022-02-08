# adapted from https://raw.githubusercontent.com/proteneer/timemachine/fa751c8e3c6ff51601d4ea1a58d4c6b7e35c4f19/tests/test_smc.py

import jax

from timemachine.md.enhanced import load_or_pregenerate_samples

jax.config.update("jax_enable_x64", True)

import time
from pathlib import Path

from rdkit import Chem
import numpy as np

import timemachine
from timemachine.constants import BOLTZ

from timemachine.fe import functional
from timemachine.fe.absolute_hydration import (
    generate_endstate_samples,
)

from timemachine.ff import Forcefield
from timemachine.ff.handlers.deserialize import deserialize_handlers

from timemachine.md import enhanced
from timemachine.md.moves import NPTMove

# (ytz): useful for visualization, so please leave this comment here!
# import asciiplotlib as apl

temperature = 300.0
pressure = 1.0

kBT = BOLTZ * temperature


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
    path_to_ff = tm_path / "timemachine/ff/params/smirnoff_1_1_0_ccc.py"
    with open(path_to_ff, "r") as f:
        ff_handlers = deserialize_handlers(f.read())
    ff = Forcefield(ff_handlers)
    return ff


mol, torsion_idxs = get_biphenyl()

ligand_masses = np.array([a.GetMass() for a in mol.GetAtoms()])
num_ligand_atoms = len(ligand_masses)


def bind_potentials(ubps, params):
    """modifies ubps in-place"""
    for u, p in zip(ubps, params):
        u.bind(p)


def construct_potential(ubps, params):
    U_fn = functional.construct_differentiable_interface_fast(ubps, params)

    def potential(xvb, lam):
        return U_fn(xvb.coords, params, xvb.box, lam)

    return potential


def construct_mover(ubps, masses, n_steps):
    seed = int(time.time())  # TODO: why overwrite?
    mover = NPTMove(ubps, None, masses, temperature, pressure, n_steps, seed)

    return mover


def construct_biphenyl_test_system(n_steps=1000):
    """
    Generate samples from the equilibrium distribution at lambda=1

    Return:
    * reduced_potential
    * mover
    * initial_samples
    """

    seed = 2022
    np.random.seed(seed)

    # set up potentials
    ff = get_ff_am1ccc()
    ubps, params, masses, _, _ = enhanced.get_solvent_phase_system(mol, ff)
    potential_fxn = construct_potential(ubps, params)

    def reduced_potential_fxn(xvb, lam):
        return potential_fxn(xvb, lam) / kBT

    bind_potentials(ubps, params)

    # set up npt mover
    npt_mover = construct_mover(ubps, masses, n_steps)

    # combine solvent and ligand samples
    solvent_xvbs, ligand_samples, ligand_log_weights = load_or_pregenerate_samples(
        mol, ff, seed, temperature=temperature, pressure=pressure
    )
    n_endstate_samples = 5000  # TODO: expose this parameter?
    all_xvbs = generate_endstate_samples(
        n_endstate_samples, solvent_xvbs, ligand_samples, ligand_log_weights, num_ligand_atoms
    )

    return reduced_potential_fxn, npt_mover, all_xvbs
