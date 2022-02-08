# adapted from https://raw.githubusercontent.com/proteneer/timemachine/fa751c8e3c6ff51601d4ea1a58d4c6b7e35c4f19/tests/test_smc.py

import jax

jax.config.update("jax_enable_x64", True)

from rdkit import Chem
import numpy as np

from timemachine.fe.absolute_hydration import setup_absolute_hydration_with_endpoint_samples


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


def construct_biphenyl_test_system(n_steps=1000):
    mol, torsion_idxs = get_biphenyl()

    return setup_absolute_hydration_with_endpoint_samples(mol, n_steps)
