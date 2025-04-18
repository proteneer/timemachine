from typing import Any

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem

RDKitMol = Any  # Chem.Mol does not work as a type annotation


def get_biphenyl() -> tuple[RDKitMol, NDArray[np.int64]]:
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


def get_triphenyl() -> tuple[RDKitMol, NDArray[np.int64]]:
    MOL_SDF = """
  Mrv2118 11122114533D

 32 34  0  0  0  0            999 V2000
   -1.3718    1.5451   -0.8596 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.6327    1.3077   -1.4412 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.7267    0.8224   -2.7557 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.5588    0.5772   -3.4951 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.2950    0.8374   -2.9324 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.1818    1.3273   -1.6037 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.7226    3.0891    0.0122 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.4679    2.8681   -0.5826 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.0992    1.5698   -1.0177 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.0275    0.5082   -0.8547 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.2999    0.7237   -0.2628 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.6325    2.0301    0.1770 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.8984   -1.4983    0.6598 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.2244   -0.3572   -0.1212 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.4813   -0.3265   -0.7817 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.3913   -1.3923   -0.6378 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.0526   -2.5135    0.1372 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.8034   -2.5702    0.7752 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.3360    1.9521    0.3835 F   0  0  0  0  0  0  0  0  0  0  0  0
   -3.4860    1.4820   -0.9075 H   0  0  0  0  0  0  0  0  0  0  0  0
   -3.6432    0.6452   -3.1720 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.6341    0.2109   -4.4468 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.7605    0.6158   -3.6747 F   0  0  0  0  0  0  0  0  0  0  0  0
    2.9764    4.0316    0.3287 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.6667    3.8942   -0.7253 F   0  0  0  0  0  0  0  0  0  0  0  0
    1.7034   -0.6954   -1.2588 F   0  0  0  0  0  0  0  0  0  0  0  0
    4.7801    2.2904    0.7517 F   0  0  0  0  0  0  0  0  0  0  0  0
    2.7580   -1.5917    1.2964 F   0  0  0  0  0  0  0  0  0  0  0  0
    5.8246    0.6739   -1.5522 F   0  0  0  0  0  0  0  0  0  0  0  0
    7.3007   -1.3578   -1.1014 H   0  0  0  0  0  0  0  0  0  0  0  0
    6.7129   -3.2875    0.2339 H   0  0  0  0  0  0  0  0  0  0  0  0
    4.5578   -3.3955    1.3267 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0  0  0  0
  2  3  1  0  0  0  0
  3  4  2  0  0  0  0
  4  5  1  0  0  0  0
  7  8  2  0  0  0  0
  8  9  1  0  0  0  0
  9 10  2  0  0  0  0
 10 11  1  0  0  0  0
 11 12  2  0  0  0  0
 12  7  1  0  0  0  0
 15 16  2  0  0  0  0
 16 17  1  0  0  0  0
 17 18  2  0  0  0  0
 18 13  1  0  0  0  0
  5  6  2  0  0  0  0
  6  1  1  0  0  0  0
  9  6  1  0  0  0  0
 13 14  2  0  0  0  0
 14 15  1  0  0  0  0
 11 14  1  0  0  0  0
  2 20  1  0  0  0  0
  3 21  1  0  0  0  0
  4 22  1  0  0  0  0
  7 24  1  0  0  0  0
 16 30  1  0  0  0  0
 17 31  1  0  0  0  0
 18 32  1  0  0  0  0
 15 29  1  0  0  0  0
 13 28  1  0  0  0  0
  5 23  1  0  0  0  0
  1 19  1  0  0  0  0
 12 27  1  0  0  0  0
 10 26  1  0  0  0  0
  8 25  1  0  0  0  0
M  END
$$$$"""
    mol = Chem.MolFromMolBlock(MOL_SDF, removeHs=False)

    # 1-indexed torsions = [5,6,9,10],[10,11,14,15]
    torsion_idxs = np.array([[4, 5, 8, 9], [9, 10, 13, 14]])

    return mol, torsion_idxs
