import time

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from timemachine.constants import DEFAULT_ATOM_MAPPING_KWARGS
from timemachine.fe import atom_mapping
from timemachine.fe.mcgregor import MaxVisitsWarning, NoMappingError
from timemachine.fe.utils import plot_atom_mapping_grid, read_sdf

pytestmark = [pytest.mark.nocuda]

hif2a_set = "timemachine/datasets/fep_benchmark/hif2a/ligands.sdf"
eg5_set = "timemachine/datasets/fep_benchmark/eg5/ligands.sdf"

datasets = [
    hif2a_set,
    eg5_set,
]


@pytest.mark.nightly(reason="Slow")
def test_connected_core_with_large_numbers_of_cores():
    """The following tests that for two mols that have a large number of matching
    cores found prior to filtering out the molecules with disconnected cores and incomplete rings.

    Previously this pair could take about 45 minutes to generate a mapping, largely due to removing
    imcomplete ring cores."""

    mol_a = Chem.MolFromMolBlock(
        """CHEMBL3645089
     RDKit          3D

 69 73  0  0  1  0  0  0  0  0999 V2000
   16.0520  -30.0883  -19.2730 H   0  0  0  0  0  0  0  0  0  0  0  0
   16.8905  -27.8674  -19.9786 H   0  0  0  0  0  0  0  0  0  0  0  0
    9.7990  -22.7096  -21.4994 H   0  0  0  0  0  0  0  0  0  0  0  0
   11.1283  -24.6386  -20.8666 H   0  0  0  0  0  0  0  0  0  0  0  0
   17.7580  -23.1584  -20.0406 H   0  0  0  0  0  0  0  0  0  0  0  0
   16.3894  -22.0537  -21.7426 H   0  0  0  0  0  0  0  0  0  0  0  0
   13.8126  -25.4606  -21.9537 H   0  0  0  0  0  0  0  0  0  0  0  0
    7.5455  -26.7156  -19.6263 H   0  0  0  0  0  0  0  0  0  0  0  0
   13.6125  -30.4066  -18.7837 H   0  0  0  0  0  0  0  0  0  0  0  0
    6.4147  -22.6157  -22.1654 H   0  0  0  0  0  0  0  0  0  0  0  0
    7.7818  -21.6172  -21.7628 H   0  0  0  0  0  0  0  0  0  0  0  0
    5.4859  -25.7846  -19.8796 H   0  0  0  0  0  0  0  0  0  0  0  0
    5.2735  -24.7054  -21.2276 H   0  0  0  0  0  0  0  0  0  0  0  0
    6.1100  -20.7065  -20.4639 H   0  0  0  0  0  0  0  0  0  0  0  0
    6.9745  -21.6761  -19.2968 H   0  0  0  0  0  0  0  0  0  0  0  0
    5.8262  -23.6987  -18.3755 H   0  0  0  0  0  0  0  0  0  0  0  0
    4.2294  -24.0842  -18.9693 H   0  0  0  0  0  0  0  0  0  0  0  0
    4.4341  -20.4540  -15.1211 H   0  0  0  0  0  0  0  0  0  0  0  0
    4.1173  -22.1766  -14.9112 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.8225  -20.9925  -14.7205 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.8100  -22.5875  -17.4137 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.3214  -21.3352  -16.2847 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.7249  -20.8898  -17.9152 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.1355  -22.0392  -19.2748 H   0  0  0  0  0  0  0  0  0  0  0  0
    4.0251  -20.6007  -19.6181 H   0  0  0  0  0  0  0  0  0  0  0  0
   18.7679  -25.7907  -17.7633 H   0  0  0  0  0  0  0  0  0  0  0  0
   18.9836  -24.4749  -18.8632 H   0  0  0  0  0  0  0  0  0  0  0  0
   18.4388  -23.7908  -16.4012 H   0  0  0  0  0  0  0  0  0  0  0  0
    9.4874  -27.7302  -19.4019 H   0  0  0  0  0  0  0  0  0  0  0  0
   15.3585  -29.2668  -19.3889 C   0  0  0  0  0  0  0  0  0  0  0  0
   15.8272  -28.0068  -19.8132 C   0  0  0  0  0  0  0  0  0  0  0  0
    9.2976  -23.5767  -21.0948 C   0  0  0  0  0  0  0  0  0  0  0  0
   10.0630  -24.6960  -20.7200 C   0  0  0  0  0  0  0  0  0  0  0  0
   16.8994  -23.6879  -20.4212 C   0  0  0  0  0  0  0  0  0  0  0  0
   16.1253  -23.0424  -21.3981 C   0  0  0  0  0  0  0  0  0  0  0  0
   14.6660  -24.9847  -21.4947 C   0  0  0  0  0  0  0  0  0  0  0  0
    8.0343  -25.8428  -20.0356 C   0  0  0  0  0  0  0  0  0  0  0  0
   14.0264  -29.4807  -19.1627 C   0  0  0  0  0  0  0  0  0  0  0  0
   14.9464  -26.9716  -20.0563 C   0  0  0  0  0  0  0  0  0  0  0  0
   15.4093  -25.6407  -20.4989 C   0  0  0  0  0  0  0  0  0  0  0  0
    7.8959  -23.5879  -20.9521 C   0  0  0  0  0  0  0  0  0  0  0  0
    7.2602  -24.7322  -20.4223 C   0  0  0  0  0  0  0  0  0  0  0  0
    9.4335  -25.8445  -20.1881 C   0  0  0  0  0  0  0  0  0  0  0  0
   16.5397  -24.9781  -19.9502 C   0  0  0  0  0  0  0  0  0  0  0  0
   15.0064  -23.7035  -21.9199 C   0  0  0  0  0  0  0  0  0  0  0  0
   13.5492  -27.2205  -19.8930 C   0  0  0  0  0  0  0  0  0  0  0  0
   11.4503  -27.3513  -19.7295 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.4202  -21.3519  -17.6391 C   0  0  0  0  0  0  0  0  0  0  0  0
    7.0982  -22.3619  -21.3534 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.7555  -24.7953  -20.2531 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.3340  -21.7347  -20.1776 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.2167  -23.7404  -19.2804 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.7049  -21.2462  -15.3040 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.0041  -21.5603  -17.1012 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.1114  -21.5772  -19.1419 C   0  0  0  0  0  0  0  0  0  0  0  0
   18.2412  -24.9640  -18.2326 C   0  0  0  0  0  0  0  0  0  0  0  0
   17.6782  -24.0232  -17.1438 C   0  0  0  0  0  0  0  0  0  0  0  0
   12.4498  -26.4940  -20.0505 N   0  0  0  0  0  0  0  0  0  0  0  0
   11.8127  -28.5638  -19.3450 N   0  0  0  0  0  0  0  0  0  0  0  0
   13.1579  -28.4895  -19.4464 N   0  0  0  0  0  0  0  0  0  0  0  0
    5.0646  -22.4207  -19.9054 N   0  0  1  0  0  0  0  0  0  0  0  0
   10.0919  -27.0171  -19.7912 N   0  0  0  0  0  0  0  0  0  0  0  0
    3.4139  -21.3934  -16.7341 N   0  0  0  0  0  0  0  0  0  0  0  0
    5.5801  -21.1166  -17.3007 O   0  0  0  0  0  0  0  0  0  0  0  0
   17.2382  -25.6368  -18.9689 O   0  0  0  0  0  0  0  0  0  0  0  0
   17.2582  -22.8587  -17.6347 F   0  0  0  0  0  0  0  0  0  0  0  0
   16.6960  -24.5926  -16.4580 F   0  0  0  0  0  0  0  0  0  0  0  0
   14.0385  -22.9925  -23.1208 Cl  0  0  0  0  0  0  0  0  0  0  0  0
    4.6384  -22.5815  -20.7956 H   0  0  0  0  0  0  0  0  0  0  0  0
  1 30  1  0
  2 31  1  0
  3 32  1  0
  4 33  1  0
  5 34  1  0
  6 35  1  0
  7 36  1  0
  8 37  1  0
  9 38  1  0
 10 49  1  0
 11 49  1  0
 12 50  1  0
 13 50  1  0
 14 51  1  0
 15 51  1  0
 16 52  1  0
 17 52  1  0
 18 53  1  0
 19 53  1  0
 20 53  1  0
 21 54  1  0
 22 54  1  0
 23 54  1  0
 24 55  1  0
 25 55  1  0
 26 56  1  0
 27 56  1  0
 28 57  1  0
 29 62  1  0
 30 31  1  0
 30 38  2  0
 31 39  2  0
 32 33  2  0
 32 41  1  0
 33 43  1  0
 34 35  2  0
 34 44  1  0
 35 45  1  0
 36 40  1  0
 36 45  2  0
 37 42  1  0
 37 43  2  0
 38 60  1  0
 39 40  1  0
 39 46  1  0
 40 44  2  0
 41 42  2  0
 41 49  1  0
 42 50  1  0
 43 62  1  0
 44 65  1  0
 45 68  1  0
 46 58  2  0
 46 60  1  0
 47 58  1  0
 47 59  2  0
 47 62  1  0
 48 55  1  0
 48 63  1  0
 48 64  2  0
 49 51  1  0
 50 52  1  0
 51 61  1  0
 52 61  1  0
 53 63  1  0
 54 63  1  0
 55 61  1  0
 56 57  1  0
 56 65  1  0
 57 66  1  0
 57 67  1  0
 59 60  1  0
 61 69  1  6
M  CHG  1  61   1
M  END
$$$$""",
        removeHs=False,
    )

    mol_b = Chem.MolFromMolBlock(
        """CHEMBL3642347
     RDKit          3D

 71 75  0  0  1  0  0  0  0  0999 V2000
   16.0526  -22.0508  -16.7423 H   0  0  0  0  0  0  0  0  0  0  0  0
   16.0564  -30.0836  -19.2752 H   0  0  0  0  0  0  0  0  0  0  0  0
   16.0658  -24.3724  -17.5803 H   0  0  0  0  0  0  0  0  0  0  0  0
   16.8973  -27.8858  -20.0025 H   0  0  0  0  0  0  0  0  0  0  0  0
    7.5163  -26.5291  -19.6315 H   0  0  0  0  0  0  0  0  0  0  0  0
   11.2546  -24.5771  -20.6469 H   0  0  0  0  0  0  0  0  0  0  0  0
    6.3536  -24.4285  -20.0170 H   0  0  0  0  0  0  0  0  0  0  0  0
   10.0923  -22.5184  -21.0223 H   0  0  0  0  0  0  0  0  0  0  0  0
   16.7996  -20.2033  -18.2796 H   0  0  0  0  0  0  0  0  0  0  0  0
   13.6181  -30.4092  -18.7935 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.1238  -21.2003  -15.9837 H   0  0  0  0  0  0  0  0  0  0  0  0
    4.1146  -22.1989  -14.9229 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.6286  -24.0515  -16.2947 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.6481  -22.9968  -17.3069 H   0  0  0  0  0  0  0  0  0  0  0  0
    5.2419  -20.2229  -16.4566 H   0  0  0  0  0  0  0  0  0  0  0  0
    6.0956  -21.7403  -16.1166 H   0  0  0  0  0  0  0  0  0  0  0  0
    5.5668  -23.7121  -17.5780 H   0  0  0  0  0  0  0  0  0  0  0  0
    4.3788  -23.4912  -18.8670 H   0  0  0  0  0  0  0  0  0  0  0  0
   19.0642  -24.0591  -21.0501 H   0  0  0  0  0  0  0  0  0  0  0  0
   18.7345  -24.1070  -22.7812 H   0  0  0  0  0  0  0  0  0  0  0  0
   19.5069  -22.6908  -22.0492 H   0  0  0  0  0  0  0  0  0  0  0  0
   16.1404  -19.7412  -23.4521 H   0  0  0  0  0  0  0  0  0  0  0  0
   15.8464  -20.2097  -21.7749 H   0  0  0  0  0  0  0  0  0  0  0  0
   17.5196  -19.9016  -22.3173 H   0  0  0  0  0  0  0  0  0  0  0  0
   17.0679  -25.8484  -19.2429 H   0  0  0  0  0  0  0  0  0  0  0  0
   17.3759  -25.4132  -20.8886 H   0  0  0  0  0  0  0  0  0  0  0  0
    6.8064  -20.5341  -18.4662 H   0  0  0  0  0  0  0  0  0  0  0  0
    7.1380  -22.2042  -18.2292 H   0  0  0  0  0  0  0  0  0  0  0  0
    5.5751  -22.5140  -20.4533 H   0  0  0  0  0  0  0  0  0  0  0  0
    5.9193  -20.8342  -20.6426 H   0  0  0  0  0  0  0  0  0  0  0  0
    9.4868  -27.7026  -19.4198 H   0  0  0  0  0  0  0  0  0  0  0  0
   14.6444  -25.0100  -20.4847 H   0  0  0  0  0  0  0  0  0  0  0  0
   16.3541  -22.2550  -17.7568 C   0  0  0  0  0  0  0  0  0  0  0  0
   15.3650  -29.2637  -19.3907 C   0  0  0  0  0  0  0  0  0  0  0  0
   16.3563  -23.5680  -18.2307 C   0  0  0  0  0  0  0  0  0  0  0  0
   15.8379  -28.0067  -19.8080 C   0  0  0  0  0  0  0  0  0  0  0  0
    8.0911  -25.6611  -19.9310 C   0  0  0  0  0  0  0  0  0  0  0  0
   10.1933  -24.5937  -20.4868 C   0  0  0  0  0  0  0  0  0  0  0  0
    7.4207  -24.4316  -20.1442 C   0  0  0  0  0  0  0  0  0  0  0  0
    9.5224  -23.3810  -20.7194 C   0  0  0  0  0  0  0  0  0  0  0  0
   16.7529  -21.2246  -18.6168 C   0  0  0  0  0  0  0  0  0  0  0  0
   14.0351  -29.4815  -19.1646 C   0  0  0  0  0  0  0  0  0  0  0  0
   16.7021  -23.8106  -19.5740 C   0  0  0  0  0  0  0  0  0  0  0  0
    9.4921  -25.7616  -20.0849 C   0  0  0  0  0  0  0  0  0  0  0  0
   14.9547  -26.9862  -20.0659 C   0  0  0  0  0  0  0  0  0  0  0  0
    8.1367  -23.2746  -20.5310 C   0  0  0  0  0  0  0  0  0  0  0  0
   13.5594  -27.2360  -19.9195 C   0  0  0  0  0  0  0  0  0  0  0  0
   17.0626  -22.7019  -20.3843 C   0  0  0  0  0  0  0  0  0  0  0  0
   11.4624  -27.3574  -19.7574 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.9480  -21.9136  -15.9600 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.6292  -23.1133  -16.8463 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.2047  -21.3058  -16.5796 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.7262  -23.0987  -17.9101 C   0  0  0  0  0  0  0  0  0  0  0  0
   18.7475  -23.4579  -21.9048 C   0  0  0  0  0  0  0  0  0  0  0  0
   16.5435  -20.2936  -22.6090 C   0  0  0  0  0  0  0  0  0  0  0  0
   16.6873  -25.2439  -20.0638 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.4066  -21.5307  -18.6553 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.3014  -21.7503  -20.1814 C   0  0  0  0  0  0  0  0  0  0  0  0
   17.1107  -21.4450  -19.8948 N   0  0  0  0  0  0  0  0  0  0  0  0
   12.4698  -26.5158  -20.1118 N   0  0  0  0  0  0  0  0  0  0  0  0
   11.8204  -28.5689  -19.3595 N   0  0  0  0  0  0  0  0  0  0  0  0
   13.1639  -28.4924  -19.4504 N   0  0  0  0  0  0  0  0  0  0  0  0
    5.1183  -21.6934  -17.9837 N   0  0  0  0  0  0  0  0  0  0  0  0
   10.1041  -26.9997  -19.8054 N   0  0  0  0  0  0  0  0  0  0  0  0
   15.3732  -25.7050  -20.4619 N   0  0  0  0  0  0  0  0  0  0  0  0
   17.4375  -22.8381  -21.7349 N   0  0  0  0  0  0  0  0  0  0  0  0
   17.6666  -22.1414  -24.2073 O   0  0  0  0  0  0  0  0  0  0  0  0
   15.3444  -22.6103  -23.2295 O   0  0  0  0  0  0  0  0  0  0  0  0
    7.5840  -22.0380  -20.7412 O   0  0  0  0  0  0  0  0  0  0  0  0
   16.6964  -22.0504  -23.1015 S   0  0  0  0  0  0  0  0  0  0  0  0
    4.4131  -21.1592  -18.4499 H   0  0  0  0  0  0  0  0  0  0  0  0
  1 33  1  0
  2 34  1  0
  3 35  1  0
  4 36  1  0
  5 37  1  0
  6 38  1  0
  7 39  1  0
  8 40  1  0
  9 41  1  0
 10 42  1  0
 11 50  1  0
 12 50  1  0
 13 51  1  0
 14 51  1  0
 15 52  1  0
 16 52  1  0
 17 53  1  0
 18 53  1  0
 19 54  1  0
 20 54  1  0
 21 54  1  0
 22 55  1  0
 23 55  1  0
 24 55  1  0
 25 56  1  0
 26 56  1  0
 27 57  1  0
 28 57  1  0
 29 58  1  0
 30 58  1  0
 31 64  1  0
 32 65  1  0
 33 35  2  0
 33 41  1  0
 34 36  1  0
 34 42  2  0
 35 43  1  0
 36 45  2  0
 37 39  2  0
 37 44  1  0
 38 40  1  0
 38 44  2  0
 39 46  1  0
 40 46  2  0
 41 59  2  0
 42 62  1  0
 43 48  2  0
 43 56  1  0
 44 64  1  0
 45 47  1  0
 45 65  1  0
 46 69  1  0
 47 60  2  0
 47 62  1  0
 48 59  1  0
 48 66  1  0
 49 60  1  0
 49 61  2  0
 49 64  1  0
 50 51  1  0
 50 52  1  0
 51 53  1  0
 52 63  1  0
 53 63  1  0
 54 66  1  0
 55 70  1  0
 56 65  1  0
 57 58  1  0
 57 63  1  0
 58 69  1  0
 61 62  1  0
 66 70  1  0
 67 70  2  0
 68 70  2  0
 63 71  1  0
M  CHG  1  63   1
M  ENDG
$$$$""",
        removeHs=False,
    )

    start = time.time()
    all_cores = atom_mapping.get_cores(
        mol_a,
        mol_b,
        ring_cutoff=0.1,
        chain_cutoff=0.2,
        max_visits=1e7,
        connected_core=True,
        max_cores=1e6,  # This pair has 350k cores
        enforce_core_core=True,
        ring_matches_ring_only=False,
        enforce_chiral=False,
        disallow_planar_torsion_flips=False,
        min_threshold=0,
    )
    assert len(all_cores) > 0

    # If this takes more then 30 minutes, something is off.
    assert time.time() - start < 1800.0


def get_mol_name(mol) -> str:
    """Return the title for the given mol"""
    return mol.GetProp("_Name")


# hif2a is easy
# eg5 is challenging
# notable outliers for eg5:
# CHEMBL1077227 -> CHEMBL1086410 has 20736 cores of size 56
# CHEMBL1077227 -> CHEMBL1083836 has 14976 cores of size 48
# CHEMBL1086410 -> CHEMBL1083836 has 10752 cores of size 52
# CHEMBL1086410 -> CHEMBL1084935 has 6912 cores of size 60
@pytest.mark.parametrize("filepath", datasets)
@pytest.mark.nightly(reason="Slow")
def test_all_pairs(filepath):
    mols = read_sdf(filepath)
    for idx, mol_a in enumerate(mols):
        for mol_b in mols[idx + 1 :]:
            # print("Processing", get_mol_name(mol_a), "->", get_mol_name(mol_b))
            start_time = time.time()
            all_cores, diagnostics = atom_mapping.get_cores_and_diagnostics(
                mol_a,
                mol_b,
                ring_cutoff=0.12,
                chain_cutoff=0.2,
                max_visits=1e7,  # 10 million max nodes to visit
                connected_core=True,
                max_cores=1000,
                enforce_core_core=True,
                ring_matches_ring_only=False,
                enforce_chiral=False,
                disallow_planar_torsion_flips=False,
                min_threshold=0,
            )
            end_time = time.time()

            # # useful for visualization
            # for core_idx, core in enumerate(all_cores[:1]):
            #     res = plot_atom_mapping_grid(mol_a, mol_b, core, num_rotations=5)
            #     with open(
            #         f"atom_mapping_{get_mol_name(mol_a)}_to_{get_mol_name(mol_b)}_core_{core_idx}.svg", "w"
            #     ) as fh:
            #         fh.write(res)

            # note that this is probably the bottleneck for hif2a
            for core in all_cores:
                # ensure more than half the atoms are mapped
                assert len(core) > mol_a.GetNumAtoms() // 2

            print(
                f"{mol_a.GetProp('_Name')} -> {mol_b.GetProp('_Name')} has {len(all_cores)} cores of size {len(all_cores[0])} | total nodes visited: {diagnostics.total_nodes_visited} | wall clock time: {end_time - start_time:.3f}"
            )


def get_mol_by_name(mols, name):
    for m in mols:
        if get_mol_name(m) == name:
            return m

    assert 0, "Mol not found"


def tuples_to_set(arr):
    res = set()
    for a, b in arr:
        key = (a, b)
        assert key not in res
        res.add(key)
    return res


def assert_cores_are_equal(core_a, core_b):
    core_set_a = tuples_to_set(core_a)
    core_set_b = tuples_to_set(core_b)
    assert core_set_a == core_set_b


def get_all_cores_fzset(all_cores):
    all_cores_fzset = set()
    for core in all_cores:
        all_cores_fzset.add(frozenset(tuples_to_set(core)))
    return all_cores_fzset


def assert_core_sets_are_equal(core_set_a, core_set_b):
    fza = get_all_cores_fzset(core_set_a)
    fzb = get_all_cores_fzset(core_set_b)
    assert fza == fzb


# spot check
def test_linker_map():
    # test that we can map a linker size change when connected_core=False, and enforce_core_core=False
    mol_a = Chem.MolFromMolBlock(
        """
  Mrv2219 11232201352D

 10 11  0  0  0  0            999 V2000
  -12.2008    0.8688    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.4558    0.0842    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.2808    0.0842    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.5357    0.8688    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.8683    1.3538    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.1785    3.3333    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.8460    2.8484    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.5134    3.3333    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.2585    4.1179    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.4335    4.1179    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  5  1  0  0  0  0
  2  3  1  0  0  0  0
  3  4  1  0  0  0  0
  4  5  1  0  0  0  0
  6  7  1  0  0  0  0
  6 10  1  0  0  0  0
  7  8  1  0  0  0  0
  8  9  1  0  0  0  0
  9 10  1  0  0  0  0
  5  7  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )
    mol_b = Chem.MolFromMolBlock(
        """
  Mrv2219 11232201352D

 11 12  0  0  0  0            999 V2000
  -12.2008    0.8688    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.4558    0.0842    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.2808    0.0842    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.5357    0.8688    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.8683    1.3538    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.1785    3.3333    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.8460    2.8484    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.5134    3.3333    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.2585    4.1179    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.4335    4.1179    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.2416    2.1140    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  5  1  0  0  0  0
  2  3  1  0  0  0  0
  3  4  1  0  0  0  0
  4  5  1  0  0  0  0
  6  7  1  0  0  0  0
  6 10  1  0  0  0  0
  7  8  1  0  0  0  0
  8  9  1  0  0  0  0
  9 10  1  0  0  0  0
  7 11  1  0  0  0  0
 11  5  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    all_cores = atom_mapping.get_cores(
        mol_a,
        mol_b,
        ring_cutoff=0.05,
        chain_cutoff=0.05,
        max_visits=1e7,  # 10 million max nodes to visit
        connected_core=False,
        max_cores=1000000,
        enforce_core_core=False,
        ring_matches_ring_only=False,
        enforce_chiral=False,
        disallow_planar_torsion_flips=False,
        min_threshold=0,
    )

    assert len(all_cores) == 1

    assert_cores_are_equal(
        [[6, 6], [4, 4], [9, 9], [8, 8], [7, 7], [5, 5], [3, 3], [2, 2], [1, 1], [0, 0]], all_cores[0]
    )

    # now set connected_core and enforce_core_core to True
    all_cores = atom_mapping.get_cores(
        mol_a,
        mol_b,
        ring_cutoff=0.05,
        chain_cutoff=0.05,
        max_visits=1e7,  # 10 million max nodes to visit
        connected_core=True,
        max_cores=1000000,
        enforce_core_core=True,
        ring_matches_ring_only=False,
        enforce_chiral=False,
        disallow_planar_torsion_flips=False,
        min_threshold=0,
    )

    # 2 possible matches, returned core ordering is fully determined
    # note that we return the larger of the two disconnected components here
    # (the 5 membered ring)
    assert len(all_cores) == 2

    expected_sets = ([[6, 6], [9, 9], [8, 8], [7, 7], [5, 5]], [[4, 4], [3, 3], [2, 2], [1, 1], [0, 0]])

    assert_core_sets_are_equal(expected_sets, all_cores)

    all_cores = atom_mapping.get_cores(
        mol_a,
        mol_b,
        ring_cutoff=0.05,
        chain_cutoff=0.05,
        max_visits=1e7,  # 10 million max nodes to visit
        connected_core=False,
        max_cores=1000000,
        enforce_core_core=True,
        ring_matches_ring_only=False,
        enforce_chiral=False,
        disallow_planar_torsion_flips=False,
        min_threshold=0,
    )

    # 2 possible matches, if we do not allow for connected_core but do
    # require core_core, we have a 9-atom disconnected map, one is a 5-membered ring
    # the other is 4-membered chain. There's 2 allowed maps due to the 2 fold symmetry.
    assert len(all_cores) == 2

    expected_sets = (
        [[6, 6], [9, 9], [8, 8], [7, 7], [5, 5], [3, 3], [2, 2], [1, 1], [0, 0]],
        [[4, 4], [9, 9], [8, 8], [7, 7], [5, 5], [3, 3], [2, 2], [1, 1], [0, 0]],
    )

    assert_core_sets_are_equal(expected_sets, all_cores)


def get_cyclohexanes_different_confs():
    """Two cyclohexane structures that differ enough in conformations to map poorly by MCS with threshold of 2.0"""
    mol_a = Chem.MolFromMolBlock(
        """
 cyclo_1

 18 18  0  0  1  0  0  0  0  0999 V2000
    0.7780    1.1695    0.1292 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.3871   -0.1008    0.2959 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6896    1.3214   -0.2192 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.5088    0.0613    0.0503 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7859   -1.2096   -0.4242 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.6085   -1.3920    0.2133 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.1105    2.1590    0.3356 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7549    1.5841   -1.2762 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.6874   -0.0047    1.1175 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.4858    0.1560   -0.4244 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6117   -1.0357   -1.4891 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.1610   -2.0015   -0.5036 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.5422   -1.8809    1.1852 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.4054   -2.0928   -0.2686 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.3677    1.7499   -0.5802 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.9940    1.7789    1.0067 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.9567   -0.0955    1.2253 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.2449   -0.1670   -0.3734 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1  3  1  0
  1 15  1  0
  1 16  1  0
  2  6  1  0
  2 17  1  0
  2 18  1  0
  3  4  1  0
  3  7  1  0
  3  8  1  0
  4  5  1  0
  4  9  1  0
  4 10  1  0
  5  6  1  0
  5 11  1  0
  5 14  1  0
  6 12  1  0
  6 13  1  0
M  END
$$$$""",
        removeHs=False,
    )

    mol_b = Chem.MolFromMolBlock(
        """
 cyclo_2

 18 18  0  0  1  0  0  0  0  0999 V2000
    0.7953    1.1614    0.0469 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.3031   -0.0613    0.5362 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6118    1.1962   -0.5144 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.9934   -0.1785   -1.1042 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6439   -1.3144   -0.1494 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.4262   -1.2251    0.6719 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2949    1.4641    0.2937 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6968    1.9715   -1.2775 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.0662   -0.1837   -1.3042 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.4970   -0.3613   -2.0575 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.6428   -1.9811    1.4121 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2637   -2.1987   -0.1345 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.4850    1.5611   -0.6965 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.8877    1.9212    0.8230 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.8010    0.1189    1.4889 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.1753   -0.3430   -0.0537 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2711   -0.8618    0.6186 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.1781   -0.6848    1.4006 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1  3  1  0
  1 13  1  0
  1 14  1  0
  2  6  1  0
  2 15  1  0
  2 16  1  0
  3  4  1  0
  3  7  1  0
  3  8  1  0
  4  5  1  0
  4  9  1  0
  4 10  1  0
  5  6  1  0
  5 12  1  0
  5 17  1  0
  6 11  1  0
  6 18  1  0
M  END
$$$$""",
        removeHs=False,
    )
    return mol_a, mol_b


def test_hif2a_failure():
    # special failure with error message:
    # pred_sgg_a = a_cycles[a] == sg_a_cycles[a], KeyError: 18
    mols = Chem.SDMolSupplier(hif2a_set, removeHs=False)
    mols = [m for m in mols]
    mol_a = get_mol_by_name(mols, "7a")
    mol_b = get_mol_by_name(mols, "224")
    all_cores = atom_mapping.get_cores(
        mol_a,
        mol_b,
        ring_cutoff=0.12,
        chain_cutoff=0.2,
        max_visits=1e7,
        connected_core=True,
        max_cores=1e6,
        enforce_core_core=True,
        ring_matches_ring_only=False,
        enforce_chiral=False,
        disallow_planar_torsion_flips=False,
        min_threshold=0,
    )

    expected_core = np.array(
        [
            [22, 15],
            [19, 12],
            [3, 3],
            [2, 2],
            [1, 1],
            [18, 24],
            [12, 21],
            [11, 11],
            [9, 9],
            [8, 8],
            [7, 7],
            [6, 6],
            [5, 5],
            [4, 4],
            [13, 22],
            [10, 10],
            [0, 0],
            [35, 33],
            [33, 32],
            [32, 31],
            [31, 30],
            [30, 29],
            [25, 28],
            [27, 27],
            [26, 26],
            [24, 20],
            [23, 19],
            [36, 18],
            [28, 17],
            [29, 16],
            [21, 14],
            [20, 13],
        ]
    )

    assert_cores_are_equal(all_cores[0], expected_core)
    # for core_idx, core in enumerate(all_cores[:1]):
    #     res = plot_atom_mapping_grid(mol_a, mol_b, core, num_rotations=5)
    #     with open(f"atom_mapping_0_to_1_core_{core_idx}.svg", "w") as fh:
    #         fh.write(res)


def test_cyclohexane_stereo():
    # test that cyclohexane in two different conformations has a core alignment that is stereo correct. Note that this needs a
    # larger than typical cutoff.
    mol_a, mol_b = get_cyclohexanes_different_confs()
    all_cores = atom_mapping.get_cores(
        mol_a,
        mol_b,
        ring_cutoff=0.15,
        chain_cutoff=0.30,
        max_visits=1e6,
        connected_core=True,
        max_cores=100000,
        enforce_core_core=True,
        ring_matches_ring_only=True,
        enforce_chiral=True,
        disallow_planar_torsion_flips=False,
        min_threshold=0,
    )

    for core_idx, core in enumerate(all_cores[:1]):
        res = plot_atom_mapping_grid(mol_a, mol_b, core, num_rotations=5)
        with open(f"atom_mapping_0_to_1_core_{core_idx}.svg", "w") as fh:
            fh.write(res)

    # 1-indexed
    expected_core = np.array(
        [
            [1, 1],  # C
            [2, 2],  # C
            [3, 3],  # C
            [4, 4],  # C
            [5, 5],  # C
            [6, 6],  # C
            [16, 14],  # C1H
            [15, 13],  # C1H
            [17, 15],  # C2H
            [18, 16],  # C2H
            [7, 7],  # C3H
            [8, 8],  # C3H
            [9, 9],  # C4H
            [10, 10],  # C4H
            [14, 17],  # C5H
            [11, 12],  # C5H
            [13, 18],  # C6H
            [12, 11],  # C6H
        ]
    )

    # 0-indexed
    expected_core -= 1

    all_cores_fzset = get_all_cores_fzset(all_cores)
    assert tuples_to_set(expected_core) in all_cores_fzset

    assert len(all_cores) == 1


def test_chiral_atom_map():
    mol_a = Chem.AddHs(Chem.MolFromSmiles("C"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("C"))

    AllChem.EmbedMolecule(mol_a, randomSeed=0)
    AllChem.EmbedMolecule(mol_b, randomSeed=0)

    core_kwargs = dict(
        ring_cutoff=np.inf,
        chain_cutoff=np.inf,
        max_visits=1e7,
        connected_core=True,
        max_cores=1e6,
        enforce_core_core=True,
        disallow_planar_torsion_flips=False,
        ring_matches_ring_only=True,
        min_threshold=0,
    )

    chiral_aware_cores = atom_mapping.get_cores(mol_a, mol_b, enforce_chiral=True, **core_kwargs)
    chiral_oblivious_cores = atom_mapping.get_cores(mol_a, mol_b, enforce_chiral=False, **core_kwargs)

    assert len(chiral_oblivious_cores) == 4 * 3 * 2 * 1, "expected all hydrogen permutations to be valid"
    assert len(chiral_aware_cores) == (len(chiral_oblivious_cores) // 2), "expected only rotations to be valid"

    for key, val in chiral_aware_cores[0]:
        assert key == val, "expected first core to be identity map"
    assert len(chiral_aware_cores[0]) == 5


@pytest.mark.parametrize(
    "ring_matches_ring_only",
    [
        pytest.param(False, marks=pytest.mark.xfail(strict=True)),
        pytest.param(True),
    ],
)
def test_ring_matches_ring_only(ring_matches_ring_only):
    mol_a = Chem.AddHs(Chem.MolFromSmiles("C(c1ccc1)"))
    AllChem.EmbedMolecule(mol_a, randomSeed=3)

    mol_b = Chem.AddHs(Chem.MolFromSmiles("C(c1ccccc1)"))
    AllChem.EmbedMolecule(mol_b, randomSeed=3)

    core_kwargs = dict(
        ring_cutoff=0.15,
        chain_cutoff=0.2,
        max_visits=1e7,
        connected_core=True,
        max_cores=1e6,
        enforce_core_core=False,
        enforce_chiral=False,
        disallow_planar_torsion_flips=False,
        min_threshold=0,
    )

    cores = atom_mapping.get_cores(mol_a, mol_b, ring_matches_ring_only=ring_matches_ring_only, **core_kwargs)

    assert cores

    # should not map ring atoms to non-ring atoms and vice-versa
    for core in cores:
        for idx_a, idx_b in core:
            assert mol_a.GetAtomWithIdx(int(idx_a)).IsInRing() == mol_b.GetAtomWithIdx(int(idx_b)).IsInRing()

    # should map all ring atoms in mol_a
    assert all(len([() for idx_a in core[:, 0] if mol_a.GetAtomWithIdx(int(idx_a)).IsInRing()]) == 4 for core in cores)


def test_max_visits_warning():
    mol_a, mol_b = get_cyclohexanes_different_confs()
    core_kwargs = dict(
        ring_cutoff=0.1,
        chain_cutoff=0.2,
        connected_core=False,
        max_cores=1000,
        enforce_core_core=True,
        ring_matches_ring_only=False,
        enforce_chiral=True,
        disallow_planar_torsion_flips=False,
        min_threshold=0,
    )
    cores = atom_mapping.get_cores(mol_a, mol_b, **core_kwargs, max_visits=10000)
    assert len(cores) > 0

    with pytest.warns(MaxVisitsWarning, match="Reached max number of visits/cores: 0 cores with 2 nodes visited"):
        with pytest.raises(NoMappingError):
            atom_mapping.get_cores(mol_a, mol_b, **core_kwargs, max_visits=1)


def test_max_cores_warning():
    mol_a, mol_b = get_cyclohexanes_different_confs()
    core_kwargs = dict(
        ring_cutoff=0.1,
        chain_cutoff=0.2,
        connected_core=False,
        enforce_core_core=True,
        ring_matches_ring_only=False,
        enforce_chiral=True,
        disallow_planar_torsion_flips=False,
        min_threshold=0,
        max_visits=1e7,
    )
    with pytest.warns(MaxVisitsWarning, match="Reached max number of visits/cores: 1 cores"):
        all_cores = atom_mapping.get_cores(mol_a, mol_b, **core_kwargs, max_cores=1)
        assert len(all_cores) == 1


def test_min_threshold():
    mol_a, mol_b = get_cyclohexanes_different_confs()
    core_kwargs = dict(
        ring_cutoff=0.1,
        chain_cutoff=0.2,
        connected_core=False,
        max_cores=1000,
        enforce_core_core=True,
        ring_matches_ring_only=False,
        enforce_chiral=True,
        disallow_planar_torsion_flips=False,
        min_threshold=mol_a.GetNumAtoms(),
    )

    with pytest.raises(NoMappingError, match="Unable to find mapping with at least 18 edges"):
        atom_mapping.get_cores(mol_a, mol_b, **core_kwargs, max_visits=10000)


def test_get_cores_and_diagnostics():
    mols = read_sdf(hif2a_set)
    n_pairs = 30
    random_pair_idxs = np.random.default_rng(2024).choice(len(mols), size=(n_pairs, 2))

    for i_a, i_b in random_pair_idxs:
        mol_a = mols[i_a]
        mol_b = mols[i_b]
        all_cores, diagnostics = atom_mapping.get_cores_and_diagnostics(mol_a, mol_b, **DEFAULT_ATOM_MAPPING_KWARGS)

        assert len(all_cores) > 0
        assert all(len(core) == len(all_cores[0]) for core in all_cores)

        assert diagnostics.core_size >= len(all_cores[0])
        assert diagnostics.num_cores >= len(all_cores)
        assert (
            diagnostics.total_nodes_visited >= diagnostics.core_size
        )  # must visit at least one node per atom pair in core
