import matplotlib.pyplot as plt
import numpy as np

from timemachine.constants import DEFAULT_ATOM_MAPPING_KWARGS
from timemachine.fe.atom_decouple import estimate_volumes_along_schedule
from timemachine.fe.atom_mapping import get_cores
from timemachine.fe.utils import read_sdf
from timemachine.ff import Forcefield


def test_atom_by_atom_decouple():
    mols = read_sdf("tests/data/benzene_subs.sdf")

    benzene_mono_sub_top = mols[0]
    benzene_mono_sub_bot = mols[1]
    benzene_di_both = mols[2]
    benzene_no_sub = mols[3]

    # for each pair of compounds, optimal transformation should induce a roughly monotone change in volume
    # without maximas or minimias along the path

    # 1. alternate deletion/insertion of different parts of the R-group (maintain constant volume)
    # 2. smoothly increase the volume without extrema
    # 3. manual core mapping, only mapping the benzene ring, (should maintain constaint volume)
    # 4. manual core mapping, only mapping the benzene ring, smoothly increase the volume without extrema
    pairs = [
        (benzene_mono_sub_top, benzene_mono_sub_bot),
        (benzene_no_sub, benzene_mono_sub_top),
        (benzene_mono_sub_top, benzene_mono_sub_top),
        (benzene_mono_sub_top, benzene_di_both),
    ]

    cores = []

    for mol_a, mol_b in pairs[:2]:
        cores.append(get_cores(mol_a, mol_b, **DEFAULT_ATOM_MAPPING_KWARGS)[0])

    # for the last transformation of two identical molecules,
    # we map the core benzene but don't map the side piece
    cores.append(
        np.array(
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
                [5, 5],
                [11, 11],
                [12, 12],
                [13, 13],
                [14, 14],
                [26, 26],
            ]
        )
    )

    cores.append(
        np.array(
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
                [5, 5],
                [11, 16],
                [12, 17],
                [13, 18],
                [14, 19],
            ]
        )
    )

    lamb_schedule = np.linspace(0, 1.0, 24)
    lamb_idxs = np.arange(len(lamb_schedule))
    ff = Forcefield.load_default()

    for (mol_a, mol_b), core in zip(pairs, cores):
        label = mol_a.GetProp("_Name") + " -> " + mol_b.GetProp("_Name")
        print("processing", label)
        vols = estimate_volumes_along_schedule(mol_a, mol_b, core, ff, lamb_schedule)

        plt.plot(lamb_idxs, vols, label=label)

    plt.xlabel("lamb_idx")
    plt.ylabel("volume")
    plt.legend()
    plt.show()
