# Utility functions to help assign and identify local geometry points

from enum import Enum
from typing import List, Optional

import numpy as np
from rdkit import Chem

from timemachine.fe import topology
from timemachine.ff import Forcefield


class LocalGeometry(Enum):
    G0_ION = 0
    G1_TERMINAL = 1  # R-X
    G2_KINK = 2  # R-X-H
    G2_LINEAR = 3  # R-X#N
    G3_PLANAR = 4  # R-X(=O)O
    G3_PYRAMIDAL = 5  # R-X(-H)H
    G4_TETRAHEDRAL = 6  # R-X(-H)(-H)H


def bond_idxs_to_adj_list(num_atoms, bond_idxs):
    """
    Convert tuples of bond idxs to a neighbor list encoded as
    a list of lists. The bonds are interpeted as undirected.

    Parameters
    ----------
    num_atoms: int
        Number of atoms

    bond_idxs: list of 2-tuple
        Pairs of atoms

    Returns
    -------
    list of list of int
        Neighorlist

    """
    cmat = np.zeros((num_atoms, num_atoms))
    for i, j in bond_idxs:
        cmat[i][j] = 1
        cmat[j][i] = 1

    nblist = []
    for i in range(num_atoms):
        nbs = []
        for j in range(num_atoms):
            if cmat[i][j]:
                nbs.append(j)
        nblist.append(nbs)

    return nblist


def label_stereo(
    num_atoms,
    bond_idxs,
    angle_idxs,
    angle_params,
    improper_idxs,
    improper_params,
):
    # list of list representation
    nblist = bond_idxs_to_adj_list(num_atoms, bond_idxs)
    atom_geometries = []
    for atom_idx, atom_nbs in enumerate(nblist):
        if len(atom_nbs) == 4:
            atom_geometries.append(LocalGeometry.G4_TETRAHEDRAL)
        elif len(atom_nbs) == 3:
            # check for impropers
            local_geometry = LocalGeometry.G3_PYRAMIDAL
            # (tbd): should we also check propers as well?
            # impropers are centered around the first atom.
            for (i, _, _, _), (k, p, n) in zip(improper_idxs, improper_params):
                if i == atom_idx and k > 0.0:
                    local_geometry = LocalGeometry.G3_PLANAR
                    break

            atom_geometries.append(local_geometry)
            # atom_stereo_flags.append(is_stereo)
        elif len(atom_nbs) == 2:
            # check angle terms:
            local_geometry = LocalGeometry.G2_KINK
            for (i, j, k), (_, angle) in zip(angle_idxs, angle_params):
                if abs(angle) < 0.05:
                    assert 0
                ii, kk = atom_nbs[0], atom_nbs[1]
                if j == atom_idx:
                    if (i, k) == (ii, kk) or (i, k) == (kk, ii):
                        if abs(angle - np.pi) < 0.05:
                            local_geometry = LocalGeometry.G2_LINEAR
                            break
            atom_geometries.append(local_geometry)
        elif len(atom_nbs) == 1:
            atom_geometries.append(LocalGeometry.G1_TERMINAL)
        elif len(atom_nbs) == 0:
            # dummy atom
            atom_geometries.append(None)
        else:
            assert 0, "Valency higher than 4 not supported."

    return atom_geometries


def classify_geometry(
    mol: Chem.Mol, ff: Optional[Forcefield] = None, core: Optional[List[int]] = None
) -> List[LocalGeometry]:
    """
    Identify the local geometry of the molecule. This currently uses a heuristic by inspecting
    the bonded forcefield indices and parameters. Tetrahedral geometries have four neighbors,
    pyramidal geometries have three neighbors but lack an improper term, and planar geometries have an improper
    term. Kink and linear geometries have two neighbors but differ in the angle term (latter is set to pi).

    Ideally, we should be generating this from gas-phase simulations of the real forcefield. Currently, 3D
    coordinates are not required, but this may change in the future.

    Parameters
    ----------
    mol: Chem.Mol
        Input molecule.

    core: List[Int] or None
        Core indices, if None then all atoms are considered to be in the core.

    Returns
    -------
    List[LocalGeometry]
        List of per atom geometries. Dummy atoms are None

    """
    if ff is None:
        ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    bt = topology.BaseTopology(mol, ff)

    bond_params, hb = bt.parameterize_harmonic_bond(ff.hb_handle.params)
    angle_params, ha = bt.parameterize_harmonic_angle(ff.ha_handle.params)
    proper_params, pt = bt.parameterize_proper_torsion(ff.pt_handle.params)
    improper_params, it = bt.parameterize_improper_torsion(ff.it_handle.params)

    bond_idxs = hb.get_idxs()
    angle_idxs = ha.get_idxs()
    proper_idxs = pt.get_idxs()
    improper_idxs = it.get_idxs()

    if core is None:
        core = np.arange(mol.GetNumAtoms())

    core_bond_idxs = []
    core_bond_params = []
    for ij, p in zip(bond_idxs, bond_params):
        if all([a in core for a in ij]):
            core_bond_idxs.append(ij)
            core_bond_params.append(p)

    core_angle_idxs = []
    core_angle_params = []
    for ijk, p in zip(angle_idxs, angle_params):
        if all([a in core for a in ijk]):
            core_angle_idxs.append(ijk)
            core_angle_params.append(p)

    core_proper_idxs = []
    core_proper_params = []
    for ijkl, p in zip(proper_idxs, proper_params):
        if all([a in core for a in ijkl]):
            core_proper_idxs.append(ijkl)
            core_proper_params.append(p)

    core_improper_idxs = []
    core_improper_params = []
    for ijkl, p in zip(improper_idxs, improper_params):
        if all([a in core for a in ijkl]):
            core_improper_idxs.append(ijkl)
            core_improper_params.append(p)

    atom_geometries = label_stereo(
        mol.GetNumAtoms(),
        core_bond_idxs,
        core_angle_idxs,
        core_angle_params,
        core_improper_idxs,
        core_improper_params,
    )

    return atom_geometries
