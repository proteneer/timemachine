import itertools

import numpy as np

from timemachine.potentials.chiral_restraints import pyramidal_volume, torsion_volume


def setup_chiral_atom_restraints(mol, conf, a_idx):
    """
    Setup chiral atom restraints for the molecule at a_idx by inspecting the
    given geometry.
    """
    nbs = mol.GetAtomWithIdx(a_idx).GetNeighbors()
    restr_idxs = []
    for a_i, a_j, a_k in itertools.combinations(nbs, 3):
        i, j, k = a_i.GetIdx(), a_j.GetIdx(), a_k.GetIdx()
        vol = pyramidal_volume(conf[a_idx], conf[i], conf[j], conf[k])
        # vol may be >0 or <0, our chiral restraint always enforces vol < 0.
        if vol < 0:
            restr_idxs.append([a_idx, i, j, k])
        else:
            restr_idxs.append([a_idx, j, i, k])

    return np.array(restr_idxs)


def setup_chiral_bond_restraints(mol, conf, src_idx, dst_idx):
    """
    Setup chiral bond restraints for the molecule at a_idx by inspecting the
    given geometry
    """
    src_nbs = [a.GetIdx() for a in mol.GetAtomWithIdx(src_idx).GetNeighbors()]
    dst_nbs = [a.GetIdx() for a in mol.GetAtomWithIdx(dst_idx).GetNeighbors()]

    assert src_idx in dst_nbs
    assert dst_idx in src_nbs

    src_nbs.remove(dst_idx)
    dst_nbs.remove(src_idx)

    # build chiral restraints
    restr_idxs = []
    signs = []
    for a in src_nbs:
        for b in dst_nbs:
            # set up torsion i,j,k,l
            i, j, k, l = a, src_idx, dst_idx, b
            vol = torsion_volume(conf[i], conf[j], conf[k], conf[l])
            restr_idxs.append([i, j, k, l])
            if vol < 0:
                # (jkaus): the restraints are turned on when the volume is positive
                # so use the opposite sign here
                signs.append(1)
            else:
                signs.append(-1)

    return np.array(restr_idxs), np.array(signs)
