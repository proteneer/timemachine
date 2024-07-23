import warnings
from collections.abc import Iterable
from enum import IntEnum
from functools import partial
from typing import Callable, Collection, Dict, FrozenSet, List, Optional, Tuple, TypeVar, Union, cast

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
from numpy.typing import NDArray
from rdkit import Chem

from timemachine.constants import DEFAULT_CHIRAL_ATOM_RESTRAINT_K, DEFAULT_CHIRAL_BOND_RESTRAINT_K
from timemachine.fe import chiral_utils, interpolate, model_utils, topology, utils
from timemachine.fe.chiral_utils import ChiralRestrIdxSet
from timemachine.fe.dummy import canonicalize_bond, generate_anchored_dummy_group_assignments
from timemachine.fe.lambda_schedule import construct_pre_optimized_relative_lambda_schedule
from timemachine.fe.system import HostGuestSystem, VacuumSystem
from timemachine.fe.topology import get_ligand_ixn_pots_params
from timemachine.potentials import (
    BoundPotential,
    ChiralAtomRestraint,
    ChiralBondRestraint,
    HarmonicAngleStable,
    HarmonicBond,
    Nonbonded,
    NonbondedPairListPrecomputed,
    PeriodicTorsion,
    SummedPotential,
)


class CoreBondChangeWarning(UserWarning):
    pass


class MissingAngleError(RuntimeError):
    pass


class ChargePertubationError(RuntimeError):
    pass


class ChiralConversionError(RuntimeError):
    pass


def recursive_map(items, mapping):
    """recursively replace items in a list of tuple
    mapping = np.arange(100)[::-1]
    items = [[0,2,3], [5,1,[2,5,6]], 3]
    result = recursive_map(items, mapping)
    # ((99, 97, 96), (94, 98, (97, 94, 93)), 96)
    """
    if isinstance(items, Iterable):
        res = []
        for item in items:
            res.append(recursive_map(item, mapping))
        return tuple(res)
    else:
        return mapping[items]


def setup_dummy_interactions_from_ff(
    ff, mol, dummy_group, root_anchor_atom, nbr_anchor_atom, core_atoms, chiral_atom_k, chiral_bond_k
):
    """
    Setup interactions involving atoms in a given dummy group.
    """
    top = topology.BaseTopology(mol, ff)

    bond_params, hb = top.parameterize_harmonic_bond(ff.hb_handle.params)
    angle_params, ha = top.parameterize_harmonic_angle(ff.ha_handle.params)
    improper_params, it = top.parameterize_improper_torsion(ff.it_handle.params)
    chiral_atom_potential, _ = top.setup_chiral_restraints(chiral_atom_k, chiral_bond_k)
    chiral_atom_idxs = chiral_atom_potential.potential.idxs
    chiral_atom_params = chiral_atom_potential.params

    # note: core atoms are not simply set(mol_a_atoms).difference(dummy_group)
    # since multiple dummy groups may be present

    return setup_dummy_interactions(
        hb.idxs,
        bond_params,
        ha.idxs,
        angle_params,
        it.idxs,
        improper_params,
        chiral_atom_idxs,
        chiral_atom_params,
        dummy_group,
        root_anchor_atom,
        nbr_anchor_atom,
        core_atoms,
    )


def setup_dummy_interactions(
    bond_idxs,
    bond_params,
    angle_idxs,
    angle_params,
    improper_idxs,
    improper_params,
    chiral_atom_idxs,
    chiral_atom_params,
    dummy_group,
    root_anchor_atom,
    nbr_anchor_atom,
    core_atoms,
):
    """
    Setup interactions involving atoms in a given dummy group. The following rules are applied:

    1) We only allow for interactions within a dummy group, never between different dummy groups.
    2) We form the augmented_dummy_group = dummy_group + [root anchor], and:
        i) we only allow bond, angle, improper torsions terms to be turned on within a dummy group.
        ii) we disable all nonbonded and proper torsions involving atoms in dummy groups
    3) We can form a secondary augmented group using nbr_anchor_atom, but we only allow interactions
        involving angles [i,j,k] where i in dummy_group, j == root_anchor_atom, and k == nbr_anchor_atom
    4) Chiral restraints naturally factorize out via symmetry, and can be safely left on, as long as
       at least one atom is a dummy atom.

    Our motivation for this is 1) to ensure that the dummy interactions are factorizable, and that 2) the
    dummy system is in an "enhanced" state so we can sample over torsional barriers etc. efficiently.

    Parameters
    ----------
    bond_idxs: list of 2-tuples
        Bond idxs

    bond_params: list of 2-tuples
        Force constants and bond lengths

    angle_idxs: list of 3-tuples
        Angle spanned by (i,j,k)

    angle_params: list of 2-tuples
        Force constants and equilibrium angles

    improper_idxs: list of 4-tuples
        Trefoil idxs for improper torsions

    improper_params: list of 3-tuples
        Force constant, phase, periods

    chiral_atom_idxs : list of 4-tuples
    chiral_atom_params: list of floats

    dummy_group: set or list of int
        Atoms to be decoupled

    root_anchor_atom: int
        A core atom we want to anchor our dummy_group to

    nbr_anchor_atom: int
        Another core atom connected to root_anchor_atom to build an angle restraint off of.

    core_atoms: list of int
        Core atoms (excluding all dummy atoms, not just the ones in this group)

    Returns
    -------
    (bonded_idxs, bonded_params)
        Returns bonds, angles, and improper idxs and parameters.
    """
    assert root_anchor_atom in core_atoms

    dummy_bond_idxs = []
    dummy_bond_params = []
    dummy_angle_idxs = []
    dummy_angle_params = []
    dummy_improper_idxs = []
    dummy_improper_params = []
    dummy_chiral_atom_idxs = []
    dummy_chiral_atom_params = []

    # dummy_group may be a set in certain cases, so sanity check.

    assert len(dummy_group) == len(list(dummy_group))
    dummy_group = list(dummy_group)

    # dummy group and anchor
    dga = dummy_group + [root_anchor_atom]

    # copy interactions that involve only root_anchor_atom
    for idxs, params in zip(bond_idxs, bond_params):
        if all([a in dga for a in idxs]):
            dummy_bond_idxs.append(tuple([int(x) for x in idxs]))  # tuples are hashable etc.
            dummy_bond_params.append(params)
    for idxs, params in zip(angle_idxs, angle_params):
        if all([a in dga for a in idxs]):
            dummy_angle_idxs.append(tuple([int(x) for x in idxs]))
            dummy_angle_params.append(params)
    for idxs, params in zip(improper_idxs, improper_params):
        if all([a in dga for a in idxs]):
            dummy_improper_idxs.append(tuple([int(x) for x in idxs]))
            dummy_improper_params.append(params)

    # certain configuration of chiral states are symmetrizable
    # . means a bond involving at least 1 dummy atom
    # | means a bond involving only core atoms
    #
    #        all allowed geometries              |  disallowed geometry
    #     center is core     center is not core  |
    #    d      c      c      d      d      d    |    c      c
    #    .      |      |      .      .      .    |    .      |
    #    c      c      c      d      d      d    |    d      c
    #   . .    . .    / .    . .    . .    . .   |   . .    / \
    #  d   d  d   d  c   d  c   d  c   c  d   d  |  c   c  c   c (only core)
    dgc = dummy_group + list(core_atoms)
    for idxs, params in zip(chiral_atom_idxs, chiral_atom_params):
        center, i, j, k = idxs
        if all([a in dgc for a in idxs]):
            # non center dummy atom count
            ncda_count = sum([a in dummy_group for a in (i, j, k)])
            if ncda_count == 1 or ncda_count == 2 or ncda_count == 3:
                assert not all(a in core_atoms for a in idxs)
                dummy_chiral_atom_idxs.append(tuple(int(x) for x in idxs))
                dummy_chiral_atom_params.append(params)

    # (ytz): copy interactions that involve nbr_anchor_atom, if not None
    # this may be set to None
    if nbr_anchor_atom is not None:
        assert nbr_anchor_atom in core_atoms
        found = False
        for idxs, params in zip(angle_idxs, angle_params):
            i, j, k = idxs
            if (i in dummy_group and j == root_anchor_atom and k == nbr_anchor_atom) or (
                k in dummy_group and j == root_anchor_atom and i == nbr_anchor_atom
            ):
                dummy_angle_idxs.append(tuple([int(x) for x in idxs]))
                dummy_angle_params.append(params)
                found = True

        if not found:
            # User provided a bad nbr_anchor_atom
            raise MissingAngleError(
                f"Missing angle interaction in mol_b, dg={dummy_group}, root={root_anchor_atom}, nbr={nbr_anchor_atom}"
            )

    bonded_idxs = (dummy_bond_idxs, dummy_angle_idxs, dummy_improper_idxs, dummy_chiral_atom_idxs)
    bonded_params = (dummy_bond_params, dummy_angle_params, dummy_improper_params, dummy_chiral_atom_params)
    return bonded_idxs, bonded_params


def canonicalize_improper_idxs(idxs) -> Tuple[int, int, int, int]:
    """
    Canonicalize an improper_idx while being symmetry aware.

    Given idxs (i,j,k,l), where i is the center, and (j,k,l) are neighbors:

    0) Canonicalize the (j,k,l) into (jj,kk,ll) by sorting
    1) Generate clockwise rotations of (jj,kk,ll)
    2) Generate counter clockwise rotations of (jj,kk,ll)
    3) We now can sort 1) and 2) and assign a mapping

    If the (j,k,l) is in the cw rotation ordered set, we're done. Otherwise it must
    be in the ccw ordered set. We look up the corresponding idx in the cw set.

    This does not do idxs[0] < idxs[-1] canonicalization.
    """
    i, j, k, l = idxs

    # i is the center
    # generate lexical order
    key = (j, k, l)

    jj, kk, ll = sorted(key)

    # generate clockwise permutations
    # note: cw/ccw has nothing to do with the direction of rotation
    # cw/ccw is related by a pair swap.
    cw_jkl = (jj, kk, ll)  # starting idxs
    cw_klj = (kk, ll, jj)  # rotate left
    cw_ljk = (ll, jj, kk)  # rotate left
    cw_items = sorted([cw_jkl, cw_klj, cw_ljk])

    if key in cw_items:
        return (i, j, k, l)

    # generate counter clockwise permutations
    ccw_kjl = (kk, jj, ll)  # swap 1st and 2nd element
    ccw_jlk = (jj, ll, kk)  # rotate left
    ccw_lkj = (ll, kk, jj)  # rotate left
    ccw_items = sorted([ccw_kjl, ccw_jlk, ccw_lkj])

    assert key in ccw_items

    for idx, cw_item in enumerate(ccw_items):
        if cw_item == key:
            break

    return (i, *cw_items[idx])


def get_num_connected_components(num_atoms: int, bonds: Collection[Tuple[int, int]]) -> int:
    g = nx.Graph()
    g.add_nodes_from(range(num_atoms))
    g.add_edges_from(bonds)
    return len(list(nx.connected_components(g)))


def canonicalize_chiral_atom_idxs(idxs):
    i, j, k, l = idxs
    rotations = [(j, k, l), (l, j, k), (k, l, j)]
    jj, kk, ll = min(rotations)
    return i, jj, kk, ll


def setup_end_state_harmonic_bond_and_chiral_potentials(
    ff, mol_a, mol_b, core, a_to_c, b_to_c
) -> Tuple[BoundPotential[HarmonicBond], BoundPotential[ChiralAtomRestraint], BoundPotential[ChiralBondRestraint]]:
    """
    Setup end-state potentials to verify chiral correctness for mol_a with dummy atoms of mol_b attached. The mapped indices will correspond
    to the alchemical molecule with dummy atoms. Note that the bond, chiral atom and chiral bond idxs are canonicalized.

    This code is identical to setup_end_state, but only handles chiral potentials to be used to quickly verify
    core mappings in verify_chiral_consistency_of_core

    Parameters
    ----------
    ff: forcefield.Forcefield
        Forcefield used to parameterize the molecule

    mol_a: Chem.Mol
        Fully interacting molecule

    mol_b: Chem.Mol
        Molecule providing the dummy atoms.

    core: list of 2-tuples
        Each pair is an atom mapping from mol_a into mol_b

    a_to_c: dict or array, supports []
        mapping from a into a common core idx

    b_to_c: dict or array, supports []
        mapping from b into a common core idx

    Returns
    -------
    Tuple of bound HarmonicBond, ChiralAtomRestraint and ChiralBondRestraint
        The potentials involved in checking the correctness and validity of chirality at endstates.

    """
    all_dummy_bond_idxs, all_dummy_bond_params = [], []
    all_dummy_chiral_atom_idxs, all_dummy_chiral_atom_params = [], []

    dummy_groups = find_dummy_groups_and_anchors(mol_a, mol_b, core[:, 0], core[:, 1])
    # gotta add 'em all!

    for anchor, (nbr, dg) in dummy_groups.items():
        all_idxs, all_params = setup_dummy_interactions_from_ff(
            ff, mol_b, dg, anchor, nbr, core[:, 1], DEFAULT_CHIRAL_ATOM_RESTRAINT_K, DEFAULT_CHIRAL_BOND_RESTRAINT_K
        )
        # append idxs
        all_dummy_bond_idxs.extend(all_idxs[0])
        all_dummy_chiral_atom_idxs.extend(all_idxs[3])
        # append params
        all_dummy_bond_params.extend(all_params[0])
        all_dummy_chiral_atom_params.extend(all_params[3])

    # generate parameters for mol_a
    mol_a_top = topology.BaseTopology(mol_a, ff)
    mol_a_bond_params, mol_a_hb = mol_a_top.parameterize_harmonic_bond(ff.hb_handle.params)
    mol_a_chiral_atom, mol_a_chiral_bond = mol_a_top.setup_chiral_restraints(
        DEFAULT_CHIRAL_ATOM_RESTRAINT_K, DEFAULT_CHIRAL_BOND_RESTRAINT_K
    )

    mol_a_bond_params = mol_a_bond_params.tolist()

    mol_a_bond_idxs = recursive_map(mol_a_hb.idxs, a_to_c)
    mol_a_chiral_atom_idxs = recursive_map(mol_a_chiral_atom.potential.idxs, a_to_c)
    mol_a_chiral_bond_idxs = recursive_map(mol_a_chiral_bond.potential.idxs, a_to_c)

    all_dummy_bond_idxs = recursive_map(all_dummy_bond_idxs, b_to_c)
    all_dummy_chiral_atom_idxs = recursive_map(all_dummy_chiral_atom_idxs, b_to_c)

    # parameterize the combined molecule
    mol_c_bond_idxs = mol_a_bond_idxs + all_dummy_bond_idxs
    mol_c_bond_params = mol_a_bond_params + all_dummy_bond_params

    # process chiral volumes, turning off ones at the end-state that have a missing bond.
    canon_mol_a_bond_idxs_set = set([canonicalize_bond(x) for x in mol_a_bond_idxs])
    # assert presence of bonds
    for c, i, j, k in mol_a_chiral_atom_idxs:
        ci = canonicalize_bond((c, i))
        cj = canonicalize_bond((c, j))
        ck = canonicalize_bond((c, k))
        assert ci in canon_mol_a_bond_idxs_set
        assert cj in canon_mol_a_bond_idxs_set
        assert ck in canon_mol_a_bond_idxs_set

    canon_mol_c_bond_idxs_set = set([canonicalize_bond(x) for x in mol_c_bond_idxs])

    # Chiral atom restraint c,i,j,k requires that all bonds ci, cj, ck be present at the
    # end-state in order to be numerically stable under small perturbations due to normalization
    # along the bond lengths. However, the angle terms defining icj, ick, and jck can be
    # either 0 or 180, since the normalized chiral volume is still smooth wrt perturbations
    all_proper_dummy_chiral_atom_idxs = []
    all_proper_dummy_chiral_atom_params = []
    for (c, i, j, k), p in zip(all_dummy_chiral_atom_idxs, all_dummy_chiral_atom_params):
        ci = canonicalize_bond((c, i))
        cj = canonicalize_bond((c, j))
        ck = canonicalize_bond((c, k))
        if ci in canon_mol_c_bond_idxs_set and cj in canon_mol_c_bond_idxs_set and ck in canon_mol_c_bond_idxs_set:
            all_proper_dummy_chiral_atom_idxs.append((c, i, j, k))
            all_proper_dummy_chiral_atom_params.append(p)
        else:
            warnings.warn(f"Chiral Volume {c,i,j,k} has a disabled bond, turning off.")

    mol_c_chiral_atom_idxs = list(mol_a_chiral_atom_idxs) + list(all_proper_dummy_chiral_atom_idxs)
    mol_c_chiral_atom_params = np.concatenate([mol_a_chiral_atom.params, all_proper_dummy_chiral_atom_params])

    # canonicalize bonds
    mol_c_bond_idxs_canon = np.array([canonicalize_bond(idxs) for idxs in mol_c_bond_idxs])
    bond_potential = HarmonicBond(mol_c_bond_idxs_canon).bind(np.array(mol_c_bond_params))

    # chiral atoms need special code for canonicalization, since triple product is invariant
    # under rotational symmetry (but not something like swap symmetry)
    canon_chiral_atom_idxs = []
    for idxs in mol_c_chiral_atom_idxs:
        canon_chiral_atom_idxs.append(canonicalize_chiral_atom_idxs(idxs))

    chiral_atom_idxs = np.array(canon_chiral_atom_idxs, dtype=np.int32).reshape((-1, 4))
    mol_c_chiral_bond_idxs_canon = [canonicalize_bond(idxs) for idxs in mol_a_chiral_bond_idxs]
    chiral_bond_idxs = np.array(mol_c_chiral_bond_idxs_canon, dtype=np.int32).reshape((-1, 4))
    chiral_bond_signs = np.array(mol_a_chiral_bond.potential.signs)

    chiral_atom_potential = ChiralAtomRestraint(chiral_atom_idxs).bind(mol_c_chiral_atom_params)
    chiral_bond_potential = ChiralBondRestraint(chiral_bond_idxs, chiral_bond_signs).bind(mol_a_chiral_bond.params)

    num_atoms = mol_a.GetNumAtoms() + mol_b.GetNumAtoms() - len(core)
    assert (
        get_num_connected_components(num_atoms, bond_potential.potential.idxs) == 1
    ), "hybrid molecule has multiple connected components"
    return bond_potential, chiral_atom_potential, chiral_bond_potential


def setup_end_state(ff, mol_a, mol_b, core, a_to_c, b_to_c):
    """
    Setup end-state for mol_a with dummy atoms of mol_b attached. The mapped indices will correspond
    to the alchemical molecule with dummy atoms. Note that the bond, angle, torsion, nonbonded pairs,
    chiral atom and chiral bond idxs are canonicalized.

    Parameters
    ----------
    ff: forcefield.Forcefield
        Forcefield used to parameterize the molecule

    mol_a: Chem.Mol
        Fully interacting molecule

    mol_b: Chem.Mol
        Molecule providing the dummy atoms.

    core: list of 2-tuples
        Each pair is an atom mapping from mol_a into mol_b

    a_to_c: dict or array, supports []
        mapping from a into a common core idx

    b_to_c: dict or array, supports []
        mapping from b into a common core idx

    Returns
    -------
    VacuumSystem
        A parameterized system in the vacuum.

    """

    all_dummy_angle_idxs, all_dummy_angle_params = [], []
    all_dummy_improper_idxs, all_dummy_improper_params = [], []

    dummy_groups = find_dummy_groups_and_anchors(mol_a, mol_b, core[:, 0], core[:, 1])
    # gotta add 'em all!

    for anchor, (nbr, dg) in dummy_groups.items():
        all_idxs, all_params = setup_dummy_interactions_from_ff(
            ff, mol_b, dg, anchor, nbr, core[:, 1], DEFAULT_CHIRAL_ATOM_RESTRAINT_K, DEFAULT_CHIRAL_BOND_RESTRAINT_K
        )
        # append idxs
        all_dummy_angle_idxs.extend(all_idxs[1])
        all_dummy_improper_idxs.extend(all_idxs[2])
        # append params
        all_dummy_angle_params.extend(all_params[1])
        all_dummy_improper_params.extend(all_params[2])

    # generate parameters for mol_a
    mol_a_top = topology.BaseTopology(mol_a, ff)
    mol_a_angle_params, mol_a_ha = mol_a_top.parameterize_harmonic_angle(ff.ha_handle.params)
    mol_a_proper_params, mol_a_pt = mol_a_top.parameterize_proper_torsion(ff.pt_handle.params)
    mol_a_improper_params, mol_a_it = mol_a_top.parameterize_improper_torsion(ff.it_handle.params)
    mol_a_nbpl_params, mol_a_nbpl = mol_a_top.parameterize_nonbonded_pairlist(
        ff.q_handle.params,
        ff.q_handle_intra.params,
        ff.lj_handle.params,
        ff.lj_handle_intra.params,
        intramol_params=True,
    )

    mol_a_angle_params = mol_a_angle_params.tolist()
    mol_a_proper_params = mol_a_proper_params.tolist()
    mol_a_improper_params = mol_a_improper_params.tolist()
    mol_a_nbpl_params = mol_a_nbpl_params.tolist()

    mol_a_angle_idxs = recursive_map(mol_a_ha.idxs, a_to_c)
    mol_a_proper_idxs = recursive_map(mol_a_pt.idxs, a_to_c)
    mol_a_improper_idxs = recursive_map(mol_a_it.idxs, a_to_c)
    mol_a_nbpl_idxs = recursive_map(mol_a_nbpl.idxs, a_to_c)

    all_dummy_angle_idxs = recursive_map(all_dummy_angle_idxs, b_to_c)
    all_dummy_improper_idxs = recursive_map(all_dummy_improper_idxs, b_to_c)

    mol_c_angle_idxs = mol_a_angle_idxs + all_dummy_angle_idxs
    mol_c_angle_params = mol_a_angle_params + all_dummy_angle_params

    mol_c_proper_idxs = mol_a_proper_idxs
    mol_c_proper_params = mol_a_proper_params

    mol_c_improper_idxs = mol_a_improper_idxs + all_dummy_improper_idxs
    mol_c_improper_params = mol_a_improper_params + all_dummy_improper_params

    # canonicalize improper with cw/ccw check
    mol_c_improper_idxs = tuple([canonicalize_improper_idxs(idxs) for idxs in mol_c_improper_idxs])

    # check that the improper idxs are canonical
    def assert_improper_idxs_are_canonical(all_idxs):
        for _, j, k, l in all_idxs:
            jj, kk, ll = sorted((j, k, l))
            assert (jj, kk, ll) == (j, k, l) or (kk, ll, jj) == (j, k, l) or (ll, jj, kk) == (j, k, l)

    assert_improper_idxs_are_canonical(mol_c_improper_idxs)

    # combine proper + improper
    mol_c_torsion_idxs = mol_c_proper_idxs + mol_c_improper_idxs
    mol_c_torsion_params = mol_c_proper_params + mol_c_improper_params

    # canonicalize angles
    mol_c_angle_idxs_canon = np.array([canonicalize_bond(idxs) for idxs in mol_c_angle_idxs])
    mol_c_stable_angle_params = np.hstack([mol_c_angle_params, np.zeros((len(mol_c_angle_params), 1))])
    angle_potential = HarmonicAngleStable(mol_c_angle_idxs_canon).bind(np.array(mol_c_stable_angle_params))

    # canonicalize torsions with idxs[0] < idxs[-1] check
    mol_c_torsion_idxs_canon = np.array([canonicalize_bond(idxs) for idxs in mol_c_torsion_idxs])
    torsion_potential = PeriodicTorsion(mol_c_torsion_idxs_canon).bind(np.array(mol_c_torsion_params))

    # dummy atoms do not have any nonbonded interactions, so we simply turn them off
    mol_c_nbpl_idxs_canon = np.array([canonicalize_bond(idxs) for idxs in mol_a_nbpl_idxs])
    mol_a_nbpl.idxs = mol_c_nbpl_idxs_canon
    nonbonded_potential = mol_a_nbpl.bind(np.array(mol_a_nbpl_params))

    bond_potential, chiral_atom_potential, chiral_bond_potential = setup_end_state_harmonic_bond_and_chiral_potentials(
        ff, mol_a, mol_b, core, a_to_c, b_to_c
    )

    return VacuumSystem(
        bond_potential,
        angle_potential,
        torsion_potential,
        nonbonded_potential,
        chiral_atom_potential,
        chiral_bond_potential,
    )


def find_dummy_groups_and_anchors(
    mol_a, mol_b, core_atoms_a: Collection[int], core_atoms_b: Collection[int]
) -> Dict[int, Tuple[Optional[int], FrozenSet[int]]]:
    """Returns an arbitrary partitioning of dummy atoms and anchor assignment for the A -> B transformation. See the
    documentation for :py:func:`timemachine.fe.dummy.generate_dummy_group_assignments` and notes below for more
    information.

    Notes
    -----
    Consider the following situation:

    D0.D1
    .  .  where (.) is the dummy bond
    C0-C1

    One of (C0.D0), (D0.D1), (C1.D1) dummy bonds needs to be broken in order to maintain factorizability. This is a
    little arbitrary, but some choices are probably more efficient than others:

    hard     easy
    D0.D1    D0 D1
       .     .  .
    C0-C1    C0-C1

    The LHS is more difficult because it has significantly more phase space that can be sampled than the fused case, but
    it's not super obvious how to best detect this. So instead, we will pick an arbitrary anchor atom. One possible
    solution later on is to minimize the number of rotatable bonds?
    """

    assignments = generate_anchored_dummy_group_assignments(mol_a, mol_b, core_atoms_a, core_atoms_b)

    # TODO: consider refining to use a heuristic rather than arbitrary selection
    # (e.g. maximize core-dummy bonds, maximize angle terms, minimize rotatable bonds, etc.)
    arbitrary_assignment = next(assignments)

    for _, (angle_anchor, _) in arbitrary_assignment.items():
        if angle_anchor is None:
            warnings.warn("Unable to find stable angle term in mol_a", CoreBondChangeWarning)

    return arbitrary_assignment


def handle_ring_opening_closing(
    f: Callable[[float, float, float], float],
    src_k: float,
    dst_k: float,
    lamb: float,
    lambda_min: float,
    lambda_max: float,
) -> float:
    """
    In the typical case (src_k != 0 and dst_k != 0), use the specified interpolation function, f.

    In the case where src_k = 0 or dst_k = 0 (e.g. ring closing and ring opening, respectively), restrict interpolation
    to the interval [lambda_min, lambda_max], and pin to the end state values outside of this range.

    Parameters
    ----------
    f : callable, (src_k: float, dst_k: float, lam: float) -> float
        interpolation function; should satisfy f(0) = src_k, f(1) = dst_k

    src_k, dst_k : float, k >= 0
        force constants at lambda=0 and lambda=1, respectively

    lambda_min, lambda_max : float, in 0 < lambda_min < lambda_max < 1
        interpolate in range [lambda_min, lambda_max] (pin to end states otherwise). Note that if dst_k=0, the
        convention is flipped so that 1 - lambda_min corresponds to f(0) and 1 - lambda_max corresponds to f(1).

    Returns
    -------
    float
        interpolated force constant
    """

    def ring_closing(dst_k, lamb):
        return interpolate.pad(f, 0.0, dst_k, lamb, lambda_min, lambda_max)

    def ring_opening(src_k, lamb):
        return ring_closing(src_k, 1.0 - lamb)

    return jnp.where(
        src_k == 0.0,
        ring_closing(dst_k, lamb),
        jnp.where(
            dst_k == 0.0,
            ring_opening(src_k, lamb),
            f(src_k, dst_k, lamb),
        ),
    )


def interpolate_harmonic_force_constant(src_k, dst_k, lamb, k_min, lambda_min, lambda_max):
    """
    Interpolate between force constants using a log-linear functional form.

    In the special case when src_k=0 or dst_k=0 (e.g. ring opening or closing transformations):

    1. Intermediates are interpolated from k_min instead of zero (since 0 is not in the range of the interpolation
       function)
    2. Interpolation is restricted to the interval [lambda_min, lambda_max] and pinned to the end state values outside
       of this range

    Parameters
    ----------
    src_k, dst_k : float, k >= 0
        force constants at lambda=0 and lambda=1, respectively

    k_min : float, k_min > 0
        minimum force constant for interpolation

    lambda_min, lambda_max : float, in 0 < lambda_min < lambda_max < 1
        interpolate in range [lambda_min, lambda_max] (pin to end states otherwise). Note that if dst_k=0, the
        convention is flipped so that 1 - lambda_min corresponds to f(0) and 1 - lambda_max corresponds to f(1).

    Returns
    -------
    float
        interpolated force constant
    """

    return jnp.where(
        lamb == 0.0,
        src_k,
        jnp.where(
            lamb == 1.0,
            dst_k,
            handle_ring_opening_closing(
                partial(interpolate.log_linear_interpolation, min_value=k_min),
                src_k,
                dst_k,
                lamb,
                lambda_min,
                lambda_max,
            ),
        ),
    )


def interpolate_harmonic_bond_params(src_params, dst_params, lamb, k_min, lambda_min, lambda_max):
    """
    Interpolate harmonic bond parameters using

    1. Log-linear interpolation for force constants*
    2. Linear interpolation for equilibrium bond lengths

    * see note on special case when src_k=0 or dst_k=0 in the docstring of `interpolate_harmonic_force_constant`.

    Parameters
    ----------
    src_params : array-like, float, (2,)
        force constant and equilibrium length at lambda=0

    dst_params : array-like, float, (2,)
        force constant and equilibrium length at lambda=1

    lamb : float
        alchemical parameter

    k_min, lambda_min, lambda_max : float
        see docstring of `interpolate_harmonic_force_constant` for documentation of these parameters

    Returns
    -------
    array, float, (2,)
        interpolated (force constant, equilibrium length)
    """

    src_k, src_x = src_params
    dst_k, dst_x = dst_params

    k = interpolate_harmonic_force_constant(src_k, dst_k, lamb, k_min, lambda_min, lambda_max)
    x = interpolate.linear_interpolation(src_x, dst_x, lamb)

    return jnp.array([k, x])


def cyclic_difference(a, b, period):
    """
    Returns the minimum difference between two points, with periodic boundaries.
    I.e. the solution of ::

        (a + x) % period = b % period

    with minimum abs(x).
    """

    d = jnp.fmod(b - a, period)

    def f(d):
        return jnp.where(d <= period / 2, d, d - period)

    return jnp.sign(d) * f(jnp.abs(d))


def interpolate_harmonic_angle_params(src_params, dst_params, lamb, k_min, lambda_min, lambda_max):
    """
    Interpolate harmonic angle parameters using

    1. Log-linear interpolation for force constants*
    2. Shortest-path linear interpolation for equilibrium angles

    * see note on special case when src_k=0 or dst_k=0 in the docstring of `interpolate_harmonic_force_constant`.

    Parameters
    ----------
    src_params : array-like, float, (2,)
        force constant and equilibrium angle at lambda=0

    dst_params : array-like, float, (2,)
        force constant and equilibrium angle at lambda=1

    lamb : float
        alchemical parameter

    k_min, lambda_min, lambda_max : float
        see docstring of `interpolate_harmonic_force_constant` for documentation of these parameters

    Returns
    -------
    array, float, (2,)
        interpolated (force constant, equilibrium phase)
    """

    src_k, src_phase, _ = src_params
    dst_k, dst_phase, _ = dst_params

    k = interpolate_harmonic_force_constant(src_k, dst_k, lamb, k_min, lambda_min, lambda_max)

    phase = interpolate.linear_interpolation(
        src_phase,
        src_phase + cyclic_difference(src_phase, dst_phase, period=2 * np.pi),
        lamb,
    )

    # Use a stable functional form with small, finite `eps` for intermediate states only. The value of `eps` for
    # intermedates was chosen to be sufficiently large that no numerical instabilities were observed in testing (even
    # with bond force constants approximately zero), and sufficiently small to have negligible impact on the overlap of
    # the end states with neighboring intermediates.
    eps = jnp.where((lamb == 0.0) | (lamb == 1.0), 0.0, 1e-3)

    return jnp.array([k, phase, eps])


def interpolate_periodic_torsion_params(src_params, dst_params, lamb, lambda_min, lambda_max):
    """
    Interpolate periodic torsion parameters using

    1. Linear interpolation for force constants*
    2. Linear interpolation for angles, using the shortest path
    3. No interpolation for periodicity (pinned to source value)

    * see note on special case when src_k=0 or dst_k=0 in the docstring of `interpolate_harmonic_force_constant`.

    Parameters
    ----------
    src_params : array-like, float, (2,)
        force constant, equilibrium dihedral angle, and periodicity at lambda=0

    dst_params : array-like, float, (2,)
        force constant and equilibrium dihedral angle, and periodicity at lambda=1

    lamb : float
        alchemical parameter

    lambda_min, lambda_max : float
        see docstring of `interpolate_harmonic_force_constant` for documentation of these parameters

    Returns
    -------
    array, float, (3,)
        interpolated (force constant, equilibrium phase, periodicity)
    """

    src_k, src_phase, src_period = src_params
    dst_k, dst_phase, _ = dst_params

    k = handle_ring_opening_closing(interpolate.linear_interpolation, src_k, dst_k, lamb, lambda_min, lambda_max)

    phase = interpolate.linear_interpolation(
        src_phase,
        src_phase + cyclic_difference(src_phase, dst_phase, period=2 * np.pi),
        lamb,
    )

    return jnp.array([k, phase, src_period])


def interpolate_w_coord(w0: float | jax.Array, w1: float | jax.Array, lamb: float):
    """Interpolate 4D coordinate using schedule optimized for RBFE calculations.

    Parameters
    ----------
    w0, w1 : float
        w coordinates at lambda = 0 and 1 respectively

    lamb : float
        alchemical parameter
    """
    lambdas = construct_pre_optimized_relative_lambda_schedule(None)
    x = jnp.linspace(0.0, 1.0, len(lambdas))
    return jnp.where(
        w0 < w1,
        interpolate.linear_interpolation(w0, w1, jnp.interp(lamb, x, lambdas)),
        interpolate.linear_interpolation(w1, w0, jnp.interp(1.0 - lamb, x, lambdas)),
    )


class AtomMapFlags(IntEnum):
    CORE = 0
    MOL_A = 1
    MOL_B = 2


class AtomMapMixin:
    """
    A Mixin class containing the atom_mapping information. This Mixin sets up the following
    members:

    self.mol_a
    self.mol_b
    self.core
    self.a_to_c
    self.b_to_c
    self.c_to_a
    self.c_to_b
    self.c_flags
    """

    def __init__(self, mol_a, mol_b, core):
        assert core.shape[1] == 2
        assert mol_a is not None
        assert mol_b is not None

        self.mol_a = mol_a
        self.mol_b = mol_b
        self.core = core
        assert mol_a is not None
        assert mol_b is not None
        assert core.shape[1] == 2

        # map into idxs in the combined molecule

        self.a_to_c = np.arange(mol_a.GetNumAtoms(), dtype=np.int32)  # identity
        self.b_to_c = np.zeros(mol_b.GetNumAtoms(), dtype=np.int32) - 1

        # mark membership:
        # AtomMapFlags.CORE: Core
        # AtomMapFlags.MOL_A: R_A (default)
        # AtomMapFlags.MOL_B: R_B
        self.c_flags = np.ones(self.get_num_atoms(), dtype=np.int32) * AtomMapFlags.MOL_A
        # test for uniqueness in core idxs for each mol
        assert len(set(tuple(core[:, 0]))) == len(core[:, 0])
        assert len(set(tuple(core[:, 1]))) == len(core[:, 1])

        for a, b in core:
            self.c_flags[a] = AtomMapFlags.CORE
            self.b_to_c[b] = a

        iota = self.mol_a.GetNumAtoms()
        for b_idx, c_idx in enumerate(self.b_to_c):
            if c_idx == -1:
                self.b_to_c[b_idx] = iota
                self.c_flags[iota] = AtomMapFlags.MOL_B
                iota += 1

        # setup reverse mappings
        self.c_to_a = {v: k for k, v in enumerate(self.a_to_c)}
        self.c_to_b = {v: k for k, v in enumerate(self.b_to_c)}

    def get_num_atoms(self):
        """
        Get the total number of atoms in the alchemical hybrid.

        Returns
        -------
        int
            Total number of atoms.
        """
        return self.mol_a.GetNumAtoms() + self.mol_b.GetNumAtoms() - len(self.core)

    def get_num_dummy_atoms(self):
        """
        Get the total number of dummy atoms in the alchemical hybrid.

        Returns
        -------
        int
            Total number of atoms.
        """
        return self.mol_a.GetNumAtoms() + self.mol_b.GetNumAtoms() - len(self.core) - len(self.core)


_Bonded = TypeVar("_Bonded", bound=Union[ChiralAtomRestraint, HarmonicAngleStable, HarmonicBond, PeriodicTorsion])


def get_neighbors(atom, bond_idxs) -> List[int]:
    nbs = []
    for i, j in bond_idxs:
        if i == atom:
            nbs.append(j)
        elif j == atom:
            nbs.append(i)
    return nbs


def check_chiral_validity(src_chiral_centers_in_mol_c, dst_chiral_restr_idx_set, src_bond_idxs):
    """Raise error unless, for every chiral center, at least 1 chiral volume is defined in both end-states."""

    for c in src_chiral_centers_in_mol_c:
        nbs = get_neighbors(c, src_bond_idxs)
        if len(nbs) == 4:
            i, j, k, l = nbs
            # (ytz): the ordering of i,j,k,l is random if we're reading directly from the mol graph,
            # which can be inconsistent with the ordering used in the chiral volume definition.
            nb_subsets = [(i, j, k), (i, j, l), (i, k, l), (j, k, l)]  # 4-choose-3 subsets
            flags = [dst_chiral_restr_idx_set.defines((c, ii, jj, kk)) for (ii, jj, kk) in nb_subsets]

            if sum(flags) == 0:
                raise ChiralConversionError(f"len(nbs) == 4 {c, i, j, k, l}")

        if len(nbs) == 3:
            i, j, k = nbs
            flag_0 = dst_chiral_restr_idx_set.defines((c, i, j, k))
            if not flag_0:
                raise ChiralConversionError(f"len(nbs) == 3 {c, i, j, k}")


def assert_chiral_consistency_and_validity(
    atom_map: AtomMapMixin,
    src_chiral_idxs,
    dst_chiral_idxs,
    src_bond_idxs: NDArray,
    dst_bond_idxs: NDArray,
):
    """
    Assert that the given the two end states chiral and bond idxs it would be both consistent and valid.

    consistency: if there are no inversions at the end-states between chiral atoms and bonds are present
    validity: if we can directly turn on the chiral volumes (after bonds) without staggering angles
    """

    for c, i, j, k in src_chiral_idxs:
        assert canonicalize_bond((c, i)) in src_bond_idxs
        assert canonicalize_bond((c, j)) in src_bond_idxs
        assert canonicalize_bond((c, k)) in src_bond_idxs

    for c, i, j, k in dst_chiral_idxs:
        assert canonicalize_bond((c, i)) in dst_bond_idxs
        assert canonicalize_bond((c, j)) in dst_bond_idxs
        assert canonicalize_bond((c, k)) in dst_bond_idxs

    src_chiral_restr_idx_set = ChiralRestrIdxSet(src_chiral_idxs)
    dst_chiral_restr_idx_set = ChiralRestrIdxSet(dst_chiral_idxs)

    # ensure that we don't have any chiral inversions between src and dst end states
    assert len(src_chiral_restr_idx_set.allowed_set.intersection(dst_chiral_restr_idx_set.disallowed_set)) == 0
    assert len(dst_chiral_restr_idx_set.allowed_set.intersection(src_chiral_restr_idx_set.disallowed_set)) == 0

    chiral_centers_in_mol_a = chiral_utils.find_chiral_atoms(atom_map.mol_a)
    chiral_centers_in_mol_b = chiral_utils.find_chiral_atoms(atom_map.mol_b)

    src_chiral_centers_in_mol_c = [atom_map.a_to_c[x] for x in chiral_centers_in_mol_a]
    dst_chiral_centers_in_mol_c = [atom_map.b_to_c[x] for x in chiral_centers_in_mol_b]

    check_chiral_validity(src_chiral_centers_in_mol_c, dst_chiral_restr_idx_set, src_bond_idxs)
    check_chiral_validity(dst_chiral_centers_in_mol_c, src_chiral_restr_idx_set, dst_bond_idxs)


def verify_chiral_consistency_of_core(mol_a: Chem.Mol, mol_b: Chem.Mol, core: NDArray, forcefield):
    """Verify that a core and forcefield would allow for valid chiral endstates.

    Refer to `assert_chiral_consistency_and_validity` for definitions of consistency and validity.

    Raises
    ------
        ChiralConversionError
            If chiral end states are incompatible for the given core and forcefield.
    """
    atom_map = AtomMapMixin(mol_a, mol_b, core)
    bond_pot_src, chiral_atom_pot_src, _ = setup_end_state_harmonic_bond_and_chiral_potentials(
        forcefield, mol_a, mol_b, core, atom_map.a_to_c, atom_map.b_to_c
    )
    bond_pot_dest, chiral_atom_pot_dest, _ = setup_end_state_harmonic_bond_and_chiral_potentials(
        forcefield, mol_b, mol_a, core[:, ::-1], atom_map.b_to_c, atom_map.a_to_c
    )
    assert_chiral_consistency_and_validity(
        atom_map,
        chiral_atom_pot_src.potential.idxs,
        chiral_atom_pot_dest.potential.idxs,
        bond_pot_src.potential.idxs,
        bond_pot_dest.potential.idxs,
    )


class SingleTopology(AtomMapMixin):
    def __init__(self, mol_a, mol_b, core, forcefield):
        """
        SingleTopology combines two molecules through a common core. The combined mol has
        atom indices laid out such that mol_a is identically mapped to the combined mol indices.
        The atoms in the mol_b's R-group is then glued on to resulting molecule.

        Parameters
        ----------
        mol_a: ROMol
            First guest

        mol_b: ROMol
            Second guest

        core: np.array (C, 2)
            Atom mapping from mol_a to mol_b.

        forcefield: ff.Forcefield
            Forcefield to be used for parameterization.
        """
        # initialize the mixin to get the a_to_c, b_to_c, c_to_a, c_to_b, and c_flags
        super().__init__(mol_a, mol_b, core)

        # store the forcefield
        self.ff = forcefield

        a_charge = Chem.GetFormalCharge(mol_a)
        b_charge = Chem.GetFormalCharge(mol_b)
        if a_charge != b_charge:
            raise ChargePertubationError(f"mol a and mol b don't have the same charge: a: {a_charge} b: {b_charge}")

        # setup end states
        self.src_system = self._setup_end_state_src()
        self.dst_system = self._setup_end_state_dst()

        assert_chiral_consistency_and_validity(
            self,
            self.src_system.chiral_atom.potential.idxs,
            self.dst_system.chiral_atom.potential.idxs,
            self.src_system.bond.potential.idxs,
            self.dst_system.bond.potential.idxs,
        )

    def combine_masses(self, use_hmr=False):
        """
        Combine masses between two end-states by taking the heavier of the two core atoms.

        Returns
        -------
        masses: list of float
            len(masses) == self.get_num_atoms()
        """
        mol_a_masses = utils.get_mol_masses(self.mol_a)
        mol_b_masses = utils.get_mol_masses(self.mol_b)

        # with HMR, apply to each molecule independently
        # then use the larger value for core atoms and the
        # HMR value for dummy atoms
        if use_hmr:
            # Can't use src_system, dst_system as these have dummy atoms attached
            mol_a_top = topology.BaseTopology(self.mol_a, self.ff)
            mol_b_top = topology.BaseTopology(self.mol_b, self.ff)
            _, mol_a_hb = mol_a_top.parameterize_harmonic_bond(self.ff.hb_handle.params)
            _, mol_b_hb = mol_b_top.parameterize_harmonic_bond(self.ff.hb_handle.params)

            mol_a_masses = model_utils.apply_hmr(mol_a_masses, mol_a_hb.idxs)
            mol_b_masses = model_utils.apply_hmr(mol_b_masses, mol_b_hb.idxs)

        mol_c_masses = []
        for c_idx in range(self.get_num_atoms()):
            indicator = self.c_flags[c_idx]
            if indicator == 0:
                mass_a = mol_a_masses[self.c_to_a[c_idx]]
                mass_b = mol_b_masses[self.c_to_b[c_idx]]
                mass = max(mass_a, mass_b)
            elif indicator == 1:
                mass = mol_a_masses[self.c_to_a[c_idx]]
            elif indicator == 2:
                mass = mol_b_masses[self.c_to_b[c_idx]]
            else:
                assert 0

            mol_c_masses.append(mass)

        return mol_c_masses

    def combine_confs(self, x_a, x_b, lamb=1.0):
        """
        Combine conformations of two molecules.

        TODO: interpolate confs based on the lambda value?

        Parameters
        ----------
        x_a: np.array of shape (N_A,3)
            First conformation

        x_b: np.array of shape (N_B,3)
            Second conformation

        lamb: optional float
            if lamb > 0.5, map atoms from x_a first, then overwrite with x_b,
            otherwise use opposite order

        Returns
        -------
        np.array of shape (self.num_atoms,3)
            Combined conformation

        """
        if lamb < 0.5:
            return self.combine_confs_lhs(x_a, x_b)
        else:
            return self.combine_confs_rhs(x_a, x_b)

    def combine_confs_rhs(self, x_a, x_b):
        """
        Combine x_a and x_b conformations for lambda=1
        """
        # place a first, then b overrides a
        assert x_a.shape == (self.mol_a.GetNumAtoms(), 3)
        assert x_b.shape == (self.mol_b.GetNumAtoms(), 3)
        x0 = np.zeros((self.get_num_atoms(), 3))
        for src, dst in enumerate(self.a_to_c):
            x0[dst] = x_a[src]
        for src, dst in enumerate(self.b_to_c):
            x0[dst] = x_b[src]

        return x0

    def combine_confs_lhs(self, x_a, x_b):
        """
        Combine x_a and x_b conformations for lambda=0
        """
        # place b first, then a overrides b
        assert x_a.shape == (self.mol_a.GetNumAtoms(), 3)
        assert x_b.shape == (self.mol_b.GetNumAtoms(), 3)
        x0 = np.zeros((self.get_num_atoms(), 3))
        for src, dst in enumerate(self.b_to_c):
            x0[dst] = x_b[src]
        for src, dst in enumerate(self.a_to_c):
            x0[dst] = x_a[src]

        return x0

    def _setup_end_state_src(self):
        """
        Setup the source end-state, where mol_a is fully interacting, with mol_b's dummy atoms attached
        in a factorizable way.

        Returns
        -------
        VacuumSystem
            Gas-phase system
        """
        return setup_end_state(self.ff, self.mol_a, self.mol_b, self.core, self.a_to_c, self.b_to_c)

    def _setup_end_state_dst(self):
        """
        Setup the source end-state, where mol_b is fully interacting, with mol_a's dummy atoms attached
        in a factorizable way.

        Returns
        -------
        VacuumSystem
            Gas-phase system
        """
        return setup_end_state(self.ff, self.mol_b, self.mol_a, self.core[:, ::-1], self.b_to_c, self.a_to_c)

    def _setup_intermediate_bonded_term(
        self, src_bond: BoundPotential[_Bonded], dst_bond: BoundPotential[_Bonded], lamb, align_fn, interpolate_fn
    ) -> BoundPotential[_Bonded]:
        src_cls_bond = type(src_bond.potential)
        dst_cls_bond = type(dst_bond.potential)

        assert src_cls_bond == dst_cls_bond

        bond_idxs_and_params = align_fn(
            src_bond.potential.idxs,
            src_bond.params,
            dst_bond.potential.idxs,
            dst_bond.params,
        )
        bond_idxs = np.array([x for x, _, _ in bond_idxs_and_params], dtype=np.int32)
        if bond_idxs_and_params:
            src_params = jnp.array([x for _, x, _ in bond_idxs_and_params])
            dst_params = jnp.array([x for _, _, x in bond_idxs_and_params])
            bond_params = jax.vmap(interpolate_fn, (0, 0, None))(src_params, dst_params, lamb)
        else:
            bond_params = jnp.array([])

        r = src_cls_bond(bond_idxs).bind(bond_params)
        return cast(BoundPotential[_Bonded], r)  # unclear why cast is needed for mypy

    def _setup_intermediate_nonbonded_term(
        self,
        src_nonbonded: BoundPotential[NonbondedPairListPrecomputed],
        dst_nonbonded: BoundPotential[NonbondedPairListPrecomputed],
        lamb,
        align_fn,
        interpolate_qlj_fn,
    ) -> BoundPotential[NonbondedPairListPrecomputed]:
        assert src_nonbonded.potential.beta == dst_nonbonded.potential.beta
        assert src_nonbonded.potential.cutoff == dst_nonbonded.potential.cutoff

        cutoff = src_nonbonded.potential.cutoff

        pair_idxs_and_params = align_fn(
            src_nonbonded.potential.idxs,
            src_nonbonded.params,
            dst_nonbonded.potential.idxs,
            dst_nonbonded.params,
        )

        pair_idxs = np.array([x for x, _, _ in pair_idxs_and_params], dtype=np.int32)

        if pair_idxs_and_params:
            src_params = jnp.array([x for _, x, _ in pair_idxs_and_params])
            dst_params = jnp.array([x for _, _, x in pair_idxs_and_params])

            src_qlj, src_w = src_params[:, :3], src_params[:, 3]
            dst_qlj, dst_w = dst_params[:, :3], dst_params[:, 3]

            is_excluded_src = jnp.all(src_qlj == 0.0, axis=1, keepdims=True)
            is_excluded_dst = jnp.all(dst_qlj == 0.0, axis=1, keepdims=True)

            # parameters for pairs that do not interact in the src state
            w = interpolate_w_coord(cutoff, dst_w, lamb)
            pair_params_excluded_src = jnp.concatenate((dst_qlj, w[:, None]), axis=1)

            # parameters for pairs that do not interact in the dst state
            w = interpolate_w_coord(src_w, cutoff, lamb)
            pair_params_excluded_dst = jnp.concatenate((src_qlj, w[:, None]), axis=1)

            # parameters for pairs that interact in both src and dst states
            w = jax.vmap(interpolate.linear_interpolation, (0, 0, None))(src_w, dst_w, lamb)
            qlj = interpolate_qlj_fn(src_qlj, dst_qlj, lamb)
            pair_params_not_excluded = jnp.concatenate((qlj, w[:, None]), axis=1)

            pair_params = jnp.where(
                is_excluded_src,
                pair_params_excluded_src,
                jnp.where(
                    is_excluded_dst,
                    pair_params_excluded_dst,
                    pair_params_not_excluded,
                ),
            )
        else:
            pair_params = jnp.array([])

        return NonbondedPairListPrecomputed(
            pair_idxs, src_nonbonded.potential.beta, src_nonbonded.potential.cutoff
        ).bind(pair_params)

    def _setup_intermediate_chiral_bond_term(
        self,
        src_bond: BoundPotential[ChiralBondRestraint],
        dst_bond: BoundPotential[ChiralBondRestraint],
        lamb,
        interpolate_fn,
    ) -> BoundPotential[ChiralBondRestraint]:
        assert isinstance(src_bond.potential, ChiralBondRestraint)
        assert isinstance(dst_bond.potential, ChiralBondRestraint)

        idxs_and_params = interpolate.align_chiral_bond_idxs_and_params(
            src_bond.potential.idxs,
            src_bond.params,
            src_bond.potential.signs,
            dst_bond.potential.idxs,
            dst_bond.params,
            dst_bond.potential.signs,
        )
        chiral_bond_idxs_ = []
        chiral_bond_params = []
        chiral_bond_signs = []
        for idxs, sign, src_k, dst_k in idxs_and_params:
            chiral_bond_idxs_.append(idxs)
            new_params = interpolate_fn(src_k, dst_k, lamb)
            chiral_bond_params.append(new_params)
            chiral_bond_signs.append(sign)

        # these should be properly sized
        chiral_bond_idxs = np.array(chiral_bond_idxs_, dtype=np.int32).reshape((-1, 4))

        return ChiralBondRestraint(chiral_bond_idxs, np.array(chiral_bond_signs)).bind(jnp.array(chiral_bond_params))

    def setup_intermediate_state(self, lamb) -> VacuumSystem:
        r"""
        Set up intermediate states at some value of the alchemical parameter :math:`\lambda`.

        Parameters
        ----------
        lamb: float

        Notes
        -----
        For transformations involving formation or deletion of valence terms (i.e., having force constants equal to zero
        in the :math:`\lambda=0` or :math:`\lambda=1` state), harmonic bond and angle terms are activated before
        torsions. This is to avoid a potential numerical instability in the torsion functional form when three atoms are
        collinear.

        - Bonds and angles with :math:`k=0` at :math:`\lambda=0` are activated in the interval :math:`0 \leq \lambda \leq 0.7`
        - Torsions with :math:`k=0` at :math:`\lambda=0` are activated in the interval :math:`0.7 \leq \lambda \leq 1.0`

        (and similarly for terms with :math:`k=0` at :math:`\lambda=0`, taking :math:`\lambda \to 1-\lambda` in the above.)

        Note that the above only applies to the interactions whose force constant is zero in one end state; otherwise,
        valence terms are interpolated simultaneously in the interval :math:`0 \leq \lambda \leq 1`)
        """
        src_system = self.src_system
        dst_system = self.dst_system

        # stagger the lambda schedule
        bonds_min, bonds_max = [0.0, 0.7]
        angles_min, angles_max = [0.0, 0.7]
        torsions_min, torsions_max = [0.7, 1.0]
        chiral_atoms_min, chiral_atoms_max = [0.7, 1.0]

        bond = self._setup_intermediate_bonded_term(
            src_system.bond,
            dst_system.bond,
            lamb,
            interpolate.align_harmonic_bond_idxs_and_params,
            partial(
                interpolate_harmonic_bond_params,
                k_min=0.1,  # ~ BOLTZ * (300 K) / (5 nm)^2
                lambda_min=bonds_min,
                lambda_max=bonds_max,
            ),
        )

        angle = self._setup_intermediate_bonded_term(
            src_system.angle,
            dst_system.angle,
            lamb,
            interpolate.align_harmonic_angle_idxs_and_params,
            partial(
                interpolate_harmonic_angle_params,
                k_min=0.05,  # ~ BOLTZ * (300 K) / (2 * pi)^2
                lambda_min=angles_min,
                lambda_max=angles_max,
            ),
        )

        assert src_system.torsion
        assert dst_system.torsion
        torsion = self._setup_intermediate_bonded_term(
            src_system.torsion,
            dst_system.torsion,
            lamb,
            interpolate.align_torsion_idxs_and_params,
            partial(interpolate_periodic_torsion_params, lambda_min=torsions_min, lambda_max=torsions_max),
        )

        nonbonded = self._setup_intermediate_nonbonded_term(
            src_system.nonbonded,
            dst_system.nonbonded,
            lamb,
            interpolate.align_nonbonded_idxs_and_params,
            interpolate.linear_interpolation,
        )

        assert src_system.chiral_atom
        assert dst_system.chiral_atom

        assert len(set(tuple(x) for x in src_system.chiral_atom.potential.idxs)) == len(
            src_system.chiral_atom.potential.idxs
        )
        assert len(set(tuple(x) for x in dst_system.chiral_atom.potential.idxs)) == len(
            dst_system.chiral_atom.potential.idxs
        )

        chiral_atom = self._setup_intermediate_bonded_term(
            src_system.chiral_atom,
            dst_system.chiral_atom,
            lamb,
            interpolate.align_chiral_atom_idxs_and_params,
            partial(
                interpolate_harmonic_force_constant,
                k_min=0.025,
                lambda_min=chiral_atoms_min,
                lambda_max=chiral_atoms_max,
            ),
        )

        assert src_system.chiral_bond
        assert dst_system.chiral_bond
        chiral_bond = self._setup_intermediate_chiral_bond_term(
            src_system.chiral_bond,
            dst_system.chiral_bond,
            lamb,
            interpolate.linear_interpolation,
        )

        return VacuumSystem(bond, angle, torsion, nonbonded, chiral_atom, chiral_bond)

    def mol(self, lamb, min_bond_k=100.0) -> Chem.Mol:
        """
        Generate an RDKit mol, with the dummy atoms attached to the molecule. Atom types and bond parameters
        guesstimated from the corresponding bond orders.

        Tricky-bits to figure out later on: Inferring bond orders and atom-types.

        Parameters
        ----------
        lamb: float
            Lambda value to use

        min_bond_k: float
            Minimum force constant required for a bond to be present in the mol

        Returns
        -------
        Chem.Mol
        """
        vs = self.setup_intermediate_state(lamb)
        N = self.get_num_atoms()

        # setup atoms
        mol_a_atomic_nums = [a.GetAtomicNum() for a in self.mol_a.GetAtoms()]
        mol_b_atomic_nums = [b.GetAtomicNum() for b in self.mol_b.GetAtoms()]
        mol = Chem.RWMol()

        for c_idx in range(N):
            if c_idx in self.c_to_a and c_idx in self.c_to_b:
                # core, in both mol_a and mol_b
                if lamb < 0.5:
                    atomic_num = mol_a_atomic_nums[self.c_to_a[c_idx]]
                else:
                    atomic_num = mol_b_atomic_nums[self.c_to_b[c_idx]]
            elif c_idx in self.c_to_a:
                # only in mol_a
                atomic_num = mol_a_atomic_nums[self.c_to_a[c_idx]]
            elif c_idx in self.c_to_b:
                # only in mol_b
                atomic_num = mol_b_atomic_nums[self.c_to_b[c_idx]]
            else:
                # in neither, assert
                assert 0
            atom = Chem.Atom(atomic_num)
            mol.AddAtom(atom)

        # setup bonds
        for (i, j), (k, b) in zip(vs.bond.potential.idxs, vs.bond.params):
            if k > min_bond_k:
                mol.AddBond(int(i), int(j), Chem.BondType.SINGLE)

        # make read-only
        return Chem.Mol(mol)

    def _get_guest_params(self, q_handle, lj_handle, lamb: float, cutoff: float) -> jax.Array:
        """
        Return an array containing the guest_charges, guest_sigmas, guest_epsilons, guest_w_coords
        for the guest at a given lambda.
        """
        guest_charges = []
        guest_sigmas = []
        guest_epsilons = []
        guest_w_coords = []

        # generate charges and lj parameters for each guest
        guest_a_q = q_handle.parameterize(self.mol_a)
        guest_a_lj = lj_handle.parameterize(self.mol_a)

        guest_b_q = q_handle.parameterize(self.mol_b)
        guest_b_lj = lj_handle.parameterize(self.mol_b)

        for idx, membership in enumerate(self.c_flags):
            if membership == 0:  # core atom
                a_idx = self.c_to_a[idx]
                b_idx = self.c_to_b[idx]

                # interpolate charges when in common-core
                q = (1 - lamb) * guest_a_q[a_idx] + lamb * guest_b_q[b_idx]
                sig = (1 - lamb) * guest_a_lj[a_idx, 0] + lamb * guest_b_lj[b_idx, 0]
                eps = (1 - lamb) * guest_a_lj[a_idx, 1] + lamb * guest_b_lj[b_idx, 1]

                # fixed at w = 0
                w = 0.0

            elif membership == 1:  # dummy_A
                a_idx = self.c_to_a[idx]
                q = guest_a_q[a_idx]
                sig = guest_a_lj[a_idx, 0]
                eps = guest_a_lj[a_idx, 1]

                # Decouple dummy group A as lambda goes from 0 to 1
                w = interpolate_w_coord(0.0, cutoff, lamb)

            elif membership == 2:  # dummy_B
                b_idx = self.c_to_b[idx]
                q = guest_b_q[b_idx]
                sig = guest_b_lj[b_idx, 0]
                eps = guest_b_lj[b_idx, 1]

                # Couple dummy group B as lambda goes from 0 to 1
                # NOTE: this is only for host-guest nonbonded ixns (there is no clash between A and B at lambda = 0.5)
                w = interpolate_w_coord(cutoff, 0.0, lamb)
            else:
                assert 0

            guest_charges.append(q)
            guest_sigmas.append(sig)
            guest_epsilons.append(eps)
            guest_w_coords.append(w)

        return jnp.stack(jnp.array([guest_charges, guest_sigmas, guest_epsilons, guest_w_coords]), axis=1)

    def _parameterize_host_nonbonded(self, host_nonbonded: BoundPotential[Nonbonded]) -> BoundPotential[Nonbonded]:
        """Parameterize host-host nonbonded interactions"""
        num_host_atoms = host_nonbonded.params.shape[0]
        num_guest_atoms = self.get_num_atoms()
        host_params = host_nonbonded.params
        cutoff = host_nonbonded.potential.cutoff
        beta = host_nonbonded.potential.beta

        exclusion_idxs = host_nonbonded.potential.exclusion_idxs
        scale_factors = host_nonbonded.potential.scale_factors

        # Note: The choice of zeros here is arbitrary. It doesn't affect the
        # potentials or grads, but any function like the seed could depend on these values.
        hg_nb_params = jnp.concatenate([host_params, np.zeros((num_guest_atoms, host_params.shape[1]))])

        combined_nonbonded = Nonbonded(
            num_host_atoms + num_guest_atoms,
            exclusion_idxs,
            scale_factors,
            beta,
            cutoff,
            atom_idxs=np.arange(num_host_atoms, dtype=np.int32),
        )

        return combined_nonbonded.bind(hg_nb_params)

    def _parameterize_host_guest_nonbonded_ixn(
        self, lamb, host_nonbonded: BoundPotential[Nonbonded], num_water_atoms: int
    ) -> BoundPotential[SummedPotential]:
        """Parameterize nonbonded interactions between the host and guest"""
        num_host_atoms = host_nonbonded.params.shape[0]
        num_guest_atoms = self.get_num_atoms()
        cutoff = host_nonbonded.potential.cutoff

        guest_ixn_water_params = self._get_guest_params(self.ff.q_handle_solv, self.ff.lj_handle_solv, lamb, cutoff)
        guest_ixn_other_params = self._get_guest_params(self.ff.q_handle, self.ff.lj_handle, lamb, cutoff)

        # L-W terms
        num_other_atoms = num_host_atoms - num_water_atoms

        def get_lig_idxs():
            return np.arange(num_guest_atoms, dtype=np.int32) + num_host_atoms

        def get_water_idxs():
            return np.arange(num_water_atoms, dtype=np.int32) + num_other_atoms

        def get_other_idxs():
            return np.arange(num_other_atoms, dtype=np.int32)

        ixn_pots, ixn_params = get_ligand_ixn_pots_params(
            get_lig_idxs(),
            get_water_idxs(),
            get_other_idxs(),
            host_nonbonded.params,
            guest_ixn_water_params,
            guest_ixn_other_params,
            beta=host_nonbonded.potential.beta,
            cutoff=cutoff,
        )

        sum_pot = SummedPotential(ixn_pots, ixn_params)
        bound_sum_pot = sum_pot.bind(jnp.concatenate(ixn_params).reshape((-1,)))

        return bound_sum_pot

    def combine_with_host(self, host_system: VacuumSystem, lamb: float, num_water_atoms: int) -> HostGuestSystem:
        """
        Setup host guest system. Bonds, angles, torsions, chiral_atom, chiral_bond and nonbonded terms are
        combined. In particular:

        1) Bond, angle, torsion, chiral_atom, chiral_bond simply have the idxs shifted by num_host_atoms.
        2) Host-host and host-guest interactions use a nonbonded potential, with exclusions set to the
            original host-host exclusions, in addition to *all* guest-guest interactions being excluded.
            Host-guest interactions are implemented as follows:
            i) at lambda = 0, the dummy atoms of mol_a are fully interacting, and become non-interacting at lambda = 1
            ii) at lambda = 0, the dummy atoms of mol_b are non-interacting, and become fully interacting at lambda = 1
            iii) the core atoms have parameters interpolated from mol_a's qlj to mol_b's qlj.
        3) guest-guest interactions use pre-computed nonbonded interactions implemented as a pairlist.

        Parameters
        ----------
        host_system: VacuumSystem
            Parameterized system of the host

        lamb: float
            Which lambda value we want to generate the combined system.

        num_water_atoms: int
            Number of water atoms as part of the host.

        Returns
        -------
        7-tuple of potentials
            bond, angle, torsion, chiral_atom, chiral_bond, guest_nonbonded_precomputed, full_nonbonded

        """

        guest_system = self.setup_intermediate_state(lamb=lamb)

        num_host_atoms = host_system.nonbonded.params.shape[0]

        assert guest_system.chiral_atom
        guest_chiral_atom_idxs = np.array(guest_system.chiral_atom.potential.idxs, dtype=np.int32) + num_host_atoms
        guest_system.chiral_atom.potential.idxs = guest_chiral_atom_idxs

        assert guest_system.chiral_bond
        guest_chiral_bond_idxs = np.array(guest_system.chiral_bond.potential.idxs, dtype=np.int32) + num_host_atoms
        guest_system.chiral_bond.potential.idxs = guest_chiral_bond_idxs

        guest_nonbonded_idxs = np.array(guest_system.nonbonded.potential.idxs, dtype=np.int32) + num_host_atoms
        guest_system.nonbonded.potential.idxs = guest_nonbonded_idxs

        combined_bond_idxs = np.concatenate(
            [host_system.bond.potential.idxs, guest_system.bond.potential.idxs + num_host_atoms]
        )
        combined_bond_params = jnp.concatenate([host_system.bond.params, guest_system.bond.params])
        combined_bond = HarmonicBond(combined_bond_idxs).bind(combined_bond_params)

        combined_angle_idxs = np.concatenate(
            [host_system.angle.potential.idxs, guest_system.angle.potential.idxs + num_host_atoms]
        )
        host_angle_params = jnp.hstack(
            [
                host_system.angle.params,
                np.zeros((host_system.angle.params.shape[0], 1)),  # eps = 0
            ]
        )
        combined_angle_params = jnp.concatenate([host_angle_params, guest_system.angle.params])
        combined_angle = HarmonicAngleStable(combined_angle_idxs).bind(combined_angle_params)

        assert guest_system.torsion

        # complex proteins have torsions
        if host_system.torsion:
            combined_torsion_idxs = np.concatenate(
                [host_system.torsion.potential.idxs, guest_system.torsion.potential.idxs + num_host_atoms]
            )
            combined_torsion_params = jnp.concatenate([host_system.torsion.params, guest_system.torsion.params])
        else:
            # solvent waters don't have torsions
            combined_torsion_idxs = np.array(guest_system.torsion.potential.idxs, dtype=np.int32) + num_host_atoms
            combined_torsion_params = jnp.array(guest_system.torsion.params)

        combined_torsion = PeriodicTorsion(combined_torsion_idxs).bind(combined_torsion_params)

        host_nonbonded = self._parameterize_host_nonbonded(host_system.nonbonded)
        host_guest_nonbonded_ixn = self._parameterize_host_guest_nonbonded_ixn(
            lamb, host_system.nonbonded, num_water_atoms
        )

        return HostGuestSystem(
            combined_bond,
            combined_angle,
            combined_torsion,
            guest_system.chiral_atom,
            guest_system.chiral_bond,
            guest_system.nonbonded,
            host_nonbonded,
            host_guest_nonbonded_ixn,
        )

    def get_component_idxs(self) -> List[NDArray]:
        """
        Return the atom indices for the two ligands in
        this topology as a list of NDArray. Both lists
        will contain atom indices for the core atoms
        as well as the unique atom indices for each ligand.
        """
        return [self.a_to_c, self.b_to_c]
