import warnings
from collections.abc import Collection, Sequence
from enum import IntEnum
from functools import partial
from typing import Any, Optional

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
from numpy.typing import NDArray
from rdkit import Chem

from timemachine.constants import (
    DEFAULT_BOND_IS_PRESENT_K,
    DEFAULT_CHIRAL_ATOM_RESTRAINT_K,
    DEFAULT_CHIRAL_BOND_RESTRAINT_K,
    NBParamIdx,
)
from timemachine.fe import interpolate, model_utils, topology, utils
from timemachine.fe.chiral_utils import ChiralRestrIdxSet
from timemachine.fe.dummy import (
    canonicalize_bond,
    generate_anchored_dummy_group_assignments,
    generate_dummy_group_assignments,
)
from timemachine.fe.interpolate import pad
from timemachine.fe.lambda_schedule import construct_pre_optimized_relative_lambda_schedule
from timemachine.fe.system import GuestSystem, HostGuestSystem, HostSystem
from timemachine.fe.topology import get_ligand_ixn_pots_params
from timemachine.ff import Forcefield
from timemachine.graph_utils import convert_to_nx
from timemachine.potentials import (
    BoundPotential,
    ChiralAtomRestraint,
    ChiralBondRestraint,
    HarmonicAngleStable,
    HarmonicBond,
    Nonbonded,
    NonbondedInteractionGroup,
    NonbondedPairListPrecomputed,
    PeriodicTorsion,
)

OpenMMTopology = Any


# Master Schedule
# tbd: separate torsions into proper and improper later
def _flip_min_max(min_max):
    """(0, 0.5) -> (1, 0.5); (0.2, 1) -> (0, 0.8); (0,1) -> (0,1)"""
    lamb_min, lamb_max = min_max
    return 1 - lamb_max, 1 - lamb_min


# (ytz): note that the boundary values 0.0, 0.3, 0.5, 0.7 etc. below are arbitrary, and are free-ish parameters
# that can overlap as well. Strict boundaries are probably the safest, but come at the cost of efficiency.

# eg, an alternative, stricter, and safer dummy chiral/bond/interpolation scheme would be:
# DUMMY_B_CHIRAL_BOND_CONVERTING_ON_MIN_MAX = [0.0, 0.3]
# which would necessitate chiral bonds being *fully* turned on before the chiral volume and chiral angle terms are turned on.
# however, this led to roughly a 10-15% increase in the # of windows. So instead, we can switch to a softer scheme by setting
# the bond interpolation bounds to [0.0, 0.7]. This way, when lambda=0.3, the start of the chiral volume interpolation, the bonds
# are still present with a force constant of ~30kJ/mol (still strong enough to keep the chiral volumes numerically stable)

# core-bonds are never turned off if enforce_core_core=True
CORE_BOND_MIN_MAX = [0.0, 1.0]
CORE_ANGLE_MIN_MAX = [0.0, 1.0]
CORE_TORSION_MIN_MAX = [0.0, 1.0]
CORE_TORSION_OFF_TO_ON_MIN_MAX = [0.7, 1.0]
CORE_TORSION_ON_TO_OFF_MIN_MAX = _flip_min_max(CORE_TORSION_OFF_TO_ON_MIN_MAX)


# core terms that are involved in chiral volumes being turned on or off
CORE_CHIRAL_ATOM_CONVERTING_ON_MIN_MAX = [0.0, 0.5]
CORE_CHIRAL_ANGLE_CONVERTING_ON_MIN_MAX = [0.5, 1.0]
CORE_CHIRAL_ATOM_CONVERTING_OFF_MIN_MAX = _flip_min_max(CORE_CHIRAL_ATOM_CONVERTING_ON_MIN_MAX)
CORE_CHIRAL_ANGLE_CONVERTING_OFF_MIN_MAX = _flip_min_max(CORE_CHIRAL_ANGLE_CONVERTING_ON_MIN_MAX)

# non-converting (may be consistently in chirality or just achiral) dummy B groups that are turning on
DUMMY_B_BOND_MIN_MAX = [0.0, 0.7]
DUMMY_B_ANGLE_MIN_MAX = [0.0, 0.7]
DUMMY_A_BOND_MIN_MAX = _flip_min_max(DUMMY_B_BOND_MIN_MAX)
DUMMY_A_ANGLE_MIN_MAX = _flip_min_max(DUMMY_B_ANGLE_MIN_MAX)

# chiral and converting dummy B groups are turning on
DUMMY_B_CHIRAL_BOND_CONVERTING_ON_MIN_MAX = [0.0, 0.7]
DUMMY_B_CHIRAL_ATOM_CONVERTING_ON_MIN_MAX = [0.3, 0.5]
DUMMY_B_CHIRAL_ANGLE_CONVERTING_ON_MIN_MAX = [0.5, 0.7]  # angles are all turned off

# chiral and converting dummy A groups are turning off
DUMMY_A_CHIRAL_BOND_CONVERTING_OFF_MIN_MAX = _flip_min_max(DUMMY_B_CHIRAL_BOND_CONVERTING_ON_MIN_MAX)
DUMMY_A_CHIRAL_ATOM_CONVERTING_OFF_MIN_MAX = _flip_min_max(DUMMY_B_CHIRAL_ATOM_CONVERTING_ON_MIN_MAX)
DUMMY_A_CHIRAL_ANGLE_CONVERTING_OFF_MIN_MAX = _flip_min_max(DUMMY_B_CHIRAL_ANGLE_CONVERTING_ON_MIN_MAX)

# torsions are the same throughout
DUMMY_B_TORSION_MIN_MAX = [0.7, 1.0]  # chiral and achiral both use the same min/max for torsions
DUMMY_A_TORSION_MIN_MAX = _flip_min_max(DUMMY_B_TORSION_MIN_MAX)


class ChiralVolumeDisabledWarning(UserWarning):
    pass


class CoreBondChangeWarning(UserWarning):
    pass


class MissingAngleError(RuntimeError):
    pass


class ChargePertubationError(RuntimeError):
    pass


class DummyGroupAssignmentError(RuntimeError):
    pass


def bond_isin(bonds: NDArray[np.int32], idxs: NDArray[np.int32]) -> NDArray[np.bool_]:
    """Returns a boolean mask indicating whether both indices of a bond are contained in idxs.

    Parameters
    ----------
    bonds: NDArray
        (n_bonds, n_bond_idxs) array of bonds (n_bond_idxs=2 for bonds, 3 for angles, 4 for torsions, etc.)

    idxs: NDArray
        1-d array of atom indices
    """
    b0 = bonds[:, :, None] == idxs[None, None, :]  # shape: (n_bonds, n_bond_idxs, n_idxs)
    b1 = b0.any(-1)  # shape: (n_bonds, n_bond_idxs)
    b2 = b1.all(-1)  # shape: (n_bonds,)
    return b2


def setup_dummy_bond_and_chiral_interactions(
    bond_idxs: NDArray,
    bond_params: NDArray,
    chiral_atom_idxs: NDArray,
    chiral_atom_params: NDArray,
    dummy_group: frozenset[int],
    root_anchor_atom: int,
    core_atoms: NDArray,
):
    assert root_anchor_atom in core_atoms

    dummy_group_arr = np.array(list(dummy_group))

    # dummy group and anchor
    dga = np.append(dummy_group_arr, root_anchor_atom)

    # keep interactions that involve only root_anchor_atom
    bond_mask = bond_isin(bond_idxs, dga)
    dummy_bond_idxs = bond_idxs[bond_mask]
    dummy_bond_params = bond_params[bond_mask]

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

    dgc = np.concatenate([dummy_group_arr, core_atoms])

    # mask indicating whether a chiral atom ixn has a non-center dummy atom
    has_ncda = (chiral_atom_idxs[:, 1:, None] == dummy_group_arr[None, None, :]).any(-1).any(-1)

    chiral_atom_ixn_mask = bond_isin(chiral_atom_idxs, dgc) & has_ncda
    dummy_chiral_atom_idxs = chiral_atom_idxs[chiral_atom_ixn_mask]
    dummy_chiral_atom_params = chiral_atom_params[chiral_atom_ixn_mask]

    bonded_idxs = (dummy_bond_idxs, dummy_chiral_atom_idxs)
    bonded_params = (dummy_bond_params, dummy_chiral_atom_params)

    return bonded_idxs, bonded_params


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

    dummy_angle_idxs = []
    dummy_angle_params = []
    dummy_improper_idxs = []
    dummy_improper_params = []

    (
        (dummy_bond_idxs, dummy_chiral_atom_idxs),
        (
            dummy_bond_params,
            dummy_chiral_atom_params,
        ),
    ) = setup_dummy_bond_and_chiral_interactions(
        bond_idxs,
        bond_params,
        chiral_atom_idxs,
        chiral_atom_params,
        dummy_group,
        root_anchor_atom,
        core_atoms,
    )

    # dummy_group may be a set in certain cases, so sanity check.

    assert len(dummy_group) == len(list(dummy_group))
    dummy_group = list(dummy_group)

    # dummy group and anchor
    dga = [*dummy_group, root_anchor_atom]

    for idxs, params in zip(angle_idxs, angle_params):
        if all([a in dga for a in idxs]):
            dummy_angle_idxs.append(tuple([int(x) for x in idxs]))
            dummy_angle_params.append(params)
    for idxs, params in zip(improper_idxs, improper_params):
        if all([a in dga for a in idxs]):
            dummy_improper_idxs.append(tuple([int(x) for x in idxs]))
            dummy_improper_params.append(params)

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


def canonicalize_bonds(bonds: NDArray[np.int32]) -> NDArray[np.int32]:
    assert bonds.ndim == 2
    assert bonds.shape[1] >= 2
    is_canonical = bonds[:, 0] < bonds[:, -1]
    return np.where(is_canonical[:, None], bonds, bonds[:, ::-1])


def canonicalize_improper_idxs(idxs) -> tuple[int, int, int, int]:
    """
    Canonicalize an improper_idx while being symmetry aware.

    Given idxs (j,c,k,l), where c is the center, and (j,k,l) are neighbors:

    0) Canonicalize the (j,k,l) into (jj,kk,ll) by sorting
    1) Generate clockwise rotations of (jj,kk,ll)
    2) Generate counter clockwise rotations of (jj,kk,ll)
    3) We now can sort 1) and 2) and assign a mapping

    If the (j,k,l) is in the cw rotation ordered set, we're done. Otherwise it must
    be in the ccw ordered set. We look up the corresponding idx in the cw set.

    This does not do idxs[0] < idxs[-1] canonicalization.
    """
    j, c, k, l = idxs

    # c is the center
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
        return (j, c, k, l)

    # generate counter clockwise permutations
    ccw_kjl = (kk, jj, ll)  # swap 1st and 2nd element
    ccw_jlk = (jj, ll, kk)  # rotate left
    ccw_lkj = (ll, kk, jj)  # rotate left
    ccw_items = sorted([ccw_kjl, ccw_jlk, ccw_lkj])

    assert key in ccw_items

    for idx, cw_item in enumerate(ccw_items):
        if cw_item == key:
            break

    j, k, l = cw_items[idx]

    return (j, c, k, l)


def get_num_connected_components(num_atoms: int, bonds: Collection[tuple[int, int]]) -> int:
    g = nx.Graph()
    g.add_nodes_from(range(num_atoms))
    g.add_edges_from(bonds)
    return len(list(nx.connected_components(g)))


def canonicalize_chiral_atom_idxs(idxs: NDArray[np.int32]) -> NDArray[np.int32]:
    assert idxs.ndim == 2
    assert idxs.shape[1] == 4
    c = idxs[:, 0:1]
    ijk = idxs[:, 1:]
    ijk_argmin = np.argmin(ijk, axis=1)
    ijks = ijk[:, [[0, 1, 2], [1, 2, 0], [2, 0, 1]]]
    ijk_canon = np.take_along_axis(ijks, ijk_argmin[:, None, None], axis=1)[:, 0]
    return np.concatenate([c, ijk_canon], axis=1)


def setup_end_state(
    ff: Forcefield,
    mol_a: Chem.Mol,
    mol_b: Chem.Mol,
    core: NDArray,
    a_to_c: NDArray,
    b_to_c: NDArray,
    anchored_dummy_groups: dict[int, tuple[Optional[int], frozenset[int]]],
) -> GuestSystem:
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

    core: array of shape (core_size, 2)
        Each pair is an atom mapping from mol_a into mol_b

    a_to_c: array
        mapping from a into a common core idx

    b_to_c: array
        mapping from b into a common core idx

    anchored_dummy_groups: dict[int, tuple[Optional[int], frozenset[int]]]
        mapping from anchor atom to (optional) angle anchor and dummy group. Indices refer to atoms in mol_b.

    Returns
    -------
    GuestSystem
        A parameterized system.

    """

    all_dummy_angle_idxs_, all_dummy_angle_params_ = [], []
    all_dummy_improper_idxs_, all_dummy_improper_params_ = [], []
    for anchor, (nbr, dg) in anchored_dummy_groups.items():
        all_idxs, all_params = setup_dummy_interactions_from_ff(
            ff, mol_b, dg, anchor, nbr, core[:, 1], DEFAULT_CHIRAL_ATOM_RESTRAINT_K, DEFAULT_CHIRAL_BOND_RESTRAINT_K
        )
        # append idxs
        all_dummy_angle_idxs_.extend(all_idxs[1])
        all_dummy_improper_idxs_.extend(all_idxs[2])
        # append params
        all_dummy_angle_params_.extend(all_params[1])
        all_dummy_improper_params_.extend(all_params[2])

    all_dummy_angle_idxs = np.array(all_dummy_angle_idxs_, np.int32).reshape(-1, 3)
    all_dummy_angle_params = np.array(all_dummy_angle_params_, np.float64).reshape(-1, 2)

    all_dummy_improper_idxs = np.array(all_dummy_improper_idxs_, np.int32).reshape(-1, 4)
    all_dummy_improper_params = np.array(all_dummy_improper_params_, np.float64).reshape(-1, 3)

    # generate parameters for mol_a
    mol_a_top = topology.BaseTopology(mol_a, ff)
    assert ff.ha_handle is not None
    mol_a_angle_params, mol_a_ha = mol_a_top.parameterize_harmonic_angle(ff.ha_handle.params)
    assert ff.pt_handle is not None
    mol_a_proper_params, mol_a_pt = mol_a_top.parameterize_proper_torsion(ff.pt_handle.params)
    assert ff.it_handle is not None
    mol_a_improper_params, mol_a_it = mol_a_top.parameterize_improper_torsion(ff.it_handle.params)

    assert ff.q_handle is not None
    assert ff.q_handle_intra is not None

    assert ff.lj_handle is not None
    assert ff.lj_handle_intra is not None
    mol_a_nbpl_params, mol_a_nbpl = mol_a_top.parameterize_nonbonded_pairlist(
        ff.q_handle.params,
        ff.q_handle_intra.params,
        ff.lj_handle.params,
        ff.lj_handle_intra.params,
        intramol_params=True,
    )

    mol_a_angle_idxs = a_to_c[mol_a_ha.idxs]
    mol_a_proper_idxs = a_to_c[mol_a_pt.idxs]
    mol_a_improper_idxs = a_to_c[mol_a_it.idxs]
    mol_a_nbpl_idxs = a_to_c[mol_a_nbpl.idxs]

    all_dummy_angle_idxs = b_to_c[all_dummy_angle_idxs]  # type: ignore
    all_dummy_improper_idxs = b_to_c[all_dummy_improper_idxs]  # type: ignore

    mol_c_angle_idxs = np.concatenate([mol_a_angle_idxs, all_dummy_angle_idxs])
    mol_c_angle_params = np.concatenate([mol_a_angle_params, all_dummy_angle_params])

    mol_c_proper_idxs = np.array([canonicalize_bond(idxs) for idxs in mol_a_proper_idxs], dtype=np.int32)
    mol_c_proper_params = mol_a_proper_params
    proper_potential = PeriodicTorsion(mol_c_proper_idxs.reshape(-1, 4)).bind(
        np.array(mol_c_proper_params.reshape(-1, 3), dtype=np.float64)
    )

    mol_c_improper_idxs = np.concatenate([mol_a_improper_idxs, all_dummy_improper_idxs])
    mol_c_improper_params = np.concatenate([mol_a_improper_params, all_dummy_improper_params])
    # canonicalize improper with clockwise/counter-clockwise check
    mol_c_improper_idxs = np.array(
        [canonicalize_improper_idxs(idxs) for idxs in mol_c_improper_idxs], np.int32
    ).reshape(-1, 4)
    improper_potential = PeriodicTorsion(mol_c_improper_idxs).bind(
        np.array(mol_c_improper_params.reshape(-1, 3), dtype=np.float64)
    )

    # canonicalize angles
    mol_c_angle_idxs_canon = np.array([canonicalize_bond(idxs) for idxs in mol_c_angle_idxs], dtype=np.int32)
    # Set the stable angle epsilon values to zero
    mol_c_stable_angle_params = np.hstack([mol_c_angle_params, np.zeros((len(mol_c_angle_params), 1))])
    angle_potential = HarmonicAngleStable(mol_c_angle_idxs_canon).bind(mol_c_stable_angle_params)

    # dummy atoms do not have any nonbonded interactions, so we simply turn them off
    mol_c_nbpl_idxs_canon = np.array([canonicalize_bond(idxs) for idxs in mol_a_nbpl_idxs], dtype=np.int32)
    mol_a_nbpl.idxs = mol_c_nbpl_idxs_canon
    nonbonded_potential = mol_a_nbpl.bind(np.array(mol_a_nbpl_params, dtype=np.float64))

    assert ff.hb_handle

    mol_a_top = topology.BaseTopology(mol_a, ff)
    mol_a_bond_params, mol_a_hb = mol_a_top.parameterize_harmonic_bond(ff.hb_handle.params)
    mol_a_chiral_atom, mol_a_chiral_bond = mol_a_top.setup_chiral_restraints(
        DEFAULT_CHIRAL_ATOM_RESTRAINT_K, DEFAULT_CHIRAL_BOND_RESTRAINT_K
    )

    mol_b_top = topology.BaseTopology(mol_b, ff)
    mol_b_bond_params, mol_b_hb = mol_b_top.parameterize_harmonic_bond(ff.hb_handle.params)
    mol_b_chiral_atom, _ = mol_b_top.setup_chiral_restraints(
        DEFAULT_CHIRAL_ATOM_RESTRAINT_K, DEFAULT_CHIRAL_BOND_RESTRAINT_K
    )

    all_dummy_bond_idxs_, all_dummy_bond_params_ = [], []
    all_dummy_chiral_atom_idxs_, all_dummy_chiral_atom_params_ = [], []

    for anchor, (_, dg) in anchored_dummy_groups.items():
        all_idxs, all_params = setup_dummy_bond_and_chiral_interactions(
            mol_b_hb.idxs,
            mol_b_bond_params,
            mol_b_chiral_atom.potential.idxs,
            np.asarray(mol_b_chiral_atom.params),
            dg,
            anchor,
            core[:, 1],
        )
        # append idxs
        all_dummy_bond_idxs_.append(all_idxs[0])
        all_dummy_chiral_atom_idxs_.append(all_idxs[1])
        # append params
        all_dummy_bond_params_.append(all_params[0])
        all_dummy_chiral_atom_params_.append(all_params[1])

    def concatenate(arrays, empty_shape, empty_dtype):
        return np.concatenate(arrays) if len(arrays) > 0 else np.empty(empty_shape, empty_dtype)

    all_dummy_bond_idxs = concatenate(all_dummy_bond_idxs_, (0, 2), np.int32)
    all_dummy_bond_params = concatenate(all_dummy_bond_params_, (0, 2), np.float64)

    all_dummy_chiral_atom_idxs = concatenate(all_dummy_chiral_atom_idxs_, (0, 4), np.int32)
    all_dummy_chiral_atom_params = concatenate(all_dummy_chiral_atom_params_, (0,), np.float64)

    mol_a_bond_idxs = a_to_c[mol_a_hb.idxs]
    mol_a_chiral_atom_idxs = a_to_c[mol_a_chiral_atom.potential.idxs]
    mol_a_chiral_bond_idxs = a_to_c[mol_a_chiral_bond.potential.idxs]

    all_dummy_bond_idxs = b_to_c[all_dummy_bond_idxs]
    all_dummy_chiral_atom_idxs = b_to_c[all_dummy_chiral_atom_idxs]

    # parameterize the combined molecule
    mol_c_bond_idxs = np.concatenate([mol_a_bond_idxs, all_dummy_bond_idxs])
    mol_c_bond_params = np.concatenate([mol_a_bond_params, all_dummy_bond_params])

    # process chiral volumes, turning off ones at the end-state that have a missing bond.

    # assert presence of bonds
    canon_mol_a_bond_idxs_set = {tuple(x) for x in canonicalize_bonds(mol_a_bond_idxs)}
    for c, i, j, k in mol_a_chiral_atom_idxs:
        ci = canonicalize_bond((c, i))
        cj = canonicalize_bond((c, j))
        ck = canonicalize_bond((c, k))
        assert ci in canon_mol_a_bond_idxs_set
        assert cj in canon_mol_a_bond_idxs_set
        assert ck in canon_mol_a_bond_idxs_set

    mol_c_bond_idxs_set = {tuple(x) for x in mol_c_bond_idxs}

    # Chiral atom restraint c,i,j,k requires that all bonds ci, cj, ck be present at the
    # end-state in order to be numerically stable under small perturbations due to normalization
    # along the bond lengths. However, the angle terms defining icj, ick, and jck can be
    # either 0 or 180, since the normalized chiral volume is still smooth wrt perturbations
    all_proper_dummy_chiral_atom_idxs_ = []
    all_proper_dummy_chiral_atom_params_ = []

    for (c, i, j, k), p in zip(all_dummy_chiral_atom_idxs, all_dummy_chiral_atom_params):
        missing_bonds = []
        for x in [i, j, k]:
            if (c, x) not in mol_c_bond_idxs_set and (x, c) not in mol_c_bond_idxs_set:
                missing_bonds.append((int(c), int(x)))

        if len(missing_bonds) == 0:
            all_proper_dummy_chiral_atom_idxs_.append((c, i, j, k))
            all_proper_dummy_chiral_atom_params_.append(p)
        else:
            warnings.warn(
                f"Chiral Volume {int(c), int(i), int(j), int(k)} has disabled bonds {missing_bonds}, turning off.",
                ChiralVolumeDisabledWarning,
            )

    all_proper_dummy_chiral_atom_idxs = np.array(all_proper_dummy_chiral_atom_idxs_, np.int32).reshape(-1, 4)
    all_proper_dummy_chiral_atom_params = np.array(all_proper_dummy_chiral_atom_params_, np.float64)

    mol_c_chiral_atom_idxs = np.concatenate([mol_a_chiral_atom_idxs, all_proper_dummy_chiral_atom_idxs])
    mol_c_chiral_atom_params = np.concatenate([mol_a_chiral_atom.params, all_proper_dummy_chiral_atom_params])

    # canonicalize bonds
    mol_c_bond_idxs_canon = canonicalize_bonds(mol_c_bond_idxs)
    bond_potential = HarmonicBond(mol_c_bond_idxs_canon).bind(np.array(mol_c_bond_params))

    # chiral atoms need special code for canonicalization, since triple product is invariant
    # under rotational symmetry (but not something like swap symmetry)
    mol_c_chiral_atom_idxs = canonicalize_chiral_atom_idxs(mol_c_chiral_atom_idxs)

    mol_c_chiral_bond_idxs = canonicalize_bonds(mol_a_chiral_bond_idxs)
    mol_c_chiral_bond_signs = mol_a_chiral_bond.potential.signs

    chiral_atom_potential = ChiralAtomRestraint(mol_c_chiral_atom_idxs).bind(mol_c_chiral_atom_params)
    chiral_bond_potential = ChiralBondRestraint(mol_c_chiral_bond_idxs, mol_c_chiral_bond_signs).bind(
        mol_a_chiral_bond.params
    )

    num_atoms = mol_a.GetNumAtoms() + mol_b.GetNumAtoms() - len(core)
    assert get_num_connected_components(num_atoms, bond_potential.potential.idxs) == 1, (
        "hybrid molecule has multiple connected components"
    )

    return GuestSystem(
        bond=bond_potential,
        angle=angle_potential,
        proper=proper_potential,
        improper=improper_potential,
        nonbonded_pair_list=nonbonded_potential,
        chiral_atom=chiral_atom_potential,
        chiral_bond=chiral_bond_potential,
    )


def find_dummy_groups_and_anchors(
    mol_a,
    mol_b,
    core_atoms_a: Sequence[int],
    core_atoms_b: Sequence[int],
) -> dict[int, tuple[Optional[int], frozenset[int]]]:
    """Returns an arbitrary dummy group assignment for the A -> B transformation.

    Refer to :py:func:`assert_chiral_consistency` for definition of "chiral consistency".

    Refer to :py:func:`timemachine.fe.dummy.generate_dummy_group_assignments` and notes below for more
    information on dummy group assignment.

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

    bond_graph_a = convert_to_nx(mol_a)
    bond_graph_b = convert_to_nx(mol_b)

    candidates = (
        anchored_dummy_groups
        for dummy_groups in generate_dummy_group_assignments(bond_graph_b, core_atoms_b)
        for anchored_dummy_groups in generate_anchored_dummy_group_assignments(
            dummy_groups, bond_graph_a, bond_graph_b, core_atoms_a, core_atoms_b
        )
    )

    # TODO: consider refining to use a heuristic rather than arbitrary selection
    # (e.g. maximize core-dummy bonds, maximize angle terms, minimize rotatable bonds, etc.)
    arbitrary_anchored_dummy_groups = next(candidates)

    for _, (angle_anchor, _) in arbitrary_anchored_dummy_groups.items():
        if angle_anchor is None:
            warnings.warn("Unable to find stable angle term in mol_a", CoreBondChangeWarning)

    return arbitrary_anchored_dummy_groups


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

    log_linear_fn = partial(interpolate.log_linear_interpolation, min_value=k_min)
    k = pad(log_linear_fn, src_k, dst_k, lamb, lambda_min, lambda_max)
    x = pad(interpolate.linear_interpolation, src_x, dst_x, lamb, lambda_min, lambda_max)

    return jnp.array([k, x])


def interpolate_chiral_volume_params(src_params, dst_params, lamb, k_min, lambda_min, lambda_max):
    src_k = src_params
    dst_k = dst_params

    log_linear_fn = partial(interpolate.log_linear_interpolation, min_value=k_min)
    k = pad(log_linear_fn, src_k, dst_k, lamb, lambda_min, lambda_max)

    return jnp.array(k)


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

    log_linear_fn = partial(interpolate.log_linear_interpolation, min_value=k_min)
    k = pad(log_linear_fn, src_k, dst_k, lamb, lambda_min, lambda_max)

    src_phase = src_phase
    dst_phase = src_phase + cyclic_difference(src_phase, dst_phase, period=2 * np.pi)
    phase = pad(interpolate.linear_interpolation, src_phase, dst_phase, lamb, lambda_min, lambda_max)

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

    k = pad(interpolate.linear_interpolation, src_k, dst_k, lamb, lambda_min, lambda_max)

    src_phase = src_phase
    dst_phase = src_phase + cyclic_difference(src_phase, dst_phase, period=2 * np.pi)
    phase = pad(interpolate.linear_interpolation, src_phase, dst_phase, lamb, lambda_min, lambda_max)

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

    def __init__(self, mol_a, mol_b, core: NDArray):
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
        self.c_to_a = {int(v): k for k, v in enumerate(self.a_to_c)}
        self.c_to_b = {int(v): k for k, v in enumerate(self.b_to_c)}

    def get_dummy_atoms_a(self) -> set[int]:
        return {idx for idx, flag in enumerate(self.c_flags) if flag == AtomMapFlags.MOL_A}

    def get_dummy_atoms_b(self) -> set[int]:
        return {idx for idx, flag in enumerate(self.c_flags) if flag == AtomMapFlags.MOL_B}

    def get_core_atoms(self) -> set[int]:
        return {idx for idx, flag in enumerate(self.c_flags) if flag == AtomMapFlags.CORE}

    def get_num_atoms(self) -> int:
        """
        Get the total number of atoms in the alchemical hybrid.

        Returns
        -------
        int
            Total number of atoms.
        """
        return self.mol_a.GetNumAtoms() + self.mol_b.GetNumAtoms() - len(self.core)

    def get_num_dummy_atoms(self) -> int:
        """
        Get the total number of dummy atoms in the alchemical hybrid.

        Returns
        -------
        int
            Total number of atoms.
        """
        return self.mol_a.GetNumAtoms() + self.mol_b.GetNumAtoms() - len(self.core) - len(self.core)


class MissingBondsInChiralVolumeException(Exception):
    pass


class TorsionsDefinedOverLinearAngleException(Exception):
    pass


def assert_default_system_constraints(system: GuestSystem | HostGuestSystem):
    # Assert that the system objects satisfy a set of constraints
    assert_bonds_defined_for_chiral_volumes(system)
    assert_torsions_defined_over_non_linear_angles(system)


def assert_bonds_defined_for_chiral_volumes(
    system: GuestSystem | HostGuestSystem, bond_k_min: float = DEFAULT_BOND_IS_PRESENT_K
):
    """
    Assert that bonds defined for every chiral volume is present and has a force constant greater than bond_k_min
    """
    bonds_present = set()

    for idxs, (bond_k, _) in zip(system.bond.potential.idxs, system.bond.params):
        if bond_k > bond_k_min:
            bonds_present.add(tuple(idxs))  # type: ignore[arg-type]

    for (c, i, j, k), chiral_k in zip(system.chiral_atom.potential.idxs, system.chiral_atom.params):
        if chiral_k > 0:
            if canonicalize_bond((c, i)) not in bonds_present:
                raise MissingBondsInChiralVolumeException(f"bond {(c, i)} missing from Chiral Volume {(c, i, j, k)}")
            if canonicalize_bond((c, j)) not in bonds_present:
                raise MissingBondsInChiralVolumeException(f"bond {(c, j)} missing from Chiral Volume {(c, i, j, k)}")
            if canonicalize_bond((c, k)) not in bonds_present:
                raise MissingBondsInChiralVolumeException(f"bond {(c, k)} missing from Chiral Volume {(c, i, j, k)}")


def assert_torsions_defined_over_non_linear_angles(system: GuestSystem | HostGuestSystem | HostSystem):
    """
    Assert that torsions are never defined over angle terms with an equilibrium value close to 180.
    """
    linear_angles = set()

    for idxs, angle_params in zip(system.angle.potential.idxs, system.angle.params):
        angle_k, angle_a0 = angle_params[0], angle_params[1]

        if angle_k > 0:
            if abs(angle_a0 - np.pi) < 0.174533:  # 10 degrees, arbitrary but conservative threshold
                linear_angles.add(tuple(idxs))

    for (i, j, k, l), (proper_k, _, _) in zip(system.proper.potential.idxs, system.proper.params):
        if proper_k > 0:
            if canonicalize_bond((i, j, k)) in linear_angles:
                raise TorsionsDefinedOverLinearAngleException(
                    f"angle {(i, j, k)} is linear in proper torsion {(i, j, k, l)}"
                )
            if canonicalize_bond((j, k, l)) in linear_angles:
                raise TorsionsDefinedOverLinearAngleException(
                    f"angle {(j, k, l)} is linear in proper torsion {(i, j, k, l)}"
                )

    for (i, j, k, l), (improper_k, _, _) in zip(system.improper.potential.idxs, system.improper.params):
        if improper_k > 0:
            if canonicalize_bond((i, j, k)) in linear_angles:
                raise TorsionsDefinedOverLinearAngleException(
                    f"angle {(i, j, k)} is linear in improper torsion {(i, j, k, l)}"
                )
            if canonicalize_bond((j, k, l)) in linear_angles:
                raise TorsionsDefinedOverLinearAngleException(
                    f"angle {(j, k, l)} is linear in improper torsion {(i, j, k, l)}"
                )


def assert_chiral_consistency(src_chiral_idxs: NDArray, dst_chiral_idxs: NDArray):
    """
    Assert that the chiral volumes are not inverting in src and dst.
    """
    src_chiral_restr_idx_set = ChiralRestrIdxSet(src_chiral_idxs)
    dst_chiral_restr_idx_set = ChiralRestrIdxSet(dst_chiral_idxs)

    # ensure that we don't have any chiral inversions between src and dst end states
    assert len(src_chiral_restr_idx_set.allowed_set.intersection(dst_chiral_restr_idx_set.disallowed_set)) == 0
    assert len(dst_chiral_restr_idx_set.allowed_set.intersection(src_chiral_restr_idx_set.disallowed_set)) == 0


class SingleTopology(AtomMapMixin):
    def __init__(self, mol_a: Chem.Mol, mol_b: Chem.Mol, core: NDArray, forcefield: Forcefield):
        """
        SingleTopology combines two molecules through a common core. The combined mol has
        atom indices laid out such that mol_a is identically mapped to the combined mol indices.
        The atoms unique to mol_b are then glued on to resulting molecule.

        Parameters
        ----------
        mol_a: Chem.Mol
            First guest

        mol_b: Chem.Mol
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

        self.anchored_dummy_groups_ab = find_dummy_groups_and_anchors(mol_a, mol_b, core[:, 0], core[:, 1])  # type: ignore[arg-type]
        self.anchored_dummy_groups_ba = find_dummy_groups_and_anchors(mol_b, mol_a, core[:, 1], core[:, 0])  # type: ignore[arg-type]

        # setup end states
        self.src_system = self._setup_end_state_src()
        self.dst_system = self._setup_end_state_dst()

        assert_chiral_consistency(
            self.src_system.chiral_atom.potential.idxs, self.dst_system.chiral_atom.potential.idxs
        )
        assert_default_system_constraints(self.src_system)
        assert_default_system_constraints(self.dst_system)

    def combine_masses(self, use_hmr: bool = False) -> list[float]:
        """
        Combine masses between two end-states by taking the heavier of the two core atoms.

        Parameters
        ----------

        use_hmr: bool
            Applies HMR to the masses. Refer to timemachine.fe.model_utils.apply_hmr for details.

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
            assert self.ff.hb_handle is not None
            _, mol_a_hb = mol_a_top.parameterize_harmonic_bond(self.ff.hb_handle.params)
            _, mol_b_hb = mol_b_top.parameterize_harmonic_bond(self.ff.hb_handle.params)

            mol_a_masses = model_utils.apply_hmr(mol_a_masses, mol_a_hb.idxs)
            mol_b_masses = model_utils.apply_hmr(mol_b_masses, mol_b_hb.idxs)

        mol_c_masses = []
        for c_idx in range(self.get_num_atoms()):
            flag = self.c_flags[c_idx]
            if flag == AtomMapFlags.CORE:
                mass_a = mol_a_masses[self.c_to_a[c_idx]]
                mass_b = mol_b_masses[self.c_to_b[c_idx]]
                mass = max(mass_a, mass_b)
            elif flag == AtomMapFlags.MOL_A:
                mass = mol_a_masses[self.c_to_a[c_idx]]
            elif flag == AtomMapFlags.MOL_B:
                mass = mol_b_masses[self.c_to_b[c_idx]]
            else:
                assert 0, f"Unknown atom flag: {flag}"

            mol_c_masses.append(mass)

        return mol_c_masses

    def combine_confs(self, x_a: NDArray, x_b: NDArray, lamb: float = 1.0) -> NDArray:
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
            otherwise use opposite order. Defaults to 1.0

        Returns
        -------
        np.array of shape (self.num_atoms,3)
            Combined conformation

        """
        if lamb < 0.5:
            return self.combine_confs_lhs(x_a, x_b)
        else:
            return self.combine_confs_rhs(x_a, x_b)

    def combine_confs_rhs(self, x_a: NDArray, x_b: NDArray) -> NDArray:
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

    def combine_confs_lhs(self, x_a: NDArray, x_b: NDArray) -> NDArray:
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
        Setup the source end-state. mol_a is fully interacting and mol_b's dummy atoms, attached
        in a factorizable way, are non-interacting.

        Returns
        -------
        GuestSystem
            Vacuum system
        """
        return setup_end_state(
            self.ff, self.mol_a, self.mol_b, self.core, self.a_to_c, self.b_to_c, self.anchored_dummy_groups_ab
        )

    def _setup_end_state_dst(self):
        """
        Setup the destination end-state. mol_b is fully interacting and mol_a's dummy atoms, attached
        in a factorizable way, are non-interacting.

        Returns
        -------
        GuestSystem
            Vacuum system
        """
        return setup_end_state(
            self.ff, self.mol_b, self.mol_a, self.core[:, ::-1], self.b_to_c, self.a_to_c, self.anchored_dummy_groups_ba
        )

    def align_and_interpolate_intramolecular_nonbonded(
        self, lamb: float
    ) -> BoundPotential[NonbondedPairListPrecomputed]:
        return self._setup_intermediate_nonbonded_term(
            self.src_system.nonbonded_pair_list,
            self.dst_system.nonbonded_pair_list,
            lamb,
            interpolate.align_nonbonded_idxs_and_params,
            interpolate.linear_interpolation,
        )

    @property
    def src_chiral_idxs(self):
        return set(tuple(x) for x in self.src_system.chiral_atom.potential.idxs)

    @property
    def dst_chiral_idxs(self):
        return set(tuple(x) for x in self.dst_system.chiral_atom.potential.idxs)

    def _setup_intermediate_nonbonded_term(
        self,
        src_nonbonded: BoundPotential[NonbondedPairListPrecomputed],
        dst_nonbonded: BoundPotential[NonbondedPairListPrecomputed],
        lamb: float,
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

            src_qlj, src_w = src_params[:, : NBParamIdx.W_IDX], src_params[:, NBParamIdx.W_IDX]
            dst_qlj, dst_w = dst_params[:, : NBParamIdx.W_IDX], dst_params[:, NBParamIdx.W_IDX]

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

        return NonbondedPairListPrecomputed(pair_idxs, src_nonbonded.potential.beta, cutoff).bind(pair_params)

    def _setup_intermediate_chiral_bond_term(
        self,
        src_bond: BoundPotential[ChiralBondRestraint],
        dst_bond: BoundPotential[ChiralBondRestraint],
        lamb: float,
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

    def all_idxs_belong_to_core(self, idxs):
        core_atoms = self.get_core_atoms()
        return all([x in core_atoms for x in idxs])

    def any_idxs_belong_to_dummy_a(self, idxs):
        dummy_atoms = self.get_dummy_atoms_a()
        return any([x in dummy_atoms for x in idxs])

    def any_idxs_belong_to_dummy_b(self, idxs):
        dummy_atoms = self.get_dummy_atoms_b()
        return any([x in dummy_atoms for x in idxs])

    def _chiral_volume_is_turning_on(self, idxs):
        return tuple(idxs) in self.dst_chiral_idxs and tuple(idxs) not in self.src_chiral_idxs

    def _chiral_volume_is_turning_off(self, idxs):
        return tuple(idxs) in self.src_chiral_idxs and tuple(idxs) not in self.dst_chiral_idxs

    def _bond_idxs_belong_to_chiral_volume_turning_on(self, idxs):
        induced_bond_idxs = set()
        for c, i, j, k in self.dst_chiral_idxs.difference(self.src_chiral_idxs):
            induced_bond_idxs.add(canonicalize_bond((c, i)))
            induced_bond_idxs.add(canonicalize_bond((c, j)))
            induced_bond_idxs.add(canonicalize_bond((c, k)))
        return idxs in induced_bond_idxs

    def _bond_idxs_belong_to_chiral_volume_turning_off(self, idxs):
        induced_bond_idxs = set()
        for c, i, j, k in self.src_chiral_idxs.difference(self.dst_chiral_idxs):
            induced_bond_idxs.add(canonicalize_bond((c, i)))
            induced_bond_idxs.add(canonicalize_bond((c, j)))
            induced_bond_idxs.add(canonicalize_bond((c, k)))
        return idxs in induced_bond_idxs

    def _angle_idxs_belong_to_chiral_volume_turning_on(self, idxs):
        induced_angle_idxs = set()
        for c, i, j, k in self.dst_chiral_idxs.difference(self.src_chiral_idxs):
            induced_angle_idxs.add(canonicalize_bond((i, c, j)))
            induced_angle_idxs.add(canonicalize_bond((i, c, k)))
            induced_angle_idxs.add(canonicalize_bond((j, c, k)))

        return idxs in induced_angle_idxs

    def _angle_idxs_belong_to_chiral_volume_turning_off(self, idxs):
        induced_angle_idxs = set()
        for c, i, j, k in self.src_chiral_idxs.difference(self.dst_chiral_idxs):
            induced_angle_idxs.add(canonicalize_bond((i, c, j)))
            induced_angle_idxs.add(canonicalize_bond((i, c, k)))
            induced_angle_idxs.add(canonicalize_bond((j, c, k)))

        return idxs in induced_angle_idxs

    def _interpolate_bond(self, idxs, src_params, dst_params, lamb: float):
        if self.all_idxs_belong_to_core(idxs):
            min_max = CORE_BOND_MIN_MAX
        elif self.any_idxs_belong_to_dummy_a(idxs):
            if self._bond_idxs_belong_to_chiral_volume_turning_on(idxs):
                assert 0
            elif self._bond_idxs_belong_to_chiral_volume_turning_off(idxs):
                min_max = DUMMY_A_CHIRAL_BOND_CONVERTING_OFF_MIN_MAX
            else:
                min_max = DUMMY_A_BOND_MIN_MAX
        elif self.any_idxs_belong_to_dummy_b(idxs):
            if self._bond_idxs_belong_to_chiral_volume_turning_on(idxs):
                min_max = DUMMY_B_CHIRAL_BOND_CONVERTING_ON_MIN_MAX
            elif self._bond_idxs_belong_to_chiral_volume_turning_off(idxs):
                assert 0
            else:
                min_max = DUMMY_B_BOND_MIN_MAX
        else:
            assert 0

        k_min = 0.1
        return interpolate_harmonic_bond_params(src_params, dst_params, lamb, k_min, *min_max)

    def _interpolate_angle(self, idxs, src_params, dst_params, lamb: float):
        if self.all_idxs_belong_to_core(idxs):
            if self._angle_idxs_belong_to_chiral_volume_turning_on(idxs):
                min_max = CORE_CHIRAL_ANGLE_CONVERTING_ON_MIN_MAX
            elif self._angle_idxs_belong_to_chiral_volume_turning_off(idxs):
                min_max = CORE_CHIRAL_ANGLE_CONVERTING_OFF_MIN_MAX
            else:
                min_max = CORE_ANGLE_MIN_MAX
        elif self.any_idxs_belong_to_dummy_a(idxs):
            if self._angle_idxs_belong_to_chiral_volume_turning_on(idxs):
                assert 0
            elif self._angle_idxs_belong_to_chiral_volume_turning_off(idxs):
                min_max = DUMMY_A_CHIRAL_ANGLE_CONVERTING_OFF_MIN_MAX
            else:
                min_max = DUMMY_A_ANGLE_MIN_MAX
        elif self.any_idxs_belong_to_dummy_b(idxs):
            if self._angle_idxs_belong_to_chiral_volume_turning_on(idxs):
                min_max = DUMMY_B_CHIRAL_ANGLE_CONVERTING_ON_MIN_MAX
            elif self._angle_idxs_belong_to_chiral_volume_turning_off(idxs):
                assert 0
            else:
                min_max = DUMMY_B_ANGLE_MIN_MAX
        else:
            assert 0

        k_min = 0.05
        return interpolate_harmonic_angle_params(src_params, dst_params, lamb, k_min, *min_max)

    def _interpolate_chiral_atom(self, idxs, src_k, dst_k, lamb: float):
        if self.all_idxs_belong_to_core(idxs):
            if self._chiral_volume_is_turning_on(idxs):
                min_max = CORE_CHIRAL_ATOM_CONVERTING_ON_MIN_MAX
            elif self._chiral_volume_is_turning_off(idxs):
                min_max = CORE_CHIRAL_ATOM_CONVERTING_OFF_MIN_MAX
            else:
                assert src_k == dst_k
                return src_k
        elif self.any_idxs_belong_to_dummy_a(idxs):
            if self._chiral_volume_is_turning_on(idxs):
                assert 0
            elif self._chiral_volume_is_turning_off(idxs):
                min_max = DUMMY_A_CHIRAL_ATOM_CONVERTING_OFF_MIN_MAX
            else:
                assert src_k == dst_k
                return src_k
        elif self.any_idxs_belong_to_dummy_b(idxs):
            if self._chiral_volume_is_turning_on(idxs):
                min_max = DUMMY_B_CHIRAL_ATOM_CONVERTING_ON_MIN_MAX
            elif self._chiral_volume_is_turning_off(idxs):
                assert 0
            else:
                assert src_k == dst_k
                return src_k
        else:
            assert 0

        k_min = 0.025
        k_final = interpolate_chiral_volume_params(src_k, dst_k, lamb, k_min, *min_max)
        return k_final

    def _interpolate_torsion(self, idxs, src_params, dst_params, lamb):
        if self.all_idxs_belong_to_core(idxs):
            if src_params[0] == 0:
                min_max = CORE_TORSION_OFF_TO_ON_MIN_MAX
            elif dst_params[0] == 0:
                min_max = CORE_TORSION_ON_TO_OFF_MIN_MAX
            else:
                min_max = CORE_TORSION_MIN_MAX
        elif self.any_idxs_belong_to_dummy_a(idxs):
            min_max = DUMMY_A_TORSION_MIN_MAX
        elif self.any_idxs_belong_to_dummy_b(idxs):
            min_max = DUMMY_B_TORSION_MIN_MAX
        else:
            assert 0

        return interpolate_periodic_torsion_params(src_params, dst_params, lamb, *min_max)

    def _align_and_interpolate_bonded_term(self, lamb, src_potential, dst_potential, align_fn, interpolate_fn):
        set_of_tuples = align_fn(
            src_potential.potential.idxs,
            src_potential.params,
            dst_potential.potential.idxs,
            dst_potential.params,
        )

        # (ytz): sigh, this is the garbage code that we need to write if we don't do re-shaping
        # at the potential level, sigh. If we have zero length arrays, for both src_potential and dst_potential,
        # then we lose information about the correct shapes.
        src_idxs_ndim = src_potential.potential.idxs.ndim
        dst_idxs_ndim = dst_potential.potential.idxs.ndim
        assert src_idxs_ndim == 2
        assert src_idxs_ndim == dst_idxs_ndim

        src_idxs_last_shape = src_potential.potential.idxs.shape[1]
        dst_idxs_last_shape = dst_potential.potential.idxs.shape[1]
        assert src_idxs_last_shape == dst_idxs_last_shape

        src_idxs_dtype = src_potential.potential.idxs.dtype
        dst_idxs_dtype = dst_potential.potential.idxs.dtype
        assert src_idxs_dtype == dst_idxs_dtype

        src_params_ndim = src_potential.params.ndim
        dst_params_ndim = dst_potential.params.ndim
        assert src_params_ndim == dst_params_ndim

        src_params_dtype = src_potential.params.dtype
        dst_params_dtype = dst_potential.params.dtype
        assert src_params_dtype == dst_params_dtype

        bonded_idxs = []
        bonded_params = []
        for idxs, src_params, dst_params in set_of_tuples:
            bonded_idxs.append(idxs)
            bonded_params.append(interpolate_fn(idxs, src_params, dst_params, lamb))

        bonded_idxs = np.array(bonded_idxs, dtype=src_idxs_dtype).reshape(-1, src_idxs_last_shape)

        if src_params_ndim == 1:
            # chiral atom
            bonded_params = jnp.array(bonded_params, dtype=src_params_dtype)
            pass
        elif src_params_ndim == 2:
            # everything else
            src_params_last_shape = src_potential.params.shape[1]
            bonded_params = jnp.array(bonded_params, dtype=src_params_dtype).reshape(-1, src_params_last_shape)
        else:
            assert 0

        assert bonded_idxs.ndim == src_idxs_ndim
        assert bonded_idxs.dtype == src_idxs_dtype
        assert bonded_params.ndim == src_params_ndim
        # this will downcast automatically if 64bit jax is disabled
        if jax.config.read("jax_enable_x64") is True:
            assert bonded_params.dtype == src_params_dtype

        # if we had automatic reshaping, then bond_class automatically reshape everything down below
        bond_class = type(src_potential.potential)
        return bond_class(bonded_idxs).bind(bonded_params)

    def align_and_interpolate_chiral_atoms(self, lamb: float):
        return self._align_and_interpolate_bonded_term(
            lamb,
            self.src_system.chiral_atom,
            self.dst_system.chiral_atom,
            interpolate.align_chiral_atom_idxs_and_params,
            self._interpolate_chiral_atom,
        )

    def align_and_interpolate_angles(self, lamb: float):
        return self._align_and_interpolate_bonded_term(
            lamb,
            self.src_system.angle,
            self.dst_system.angle,
            interpolate.align_harmonic_angle_idxs_and_params,
            self._interpolate_angle,
        )

    def align_and_interpolate_bonds(self, lamb: float):
        return self._align_and_interpolate_bonded_term(
            lamb,
            self.src_system.bond,
            self.dst_system.bond,
            interpolate.align_harmonic_bond_idxs_and_params,
            self._interpolate_bond,
        )

    # proper and impropers use the same logic for alignment
    def align_and_interpolate_propers(self, lamb: float):
        res = self._align_and_interpolate_bonded_term(
            lamb,
            self.src_system.proper,
            self.dst_system.proper,
            interpolate.align_proper_idxs_and_params,
            self._interpolate_torsion,
        )
        return res

    def align_and_interpolate_impropers(self, lamb: float):
        return self._align_and_interpolate_bonded_term(
            lamb,
            self.src_system.improper,
            self.dst_system.improper,
            interpolate.align_improper_idxs_and_params,
            self._interpolate_torsion,
        )

    def setup_intermediate_state(self, lamb: float) -> GuestSystem:
        r"""
        Set up intermediate states at some value of the alchemical parameter :math:`\lambda`.

        Parameters
        ----------
        lamb: float
            Alchemical parameter :math:`\lambda`.

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

        # branching diagram for the interpolation of bonded parameters
        #
        #                     bonded terms
        #                      /       \
        #                     /         \_____
        #                    /                \
        #                  core              dummy
        #                 /   \              |    \
        #          ______/     \             |     \_________________
        #         /             |            |                       \
        #        /              |            |                        \
        #   converting    non-converting     A                         B
        #    /      \                       / \                       / \
        #   /        \                     /   \                     /   \
        # on->off   off->on        converting non-converting converting  non-converting
        #                           (on->off)                (off->on)
        src_system = self.src_system
        dst_system = self.dst_system

        bond = self.align_and_interpolate_bonds(lamb)
        angle = self.align_and_interpolate_angles(lamb)
        chiral_atom = self.align_and_interpolate_chiral_atoms(lamb)
        proper = self.align_and_interpolate_propers(lamb)
        improper = self.align_and_interpolate_impropers(lamb)
        nonbonded = self.align_and_interpolate_intramolecular_nonbonded(lamb)

        assert src_system.chiral_bond
        assert dst_system.chiral_bond
        # (ytz): dead code, chiral bond not simulated in production
        chiral_bond = self._setup_intermediate_chiral_bond_term(
            src_system.chiral_bond,
            dst_system.chiral_bond,
            lamb,
            interpolate.linear_interpolation,
        )

        return GuestSystem(
            bond=bond,
            angle=angle,
            proper=proper,
            improper=improper,
            nonbonded_pair_list=nonbonded,
            chiral_atom=chiral_atom,
            chiral_bond=chiral_bond,
        )

    def mol(self, lamb: float, min_bond_k: float = DEFAULT_BOND_IS_PRESENT_K) -> Chem.Mol:
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
            flag = self.c_flags[c_idx]
            if flag == AtomMapFlags.CORE:
                # core, in both mol_a and mol_b
                if lamb < 0.5:
                    atomic_num = mol_a_atomic_nums[self.c_to_a[c_idx]]
                else:
                    atomic_num = mol_b_atomic_nums[self.c_to_b[c_idx]]
            elif flag == AtomMapFlags.MOL_A:
                # only in mol_a
                atomic_num = mol_a_atomic_nums[self.c_to_a[c_idx]]
            elif flag == AtomMapFlags.MOL_B:
                # only in mol_b
                atomic_num = mol_b_atomic_nums[self.c_to_b[c_idx]]
            else:
                # in neither, assert
                assert 0
            atom = Chem.Atom(atomic_num)
            mol.AddAtom(atom)

        # setup bonds
        for (i, j), (k, _) in zip(vs.bond.potential.idxs, vs.bond.params):
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
            if membership == AtomMapFlags.CORE:  # core atom
                a_idx = self.c_to_a[idx]
                b_idx = self.c_to_b[idx]

                # interpolate charges when in common-core
                q = (1 - lamb) * guest_a_q[a_idx] + lamb * guest_b_q[b_idx]
                sig = (1 - lamb) * guest_a_lj[a_idx, 0] + lamb * guest_b_lj[b_idx, 0]
                eps = (1 - lamb) * guest_a_lj[a_idx, 1] + lamb * guest_b_lj[b_idx, 1]

                # fixed at w = 0
                w = 0.0

            elif membership == AtomMapFlags.MOL_A:  # dummy_A
                a_idx = self.c_to_a[idx]
                q = guest_a_q[a_idx]
                sig = guest_a_lj[a_idx, 0]
                eps = guest_a_lj[a_idx, 1]

                # Decouple dummy group A as lambda goes from 0 to 1
                w = interpolate_w_coord(0.0, cutoff, lamb)

            elif membership == AtomMapFlags.MOL_B:  # dummy_B
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
        self,
        lamb: float,
        host_nonbonded: BoundPotential[Nonbonded],
        num_water_atoms: int,
        ff: Forcefield,
        omm_topology: OpenMMTopology,
    ) -> BoundPotential[NonbondedInteractionGroup]:
        """Parameterize nonbonded interactions between the host and guest"""
        num_host_atoms = host_nonbonded.params.shape[0]
        num_guest_atoms = self.get_num_atoms()
        cutoff = host_nonbonded.potential.cutoff

        guest_ixn_env_params = self._get_guest_params(self.ff.q_handle, self.ff.lj_handle, lamb, cutoff)

        # L-W terms
        num_other_atoms = num_host_atoms - num_water_atoms

        def get_lig_idxs() -> NDArray[np.int32]:
            return np.arange(num_guest_atoms, dtype=np.int32) + num_host_atoms

        def get_water_idxs() -> NDArray[np.int32]:
            return np.arange(num_water_atoms, dtype=np.int32) + num_other_atoms

        def get_other_idxs() -> NDArray[np.int32]:
            return np.arange(num_other_atoms, dtype=np.int32)

        def get_env_idxs() -> NDArray[np.int32]:
            return np.concatenate([get_other_idxs(), get_water_idxs()])

        hg_nb_ixn_params = host_nonbonded.params.copy()
        if ff.env_bcc_handle is not None:
            env_bcc_h = ff.env_bcc_handle.get_env_handle(omm_topology, ff)
            hg_nb_ixn_params[:, NBParamIdx.Q_IDX] = env_bcc_h.parameterize(ff.env_bcc_handle.params)

        ixn_pot, ixn_params = get_ligand_ixn_pots_params(
            get_lig_idxs(),
            get_env_idxs(),
            hg_nb_ixn_params,
            guest_ixn_env_params,
            beta=host_nonbonded.potential.beta,
            cutoff=cutoff,
        )

        bound_ixn_pot = ixn_pot.bind(ixn_params)
        return bound_ixn_pot

    def combine_with_host(
        self,
        host_system: HostSystem,
        lamb: float,
        num_water_atoms: int,
        ff: Forcefield,
        omm_topology: OpenMMTopology,
    ) -> HostGuestSystem:
        """
        Setup host guest system. Bonds, angles, torsions, chiral_atom, chiral_bond and nonbonded terms are
        combined. In particular:

        1) Bond, angle, torsion, chiral_atom, chiral_bond idxs are incremented by num_host_atoms.
        2) Host-host and host-guest interactions use a nonbonded potential, with exclusions set to the
            original host-host exclusions, in addition to *all* guest-guest interactions being excluded.
            Host-guest interactions are implemented as follows:
            i) at lambda = 0, the dummy atoms of mol_a are fully interacting, and become non-interacting at lambda = 1
            ii) at lambda = 0, the dummy atoms of mol_b are non-interacting, and become fully interacting at lambda = 1
            iii) the core atoms have parameters interpolated from mol_a's qlj to mol_b's qlj.
        3) guest-guest interactions use pre-computed nonbonded interactions implemented as a pairlist.

        Parameters
        ----------
        host_system: HostSystem
            Parameterized system of the host

        lamb: float
            Which lambda value we want to generate the combined system.

        num_water_atoms: int
            Number of water atoms as part of the host.

        ff:
            Forcefield object

        omm_topology:
            Openmm topology for the host.

        Returns
        -------
        HostGuestSystem
            dataclass representing all of the potentials of the host-guest system.
        """

        guest_system = self.setup_intermediate_state(lamb=lamb)
        num_host_atoms = host_system.nonbonded_all_pairs.params.shape[0]
        guest_chiral_atom_idxs = np.array(guest_system.chiral_atom.potential.idxs, dtype=np.int32) + num_host_atoms
        guest_system.chiral_atom.potential.idxs = guest_chiral_atom_idxs
        guest_chiral_bond_idxs = np.array(guest_system.chiral_bond.potential.idxs, dtype=np.int32) + num_host_atoms
        guest_system.chiral_bond.potential.idxs = guest_chiral_bond_idxs

        def prune_empty_arrays(arrays):
            return [arr for arr in arrays if arr.size > 0]

        guest_nonbonded_idxs = (
            np.array(guest_system.nonbonded_pair_list.potential.idxs, dtype=np.int32) + num_host_atoms
        )
        guest_system.nonbonded_pair_list.potential.idxs = guest_nonbonded_idxs

        combined_bond_idxs = np.concatenate(
            prune_empty_arrays([host_system.bond.potential.idxs, guest_system.bond.potential.idxs + num_host_atoms])
        )
        combined_bond_params = jnp.concatenate(prune_empty_arrays([host_system.bond.params, guest_system.bond.params]))
        combined_bond = HarmonicBond(combined_bond_idxs).bind(combined_bond_params)

        combined_angle_idxs = np.concatenate(
            [host_system.angle.potential.idxs, guest_system.angle.potential.idxs + num_host_atoms]
        )
        host_angle_params = jnp.hstack(
            prune_empty_arrays(
                [
                    host_system.angle.params,
                    np.zeros((host_system.angle.params.shape[0], 1)),  # stable angle epsilon = 0
                ]
            )
        )
        combined_angle_params = jnp.concatenate(prune_empty_arrays([host_angle_params, guest_system.angle.params]))
        combined_angle = HarmonicAngleStable(combined_angle_idxs).bind(combined_angle_params)

        # print(host_system.proper.potential.idxs.shape, guest_system.proper.potential.idxs.shape)
        combined_proper_idxs = np.concatenate(
            prune_empty_arrays(
                [host_system.proper.potential.idxs, guest_system.proper.potential.idxs + num_host_atoms]
            ),
        )

        # print(host_system.proper.params.shape, guest_system.proper.params.shape)
        combined_proper_params = jnp.concatenate(
            prune_empty_arrays([host_system.proper.params, guest_system.proper.params])
        )
        combined_proper = PeriodicTorsion(combined_proper_idxs).bind(combined_proper_params)

        combined_improper_idxs = np.concatenate(
            prune_empty_arrays(
                [host_system.improper.potential.idxs, guest_system.improper.potential.idxs + num_host_atoms]
            )
        )
        combined_improper_params = jnp.concatenate(
            prune_empty_arrays([host_system.improper.params, guest_system.improper.params])
        )
        combined_improper = PeriodicTorsion(combined_improper_idxs).bind(combined_improper_params)

        host_nonbonded_all_pairs = self._parameterize_host_nonbonded(host_system.nonbonded_all_pairs)
        host_guest_nonbonded_ixn_group = self._parameterize_host_guest_nonbonded_ixn(
            lamb,
            host_system.nonbonded_all_pairs,
            num_water_atoms,
            ff,
            omm_topology,
        )

        return HostGuestSystem(
            bond=combined_bond,
            angle=combined_angle,
            proper=combined_proper,
            improper=combined_improper,
            chiral_atom=guest_system.chiral_atom,
            chiral_bond=guest_system.chiral_bond,
            nonbonded_pair_list=guest_system.nonbonded_pair_list,
            nonbonded_all_pairs=host_nonbonded_all_pairs,
            nonbonded_ixn_group=host_guest_nonbonded_ixn_group,
        )
