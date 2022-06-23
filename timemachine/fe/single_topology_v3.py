import warnings
from collections.abc import Iterable

import numpy as np

from timemachine.fe import system, topology, utils
from timemachine.fe.dummy import canonicalize_bond, identify_dummy_groups, identify_root_anchors
from timemachine.lib import potentials


class MultipleAnchorWarning(UserWarning):
    pass


class CoreBondChangeWarning(UserWarning):
    pass


class MissingAngleError(RuntimeError):
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


def setup_dummy_interactions_from_ff(ff, mol, dummy_group, root_anchor_atom, nbr_anchor_atom):
    """
    Setup interactions involving atoms in a given dummy group.
    """
    top = topology.BaseTopology(mol, ff)

    bond_params, hb = top.parameterize_harmonic_bond(ff.hb_handle.params)
    angle_params, ha = top.parameterize_harmonic_angle(ff.ha_handle.params)
    improper_params, it = top.parameterize_improper_torsion(ff.it_handle.params)

    bond_idxs = hb.get_idxs()
    angle_idxs = ha.get_idxs()
    improper_idxs = it.get_idxs()

    return setup_dummy_interactions(
        bond_idxs,
        bond_params,
        angle_idxs,
        angle_params,
        improper_params,
        improper_idxs,
        dummy_group,
        root_anchor_atom,
        nbr_anchor_atom,
    )


def setup_dummy_interactions(
    bond_idxs,
    bond_params,
    angle_idxs,
    angle_params,
    improper_idxs,
    improper_params,
    dummy_group,
    root_anchor_atom,
    nbr_anchor_atom,
):
    """
    Setup interactions involving atoms in a given dummy group. The following rules are applied:

    1) We only allow for interactions within a dummy group, never between different dummy groups.
    2) We form the augmented_dummy_group = dummy_group + [root anchor], and:
        i) we only allow bond, angle, improper torsions terms to be turned on within a dummy group.
        ii) we disable all nonbonded and proper torsions involving atoms in dummy groups
    3) We can form a secondary augmented group using nbr_anchor_atom, but we only allow interactions
        involving angles [i,j,k] where i in dummy_group, j == root_anchor_atom, and k == nbr_anchor_atom

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

    dummy_group: set or list of int
        Atoms to be decoupled

    root_anchor_atom: int
        A core atom we want to anchor our dummy_group to

    nbr_anchor_atom: int
        Another core atom connected to root_anchor_atom to build an angle restraint off of.

    Returns
    -------
    (bonded_idxs, bonded_params)
        Returns bonds, angles, and improper idxs and parameters.
    """

    dummy_bond_idxs = []
    dummy_bond_params = []
    dummy_angle_idxs = []
    dummy_angle_params = []
    dummy_improper_idxs = []
    dummy_improper_params = []

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

    # (ytz): copy interactions that involve nbr_anchor_atom, if not None
    # this may be set to None
    if nbr_anchor_atom is not None:
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

    return (dummy_bond_idxs, dummy_angle_idxs, dummy_improper_idxs), (
        dummy_bond_params,
        dummy_angle_params,
        dummy_improper_params,
    )


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

    all_dummy_bond_idxs, all_dummy_bond_params = [], []
    all_dummy_angle_idxs, all_dummy_angle_params = [], []
    all_dummy_improper_idxs, all_dummy_improper_params = [], []

    dgs, jks = find_dummy_groups_and_anchors(mol_a, mol_b, core[:, 0], core[:, 1])
    # gotta add 'em all!
    for dg, (anchor, nbr) in zip(dgs, jks):

        all_idxs, all_params = setup_dummy_interactions_from_ff(ff, mol_b, dg, anchor, nbr)
        all_dummy_bond_idxs.extend(all_idxs[0])
        all_dummy_angle_idxs.extend(all_idxs[1])
        all_dummy_improper_idxs.extend(all_idxs[2])
        all_dummy_bond_params.extend(all_params[0])
        all_dummy_angle_params.extend(all_params[1])
        all_dummy_improper_params.extend(all_params[2])

    # generate parameters for mol_a
    mol_a_top = topology.BaseTopology(mol_a, ff)
    mol_a_bond_params, mol_a_hb = mol_a_top.parameterize_harmonic_bond(ff.hb_handle.params)
    mol_a_angle_params, mol_a_ha = mol_a_top.parameterize_harmonic_angle(ff.ha_handle.params)
    mol_a_proper_params, mol_a_pt = mol_a_top.parameterize_proper_torsion(ff.pt_handle.params)
    mol_a_improper_params, mol_a_it = mol_a_top.parameterize_improper_torsion(ff.it_handle.params)
    mol_a_nbpl_params, mol_a_nbpl = mol_a_top.parameterize_nonbonded_pairlist(ff.q_handle.params, ff.lj_handle.params)
    mol_a_chiral_atom, mol_a_chiral_bond = mol_a_top.setup_chiral_restraints()

    mol_a_bond_params = mol_a_bond_params.tolist()
    mol_a_angle_params = mol_a_angle_params.tolist()
    mol_a_proper_params = mol_a_proper_params.tolist()
    mol_a_improper_params = mol_a_improper_params.tolist()
    mol_a_nbpl_params = mol_a_nbpl_params.tolist()

    mol_a_bond_idxs = recursive_map(mol_a_hb.get_idxs(), a_to_c)
    mol_a_angle_idxs = recursive_map(mol_a_ha.get_idxs(), a_to_c)
    mol_a_proper_idxs = recursive_map(mol_a_pt.get_idxs(), a_to_c)
    mol_a_improper_idxs = recursive_map(mol_a_it.get_idxs(), a_to_c)
    mol_a_nbpl_idxs = recursive_map(mol_a_nbpl.get_idxs(), a_to_c)
    mol_a_chiral_atom_idxs = recursive_map(mol_a_chiral_atom.get_idxs(), a_to_c)
    mol_a_chiral_bond_idxs = recursive_map(mol_a_chiral_bond.get_idxs(), a_to_c)

    all_dummy_bond_idxs = recursive_map(all_dummy_bond_idxs, b_to_c)
    all_dummy_angle_idxs = recursive_map(all_dummy_angle_idxs, b_to_c)
    all_dummy_improper_idxs = recursive_map(all_dummy_improper_idxs, b_to_c)

    # parameterize the combined molecule
    mol_c_bond_idxs = mol_a_bond_idxs + all_dummy_bond_idxs
    mol_c_bond_params = mol_a_bond_params + all_dummy_bond_params

    mol_c_angle_idxs = mol_a_angle_idxs + all_dummy_angle_idxs
    mol_c_angle_params = mol_a_angle_params + all_dummy_angle_params

    mol_c_proper_idxs = mol_a_proper_idxs
    mol_c_proper_params = mol_a_proper_params

    mol_c_improper_idxs = mol_a_improper_idxs + all_dummy_improper_idxs
    mol_c_improper_params = mol_a_improper_params + all_dummy_improper_params

    # combine proper + improper
    mol_c_torsion_idxs = mol_c_proper_idxs + mol_c_improper_idxs
    mol_c_torsion_params = mol_c_proper_params + mol_c_improper_params

    # canonicalize bonds
    mol_c_bond_idxs_canon = [canonicalize_bond(idxs) for idxs in mol_c_bond_idxs]
    bond_potential = potentials.HarmonicBond(mol_c_bond_idxs_canon).bind(mol_c_bond_params)

    mol_c_angle_idxs_canon = [canonicalize_bond(idxs) for idxs in mol_c_angle_idxs]
    angle_potential = potentials.HarmonicAngle(mol_c_angle_idxs_canon).bind(mol_c_angle_params)

    mol_c_torsion_idxs_canon = [canonicalize_bond(idxs) for idxs in mol_c_torsion_idxs]
    torsion_potential = potentials.PeriodicTorsion(mol_c_torsion_idxs_canon).bind(mol_c_torsion_params)

    # dummy atoms do not have any nonbonded interactions, so we simply turn them off

    mol_c_nbpl_idxs_canon = [canonicalize_bond(idxs) for idxs in mol_a_nbpl_idxs]
    mol_a_nbpl.set_idxs(mol_c_nbpl_idxs_canon)
    nonbonded_potential = mol_a_nbpl.bind(mol_a_nbpl_params)

    # chiral atoms need special code for canonicalization, since triple product is invariant
    # under rotational symmetry (but not something like swap symmetry)
    canon_chiral_atom_idxs = []
    for i, j, k, l in mol_a_chiral_atom_idxs:
        rotations = [(j, k, l), (l, j, k), (k, l, j)]
        jj, kk, ll = min(rotations)
        canon_chiral_atom_idxs.append((i, jj, kk, ll))

    chiral_atom_idxs = np.array(canon_chiral_atom_idxs, dtype=np.int32).reshape((-1, 4))
    mol_c_chiral_bond_idxs_canon = [canonicalize_bond(idxs) for idxs in mol_a_chiral_bond_idxs]
    chiral_bond_idxs = np.array(mol_c_chiral_bond_idxs_canon, dtype=np.int32).reshape((-1, 4))
    chiral_bond_signs = np.array(mol_a_chiral_bond.get_signs())

    chiral_atom_potential = potentials.ChiralAtomRestraint(chiral_atom_idxs).bind(mol_a_chiral_atom.params)
    chiral_bond_potential = potentials.ChiralBondRestraint(chiral_bond_idxs, chiral_bond_signs).bind(
        mol_a_chiral_bond.params
    )

    return system.VacuumSystem(
        bond_potential,
        angle_potential,
        torsion_potential,
        nonbonded_potential,
        chiral_atom_potential,
        chiral_bond_potential,
    )


def find_dummy_groups_and_anchors(mol_a, mol_b, core_a, core_b):
    """
    Find dummy groups for mol_b and appropriate bond/angle atoms in mol_a to
    build restraints off of.

    Parameters
    ----------
    mol_a: Chem.Mol
        Fully interacting molecule

    mol_b: Chem.Mol
        Molecule that will provide dummy atoms

    core_a: list of int of length C
        Core atoms in mol_a, unique atoms only

    core_b: list of int of length C
        Core atoms in mol_b, unique atoms only

    Returns
    -------
    list of 2-tuples
        Returns a set of dummy_groups, and a list of 2-tuples (j,k). j is the root_anchor atom,
        and k is a neighboring core atom. Note that `k` may be None if no suitable neighbor can be found.

    """
    assert len(core_a) == len(core_b)
    assert len(set(core_a)) == len(core_a)
    assert len(set(core_b)) == len(core_b)

    bond_idxs_b = utils.get_romol_bonds(mol_b)
    dummy_groups_b = identify_dummy_groups(bond_idxs_b, core_b)

    core_b_to_a = dict()
    for ca, cb in zip(core_a, core_b):
        core_b_to_a[cb] = ca

    bond_idxs_a = utils.get_romol_bonds(mol_a)

    all_jks = []

    for dg in dummy_groups_b:
        dummy_atom = list(dg)[0]
        root_anchors = identify_root_anchors(bond_idxs_b, core_b, dummy_atom)

        if len(root_anchors) == 0:
            assert 0, "Disconnected dummy group"

        if len(root_anchors) > 1:
            # Consider the following situation:
            # D0.D1
            # .  .  where (.) is the dummy bond
            # C0-C1
            #
            # One of (C0.D0), (D0.D1), (C1.D1) dummy bonds needs to be broken in order to maintain factorizability.
            # This is a little arbitrary, but some choices are probably more efficient than others:
            #
            # hard     easy
            # D0.D1    D0 D1
            #    .     .  .
            # C0-C1    C0-C1
            #
            # the lhs is more difficult because it has significantly more phase space that can be
            # sampled than the fused case, but it's not super obvious how to best detect this.
            # So instead, we will pick a random anchor atom. One possible solution later on is to
            # minimize the number of rotatable bonds?
            warnings.warn(
                f"Multiple root anchors {root_anchors} found for dummy group: {dg}, picking a random anchor.",
                MultipleAnchorWarning,
            )

        # (i,j,k) where i is a dummy, j is anchor, and k is the angle anchor
        angle_jk = None

        # find an arbitrary but stable angle core atom that is one bond away from the root atom that is
        # present in mol_a
        for ra in root_anchors:
            core_b_nbs = [a.GetIdx() for a in mol_b.GetAtomWithIdx(ra).GetNeighbors() if a.GetIdx() in core_b]
            for nb in core_b_nbs:
                # see if this core_bond is present in mol_a
                # first, look up the idxs of the atoms in mol_a:
                ra_a, nb_a = core_b_to_a[ra], core_b_to_a[nb]
                if (ra_a, nb_a) in bond_idxs_a or (nb_a, ra_a) in bond_idxs_a:
                    angle_jk = ra, nb
                    break

        if angle_jk is None:
            warnings.warn("Unable to find stable angle term in mol_a", CoreBondChangeWarning)
            # pick an arbitrary root_anchor
            angle_jk = root_anchors[0], None

        # revert me
        # angle_jk = root_anchors[0], None

        all_jks.append(angle_jk)

    return dummy_groups_b, all_jks


class SingleTopologyV3:
    def __init__(self, mol_a, mol_b, core, forcefield):
        """
        SingleTopology combines two molecules through a common core. The combined mol has
        atom indices laid out such that mol_a is identically mapped to the combined mol indices.
        The atoms in the mol_b's R-group is then glued on to resulting molecule.

        Parameters
        ----------
        mol_a: ROMol
            First ligand

        mol_b: ROMol
            Second ligand

        core: np.array (C, 2)
            Atom mapping from mol_a to mol_b.

        forcefield: ff.Forcefield
            Forcefield to be used for parameterization.

        """
        assert mol_a is not None
        assert mol_b is not None
        self.mol_a = mol_a
        self.mol_b = mol_b
        self.core = core
        self.ff = forcefield

        assert core.shape[1] == 2

        # map into idxs in the combined molecule
        self.a_to_c = np.arange(mol_a.GetNumAtoms(), dtype=np.int32)  # identity
        self.b_to_c = np.zeros(mol_b.GetNumAtoms(), dtype=np.int32) - 1

        # test for uniqueness in core idxs for each mol
        assert len(set(tuple(core[:, 0]))) == len(core[:, 0])
        assert len(set(tuple(core[:, 1]))) == len(core[:, 1])

        for a, b in core:
            self.b_to_c[b] = a

        iota = self.mol_a.GetNumAtoms()
        for b_idx, c_idx in enumerate(self.b_to_c):
            if c_idx == -1:
                self.b_to_c[b_idx] = iota
                iota += 1

    def get_num_atoms(self):
        """
        Get the total number of atoms in the alchemical hybrid.

        Returns
        -------
        int
            Total number of atoms.
        """
        return self.mol_a.GetNumAtoms() + self.mol_b.GetNumAtoms() - len(self.core)

    def combine_confs(self, x_a, x_b):
        """
        Combine conformations of two molecules

        Parameters
        ----------
        x_a: np.array of shape (N_A,3)
            First conformation

        x_b: np.array of shape (N_B,3)
            Second conformation

        Returns
        -------
        np.array of shape (self.num_atoms,3)
            Combined conformation

        """
        # tbd: allow input to take lambda schedule and return an array for each value of lambda
        assert x_a.shape == (self.mol_a.GetNumAtoms(), 3)
        assert x_b.shape == (self.mol_b.GetNumAtoms(), 3)
        x0 = np.zeros((self.get_num_atoms(), 3))
        for src, dst in enumerate(self.a_to_c):
            x0[dst] = x_a[src]
        for src, dst in enumerate(self.b_to_c):
            x0[dst] = x_b[src]
        return x0

    def setup_end_state_src(self):
        """
        Setup the source end-state, where mol_a is fully interacting, with mol_b's dummy atoms attached
        in a factorizable way.

        Returns
        -------
        VacuumSystem
            Gas-phase system
        """
        return setup_end_state(self.ff, self.mol_a, self.mol_b, self.core, self.a_to_c, self.b_to_c)

    def setup_end_state_dst(self):
        """
        Setup the source end-state, where mol_a is fully interacting, with mol_b's dummy atoms attached
        in a factorizable way.

        Returns
        -------
        VacuumSystem
            Gas-phase system
        """
        new_core = np.stack([self.core[:, 1], self.core[:, 0]], axis=1)
        return setup_end_state(self.ff, self.mol_b, self.mol_a, new_core, self.b_to_c, self.a_to_c)

    def get_U_fn(self, lamb):
        """
        Get a jax compatible energy function parameterized at some value of lambda, using linear
        energy interpolation.

        Parameters
        ----------
        lamb: float
            0 <= lamb <= 1

        Returns
        -------
        Callable:
            An energy function f: R^(NCx3) -> R^1

        """
        U0_fn = self.setup_end_state_src().get_U_fn()
        U1_fn = self.setup_end_state_dst().get_U_fn()

        # revisit more efficient methods later
        def U_fn(x):
            return (1 - lamb) * U0_fn(x) + lamb * U1_fn(x)

        return U_fn
