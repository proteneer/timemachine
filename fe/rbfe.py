# (ytz): utility functions for an RBFE calculation.

# in an RBFE calculation, we need to modify the recipe in two stages:
# stage 0 is linear mixing:
# (1-lambda)*com_restraint + (lambda)*core_restraint
# plane_idxs of non-interacting ligand is set to 1.

# stage 1 is non-bonded insertion
# core restraints are always on.
# ligand B is decoupled from the environment

import numpy as np

from timemachine.lib import potentials

def add_nonbonded_exclusions(recipe, a_idxs, b_idxs):

    extra_exclusions = []
    extra_scale_factors = []
    for i in a_idxs:
        for j in b_idxs:
            extra_exclusions.append((i, j))
            extra_scale_factors.append((1.0, 1.0)) # vdw/q

    extra_exclusions = np.array(extra_exclusions, dtype=np.int32)
    extra_scale_factors = np.array(extra_scale_factors, dtype=np.float64)

    for bp in recipe.bound_potentials:
        if isinstance(bp, potentials.Nonbonded):

            bp.set_exclusion_idxs(np.concatenate([bp.get_exclusion_idxs(), extra_exclusions]))
            bp.set_scale_factors(np.concatenate([bp.get_scale_factors(), extra_scale_factors]))


def set_nonbonded_lambda_idxs(recipe, atom_idxs, plane, offset):
    """
    Set the nonbonded lambda parameters in a recipe

    Parameters
    ----------
    atom_idxs: np.array int
        Which idxs we want to modify

    """

    for bp in recipe.bound_potentials:
        if isinstance(bp, potentials.Nonbonded):

            plane_idxs = bp.get_lambda_plane_idxs()
            plane_idxs[atom_idxs] = plane

            offset_idxs = bp.get_lambda_offset_idxs()
            offset_idxs[atom_idxs] = offset

def create_centroid_restraints(core_pairs, com_k, masses):
    """
    Create a centroid restraint between core atoms

    Parameters
    ----------
    core_pairs: np.array
        Tuple of atom pairs we wish to restrain.

    core_k: float
        Force constant of the restraints

    masses: float
        Atomic masses

    """
    a_idxs = []
    b_idxs = []
    for i,j in core_pairs:
        assert i not in a_idxs
        assert j not in b_idxs
        a_idxs.append(i)
        b_idxs.append(j)

    a_idxs = np.array(a_idxs, dtype=np.int32)
    b_idxs = np.array(b_idxs, dtype=np.int32)
    masses = np.array(masses, dtype=np.float64)

    return potentials.CentroidRestraint(a_idxs, b_idxs, masses, com_k, 0.0).bind(np.array([]))


def create_core_restraints(core_pairs, core_k):
    """
    Create a set of harmonic restraints on core atoms:

    Parameters
    ----------
    core_pairs: np.array
        Tuple of atom pairs we wish to restrain.

    core_k: float
        Force constant of the restraints

    """
    bond_idxs = []
    bond_params = []
    for i, j in core_pairs:
        # identity check
        assert i != j
        # symmetry check
        assert (i, j) not in bond_idxs
        assert (j, i) not in bond_idxs
        bond_idxs.append((i, j))
        bond_params.append((core_k, 0))

    bond_idxs = np.array(bond_idxs, dtype=np.int32)
    bond_params = np.array(bond_params, dtype=np.float64)

    return potentials.CoreRestraint(bond_idxs).bind(bond_params)

def stage_0(recipe, b_idxs, core_pairs, centroid_k, core_k):
    """
    Modify a recipe to allow for restraint conversion. This PR will add an alchemical
    potential that interpolates between centroid and core restraints, as well as modify
    the nonbonded terms to make the system non-interacting.

    Parameters
    ----------

    b_idxs: np.array
        Atoms in b_idxs will have their lambda plane idxs set to 1.

    core_pairs: np.array
        Tuple of atom pairs we wish to restrain.

    centroid_k: float
        Force constant for center of mass restraints

    core_k: float
        Force constant for core restraints

    """
    N = len(recipe.masses)

    core_restraints = create_core_restraints(core_pairs, core_k)
    centroid_restraints = create_centroid_restraints(core_pairs, centroid_k, recipe.masses)

    lhs = potentials.LambdaPotential(core_restraints, N, len(core_restraints.params), 1.0, 0.0)
    rhs = potentials.LambdaPotential(centroid_restraints, N, len(centroid_restraints.params), -1.0, 1.0)

    recipe.bound_potentials.append(lhs)
    recipe.bound_potentials.append(rhs)

    set_nonbonded_lambda_idxs(recipe, b_idxs, 1, 0)

def stage_1(recipe, a_idxs, b_idxs, core_pairs, core_k):
    """
    Modify a recipe for stage 1 decoupling. A vanilla core restraint is added. The nonbonded
    plane idxs are all zero, and offset idxs are modified for b_idxs. In addition, a fully dense
    set of exclusions between a_idxs and b_idxs are added .

    Parameters
    ----------
    a_idxs: np.array
        Indices of first mol.

    b_idxs: np.array
        Indices of second mol. Atoms in b_idxs will have their lambda offset idxs set to 1.

    core_pairs: np.array
        Tuple of atom pairs we wish to restrain.

    centroid_k: float
        Force constant for center of mass restraints

    core_k: float
        Force constant for core restraints

    """

    N = len(recipe.masses)

    core_restraints = create_core_restraints(core_pairs, core_k)

    recipe.bound_potentials.append(core_restraints)

    add_nonbonded_exclusions(recipe, a_idxs, b_idxs)
    set_nonbonded_lambda_idxs(recipe, b_idxs, 0, 1)
