import functools

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from rdkit import Chem

from timemachine.fe import geometry, utils
from timemachine.fe.single_topology import SingleTopologyV2
from timemachine.ff import Forcefield
from timemachine.integrator import simulate
from timemachine.potentials import bonded, nonbonded

# test that we do not invert stereochemical barriers at the end-states for various susceptible transformations.
# most of these tests proceed by measuring the chiral volume defined by 4 atoms and ensuring that they're consistent
# at both end-states.


def minimize_scipy(x0, U_fn):
    N = x0.shape[0]

    def U_bfgs(x_flat):
        x_full = x_flat.reshape(N, 3)
        return U_fn(x_full)

    grad_bfgs_fn = jax.grad(U_bfgs)
    res = scipy.optimize.minimize(U_bfgs, x0.reshape(-1), jac=grad_bfgs_fn)
    xi = res.x.reshape(N, 3)
    return xi


def simulate_idxs_and_params(idxs_and_params, x0):

    (
        (bond_idxs, bond_params),
        (angle_idxs, angle_params),
        (proper_idxs, proper_params),
        (improper_idxs, improper_params),
        (nbpl_idxs, nbpl_rescale_mask, nbpl_beta, nbpl_cutoff, nbpl_params),
        (x_angle_idxs, x_angle_params),
        (c_angle_idxs, c_angle_params),
    ) = idxs_and_params

    box = None
    bond_U = functools.partial(
        bonded.harmonic_bond, params=np.array(bond_params), box=box, lamb=0.0, bond_idxs=np.array(bond_idxs)
    )
    angle_U = functools.partial(
        bonded.harmonic_angle, params=np.array(angle_params), box=box, lamb=0.0, angle_idxs=np.array(angle_idxs)
    )
    proper_U = functools.partial(
        bonded.periodic_torsion, params=np.array(proper_params), box=box, lamb=0.0, torsion_idxs=np.array(proper_idxs)
    )
    improper_U = functools.partial(
        bonded.periodic_torsion,
        params=np.array(improper_params),
        box=box,
        lamb=0.0,
        torsion_idxs=np.array(improper_idxs),
    )

    nbpl_U = functools.partial(
        nonbonded.nonbonded_v3_on_specific_pairs,
        pairs=np.array(nbpl_idxs),
        params=np.array(nbpl_params),
        box=box,
        beta=nbpl_beta,
        cutoff=nbpl_cutoff,
        rescale_mask=np.array(nbpl_rescale_mask),
    )
    c_angle_U = functools.partial(
        bonded.harmonic_c_angle, params=np.array(c_angle_params), box=box, lamb=0.0, angle_idxs=c_angle_idxs
    )
    x_angle_U = functools.partial(
        bonded.harmonic_x_angle, params=np.array(x_angle_params), box=box, lamb=0.0, angle_idxs=x_angle_idxs
    )

    def U_fn(x):
        vdw, q = nbpl_U(x)
        return (
            bond_U(x)
            + angle_U(x)
            + proper_U(x)
            + improper_U(x)
            + c_angle_U(x)
            + x_angle_U(x)
            + jnp.sum(vdw)
            + jnp.sum(q)
        )

    num_atoms = x0.shape[0]
    x_min = minimize_scipy(x0, U_fn)

    num_workers = 1
    num_batches = 2000
    frames, _ = simulate(x_min, U_fn, 300.0, np.ones(num_atoms) * 4.0, 1000, num_batches, num_workers)
    # (ytz): discard burn in later
    # burn_in_batches = num_batches//10
    burn_in_batches = 0
    frames = frames[:, burn_in_batches:, :, :]
    # collect over all workers
    frames = frames.reshape(-1, num_atoms, 3)

    return frames


def test_adj_list_conversion():
    mol = Chem.MolFromSmiles("CC1CC1C(C)(C)C")
    bond_idxs = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
    nblist = geometry.bond_idxs_to_adj_list(mol.GetNumAtoms(), bond_idxs)

    expected = [[1], [0, 2, 3], [1, 3], [1, 2, 4], [3, 5, 6, 7], [4], [4], [4]]

    np.testing.assert_array_equal(nblist, expected)


def measure_chiral_volume(x0, x1, x2, x3):
    """
    Compute the normalized chiral volume given four points.

    Parameters
    ----------
    np.array: (3,)
        Center point

    np.array: (3,)
        First point

    np.array: (3,)
        Second point

    np.array: (3,)
        Third point

    Returns
    -------
    float
        A number between -1.0<x<1.0 denoting the normalized chirality

    """
    # compute vectors
    v0 = x1 - x0
    v1 = x2 - x0
    v2 = x3 - x0

    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    return np.dot(np.cross(v0, v1), v2)


def test_stereo_water_to_tetrahedral():
    mol_a = Chem.MolFromMolBlock(
        """
  Mrv2202 05192218593D

  3  2  0  0  0  0            999 V2000
    0.5038    0.9940    0.3645 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.4342    0.1988    0.9333 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.5015    0.5419   -0.8126 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  3  1  0  0  0  0
  1  2  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    mol_b = Chem.MolFromMolBlock(
        """
  Mrv2202 05192219003D

  5  4  0  0  0  0            999 V2000
    0.5038    0.9940    0.3645 C   0  0  1  0  0  0  0  0  0  0  0  0
    0.4342    0.1988    0.9333 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.5015    0.5419   -0.8126 F   0  0  0  0  0  0  0  0  0  0  0  0
    1.6483    2.0245    0.3645 Cl  0  0  0  0  0  0  0  0  0  0  0  0
   -0.5266    2.1385    0.3645 Br  0  0  0  0  0  0  0  0  0  0  0  0
  1  3  1  0  0  0  0
  1  2  1  0  0  0  0
  1  4  1  0  0  0  0
  1  5  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    core = np.array([[0, 0], [1, 1], [2, 2]])
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    s_top = SingleTopologyV2(mol_a, mol_b, core, ff)

    x_a = utils.get_romol_conf(mol_a)
    x_b = utils.get_romol_conf(mol_b)
    x0 = s_top.combine_confs(x_a, x_b)

    vol_cl = measure_chiral_volume(x0[0], x0[1], x0[2], x0[3])
    vol_br = measure_chiral_volume(x0[0], x0[1], x0[2], x0[4])

    assert vol_cl > 0 and vol_br < 0

    idxs_and_params = s_top.generate_end_state_mol_a()
    frames = simulate_idxs_and_params(idxs_and_params, x0)

    for f in frames:
        vol_cl = measure_chiral_volume(f[0], f[1], f[2], f[3])
        vol_br = measure_chiral_volume(f[0], f[1], f[2], f[4])
        assert vol_cl > 0 and vol_br < 0

    idxs_and_params = s_top.generate_end_state_mol_b()
    frames = simulate_idxs_and_params(idxs_and_params, x0)

    for f in frames:
        vol_cl = measure_chiral_volume(f[0], f[1], f[2], f[3])
        vol_br = measure_chiral_volume(f[0], f[1], f[2], f[4])
        assert vol_cl > 0 and vol_br < 0


def test_halomethyl_to_halomethylamine():
    # test that we preserve stereochemistry when morphing tetrahedral
    # geometries
    mol_a = Chem.MolFromMolBlock(
        """
  Mrv2202 05192216353D

  5  4  0  0  0  0            999 V2000
    0.3495    0.4000   -0.7530 C   0  0  2  0  0  0  0  0  0  0  0  0
   -0.5582   -0.1718    0.8478 Br  0  0  0  0  0  0  0  0  0  0  0  0
   -0.8566    1.4388   -1.8390 Br  0  0  0  0  0  0  0  0  0  0  0  0
    0.9382   -1.1442   -1.7440 Br  0  0  0  0  0  0  0  0  0  0  0  0
    2.0273    1.5851   -0.2289 I   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
  1  5  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    mol_b = Chem.MolFromMolBlock(
        """
  Mrv2202 05192216363D

  7  6  0  0  0  0            999 V2000
    0.3495    0.4000   -0.7530 C   0  0  2  0  0  0  0  0  0  0  0  0
   -0.5582   -0.1718    0.8478 Br  0  0  0  0  0  0  0  0  0  0  0  0
   -0.8566    1.4388   -1.8390 Br  0  0  0  0  0  0  0  0  0  0  0  0
    0.9382   -1.1442   -1.7440 Br  0  0  0  0  0  0  0  0  0  0  0  0
    2.0273    1.5851   -0.2289 N   0  0  0  0  0  0  0  0  0  0  0  0
    2.2242    3.2477   -0.2289 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.1718    0.5547   -0.2289 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
  1  5  1  0  0  0  0
  5  6  1  0  0  0  0
  5  7  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    s_top = SingleTopologyV2(mol_a, mol_b, core, ff)

    x_a = utils.get_romol_conf(mol_a)
    x_b = utils.get_romol_conf(mol_b)
    x0 = s_top.combine_confs(x_a, x_b)

    # check initial chirality
    vol_a = measure_chiral_volume(x0[0], x0[1], x0[2], x0[4])
    vol_d = measure_chiral_volume(x0[0], x0[1], x0[2], x0[5])
    assert vol_a < 0 and vol_d < 0

    idxs_and_params = s_top.generate_end_state_mol_a()
    frames = simulate_idxs_and_params(idxs_and_params, x0)

    for f in frames:
        vol_a = measure_chiral_volume(f[0], f[1], f[2], f[4])
        vol_d = measure_chiral_volume(f[0], f[1], f[2], f[5])
        assert vol_a < 0 and vol_d < 0

    idxs_and_params = s_top.generate_end_state_mol_b()
    frames = simulate_idxs_and_params(idxs_and_params, x0)

    for f in frames:
        vol_a = measure_chiral_volume(f[0], f[1], f[2], f[4])
        vol_d = measure_chiral_volume(f[0], f[1], f[2], f[5])
        assert vol_a < 0 and vol_d < 0


def test_halomethyl_to_halomethylamine_inverted():
    # test that we preserve stereochemistry when morphing from SP3->SP3, except
    # the nitrogen is assigned an alternative chirality

    mol_a = Chem.MolFromMolBlock(
        """
  Mrv2202 05192216353D

  5  4  0  0  0  0            999 V2000
    0.3495    0.4000   -0.7530 C   0  0  2  0  0  0  0  0  0  0  0  0
   -0.5582   -0.1718    0.8478 Br  0  0  0  0  0  0  0  0  0  0  0  0
   -0.8566    1.4388   -1.8390 Br  0  0  0  0  0  0  0  0  0  0  0  0
    0.9382   -1.1442   -1.7440 Br  0  0  0  0  0  0  0  0  0  0  0  0
    2.0273    1.5851   -0.2289 I   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
  1  5  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    mol_b = Chem.MolFromMolBlock(
        """
  Mrv2202 05192216593D

  7  6  0  0  0  0            999 V2000
   -0.0814    0.0208   -1.3024 C   0  0  1  0  0  0  0  0  0  0  0  0
   -0.0096    0.1615    0.6181 Br  0  0  0  0  0  0  0  0  0  0  0  0
   -1.6626    0.9097   -1.9529 Br  0  0  0  0  0  0  0  0  0  0  0  0
   -0.1350   -1.8376   -1.8089 Br  0  0  0  0  0  0  0  0  0  0  0  0
   -1.1201   -0.6482    0.3122 N   0  0  1  0  0  0  0  0  0  0  0  0
   -2.2975   -0.5188    1.0529 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.3037   -1.9164    0.9197 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
  1  5  1  0  0  0  0
  5  6  1  0  0  0  0
  5  7  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    s_top = SingleTopologyV2(mol_a, mol_b, core, ff)

    x_a = utils.get_romol_conf(mol_a)
    x_b = utils.get_romol_conf(mol_b)
    x0 = s_top.combine_confs(x_a, x_b)

    # check initial chirality
    vol_a = measure_chiral_volume(x0[0], x0[1], x0[2], x0[4])
    vol_d = measure_chiral_volume(x0[0], x0[1], x0[2], x0[5])
    assert vol_a < 0 and vol_d > 0

    idxs_and_params = s_top.generate_end_state_mol_a()
    frames = simulate_idxs_and_params(idxs_and_params, x0)

    for f in frames:
        vol_a = measure_chiral_volume(f[0], f[1], f[2], f[4])
        vol_d = measure_chiral_volume(f[0], f[1], f[2], f[5])
        assert vol_a < 0 and vol_d > 0

    idxs_and_params = s_top.generate_end_state_mol_b()
    frames = simulate_idxs_and_params(idxs_and_params, x0)

    for f in frames:
        vol_a = measure_chiral_volume(f[0], f[1], f[2], f[4])
        vol_d = measure_chiral_volume(f[0], f[1], f[2], f[5])
        assert vol_a < 0 and vol_d > 0


def test_ammonium_to_chloromethyl():
    # NH3 easily interconverts between the two chiral states. In the event that we
    # morph NH3 to something that is actually chiral, we should still be able to
    # ensure enantiopurity of the end-states.
    # we expect the a-state to be:
    #   mixed stereo on vol_a, fixed stereo on vol_b
    # we expect the b-state to be:
    #   fixed stereo on vol_a, fixed stereo on vol_b

    mol_a = Chem.MolFromMolBlock(
        """
  Mrv2202 05192218063D

  4  3  0  0  0  0            999 V2000
   -0.0541    0.5427   -0.3433 N   0  0  0  0  0  0  0  0  0  0  0  0
    0.4368    0.0213    0.3859 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.9636    0.0925   -0.4646 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.4652    0.3942   -1.2109 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    mol_b = Chem.MolFromMolBlock(
        """
  Mrv2202 05192218063D

  5  4  0  0  0  0            999 V2000
   -0.0541    0.5427   -0.3433 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.4368    0.0213    0.3859 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.9636    0.0925   -0.4646 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.4652    0.3942   -1.2109 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0541    2.0827   -0.3433 Cl  0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
  1  5  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    # first test: 4 core mapping
    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    s_top = SingleTopologyV2(mol_a, mol_b, core, ff)

    x_a = utils.get_romol_conf(mol_a)
    x_b = utils.get_romol_conf(mol_b)
    x0 = s_top.combine_confs(x_a, x_b)

    # check initial chirality
    vol_a = measure_chiral_volume(x0[0], x0[1], x0[2], x0[3])
    vol_d = measure_chiral_volume(x0[0], x0[1], x0[2], x0[4])
    assert vol_a > 0 and vol_d < 0

    # G3_PYRAMIDAL -> G4_TETRAHEDRAL
    idxs_and_params = s_top.generate_end_state_mol_a()
    frames = simulate_idxs_and_params(idxs_and_params, x0)

    num_vol_a_pos = 0
    num_vol_a_neg = 0

    for f in frames:
        vol_a = measure_chiral_volume(f[0], f[1], f[2], f[3])
        vol_d = measure_chiral_volume(f[0], f[1], f[2], f[4])
        if vol_a < 0:
            num_vol_a_neg += 1
        else:
            num_vol_a_pos += 1
        assert vol_d < 0

    # should be within 5% of 50/50
    assert abs(num_vol_a_pos / len(frames) - 0.5) < 0.05

    # G3_PYRAMIDAL -> G3_PYRAMIDAL (no dummy atoms)
    idxs_and_params = s_top.generate_end_state_mol_b()
    frames = simulate_idxs_and_params(idxs_and_params, x0)

    for f in frames:
        vol_a = measure_chiral_volume(f[0], f[1], f[2], f[3])
        vol_d = measure_chiral_volume(f[0], f[1], f[2], f[4])
        assert vol_a > 0 and vol_d < 0

    # # second test: 3 core mapping
    core = np.array([[0, 0], [1, 1], [2, 2]])
    s_top = SingleTopologyV2(mol_a, mol_b, core, ff)

    x_a = utils.get_romol_conf(mol_a)
    x_b = utils.get_romol_conf(mol_b)
    x0 = s_top.combine_confs(x_a, x_b)

    # check initial chirality
    vol_a = measure_chiral_volume(x0[0], x0[1], x0[2], x0[3])
    vol_d = measure_chiral_volume(x0[0], x0[1], x0[2], x0[5])
    assert vol_a > 0 and vol_d < 0

    # G2_KINK -> G4_TETRAHEDRAL
    idxs_and_params = s_top.generate_end_state_mol_a()
    frames = simulate_idxs_and_params(idxs_and_params, x0)

    num_vol_a_pos = 0
    num_vol_a_neg = 0

    for f in frames:
        vol_a = measure_chiral_volume(f[0], f[1], f[2], f[3])
        vol_d = measure_chiral_volume(f[0], f[1], f[2], f[5])
        if vol_a < 0:
            num_vol_a_neg += 1
        else:
            num_vol_a_pos += 1
        assert vol_d < 0

    # should be within 5% of 50/50
    assert abs(num_vol_a_pos / len(frames) - 0.5) < 0.05

    # G2_KINK -> G3_PYRAMIDAL
    idxs_and_params = s_top.generate_end_state_mol_b()
    frames = simulate_idxs_and_params(idxs_and_params, x0)

    # in this case, the nitrogen is forced to be planar via a
    # centroid restraint, so we will observed a mixture
    num_vol_a_pos = 0
    num_vol_a_neg = 0
    for f in frames:

        vol_a = measure_chiral_volume(f[0], f[1], f[2], f[3])
        vol_d = measure_chiral_volume(f[0], f[1], f[2], f[5])
        if vol_a < 0:
            num_vol_a_neg += 1
        else:
            num_vol_a_pos += 1
        assert vol_d < 0

    assert abs(num_vol_a_pos / len(frames) - 0.5) < 0.05
