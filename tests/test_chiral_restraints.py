import multiprocessing
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(multiprocessing.cpu_count())

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from rdkit import Chem
from rdkit.Chem import AllChem

from timemachine.fe import chiral_utils, topology, utils
from timemachine.ff import Forcefield
from timemachine.integrator import simulate
from timemachine.potentials.chiral_restraints import (
    U_chiral_atom,
    U_chiral_atom_batch,
    U_chiral_bond,
    U_chiral_bond_batch,
    pyramidal_volume,
    torsion_volume,
)


def minimize_scipy(x0, U_fn):
    shape = x0.shape

    def U_flat(x_flat):
        x_full = x_flat.reshape(*shape)
        return U_fn(x_full)

    grad_bfgs_fn = jax.grad(U_flat)
    res = scipy.optimize.minimize(U_flat, x0.reshape(-1), jac=grad_bfgs_fn)
    xi = res.x.reshape(*shape)
    return xi


def simulate_system(U_fn, x0, num_samples=20000):
    num_atoms = x0.shape[0]
    x_min = minimize_scipy(x0, U_fn)
    seed = 2023

    num_workers = multiprocessing.cpu_count()
    samples_per_worker = int(np.ceil(num_samples / num_workers))

    # batches_per_worker = num_samples // num_workers
    burn_in_batches = 2000
    frames, _ = simulate(
        x_min, U_fn, 300.0, np.ones(num_atoms) * 4.0, 500, samples_per_worker + burn_in_batches, num_workers, seed=seed
    )
    # (ytz): discard burn in batches
    frames = frames[:, burn_in_batches:, :, :]
    # collect over all workers
    frames = frames.reshape(-1, num_atoms, 3)[:num_samples]
    # sanity check that we didn't undersample
    assert len(frames) == num_samples
    return frames


def test_setup_chiral_atom_restraints():
    """On a methane conformer, assert that permuting coordinates or permuting restr_idxs
    both independently toggle the chiral restraint"""
    mol = Chem.MolFromMolBlock(
        """
  Mrv2202 06072215563D

  5  4  0  0  0  0            999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.3633   -0.5138    0.8900 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.0900    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.3633    1.0277    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.3633   -0.5138   -0.8900 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
  1  5  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    # needs to be batched in order for jax to play nicely
    x0 = utils.get_romol_conf(mol)
    normal_restr_idxs = chiral_utils.setup_chiral_atom_restraints(mol, x0, 0)

    x0_inverted = x0[[0, 2, 1, 3, 4]]  # swap two atoms
    inverted_restr_idxs = chiral_utils.setup_chiral_atom_restraints(mol, x0_inverted, 0)

    # check the sign of the resulting idxs
    k = 1000.0
    assert np.all(np.asarray(U_chiral_atom_batch(x0, normal_restr_idxs, k)) == 0)
    assert np.all(np.asarray(U_chiral_atom_batch(x0, inverted_restr_idxs, k)) > 0)
    assert np.all(np.asarray(U_chiral_atom_batch(x0_inverted, normal_restr_idxs, k)) > 0)
    assert np.all(np.asarray(U_chiral_atom_batch(x0_inverted, inverted_restr_idxs, k)) == 0)


def test_setup_chiral_bond_restraints():
    """On a 'Cl/C(F)=N/F' conformer, assert that flipping a dihedral angle or permuting restr_idxs
    both independently toggle the chiral bond restraint"""

    mol_cis = Chem.MolFromSmiles(r"Cl\C(F)=N/F")
    mol_trans = Chem.MolFromSmiles(r"Cl\C(F)=N\F")

    AllChem.EmbedMolecule(mol_cis)
    AllChem.EmbedMolecule(mol_trans)

    # needs to be batched in order for jax to play nicely
    x0_cis = utils.get_romol_conf(mol_cis)
    x0_trans = utils.get_romol_conf(mol_trans)
    src_atom = 1
    dst_atom = 3
    normal_restr_idxs, signs = chiral_utils.setup_chiral_bond_restraints(mol_cis, x0_cis, src_atom, dst_atom)

    inverted_restr_idxs, inverted_signs = chiral_utils.setup_chiral_bond_restraints(
        mol_trans, x0_trans, src_atom, dst_atom
    )
    k = 1000.0

    assert np.all(np.asarray(U_chiral_bond_batch(x0_cis, normal_restr_idxs, k, signs)) == 0)
    assert np.all(np.asarray(U_chiral_bond_batch(x0_cis, inverted_restr_idxs, k, inverted_signs)) > 0)
    assert np.all(np.asarray(U_chiral_bond_batch(x0_trans, normal_restr_idxs, k, signs)) > 0)
    assert np.all(np.asarray(U_chiral_bond_batch(x0_trans, inverted_restr_idxs, k, inverted_signs)) == 0)


def test_chiral_restraints_pyramidal():
    """For ammonium, assert that:
    * without chiral restraints, up and down states are ~ equally sampled
    * with chiral restraints, ~ only the specified state is sampled"""
    mol = Chem.MolFromMolBlock(
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

    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    s_top = topology.BaseTopology(mol, ff)
    x0 = utils.get_romol_conf(mol)

    # check initial chirality
    restr_idxs = np.array([0, 1, 2, 3])
    vol = pyramidal_volume(*x0[restr_idxs])
    assert vol > 0.5
    system = s_top.setup_end_state()
    U_fn = system.get_U_fn()
    vols_orig = []
    frames = simulate_system(U_fn, x0)
    for f in frames:
        vol = pyramidal_volume(*f[restr_idxs])
        vols_orig.append(vol)

    def U_total(x):
        return U_fn(x) + U_chiral_atom(x, restr_idxs, 1000.0)

    vols_chiral = []
    frames = simulate_system(U_total, x0)
    for f in frames:
        vol = pyramidal_volume(*f[restr_idxs])
        vols_chiral.append(vol)

    vols_orig = np.array(vols_orig)
    vols_chiral = np.array(vols_chiral)

    # should be within 5% of 50/50 for the original distribution, the <0 case
    # is implied
    assert np.abs(np.mean(vols_orig < 0) - 0.5) < 0.05
    assert np.abs(np.mean(vols_chiral < 0) - 0.95) < 0.05

    ref_dist = [x for x in vols_orig if x < 0]
    test_dist = [x for x in vols_chiral if x < 0]

    # should be indistinguishable under KS-test
    ks, pv = scipy.stats.ks_2samp(ref_dist, test_dist)
    assert ks < 0.05
    assert pv > 0.10

    # # useful plotting diagnostics, do not remove.
    # # compare original distributions
    # plt.hist(vols_orig, bins=np.linspace(-1, 1, 80), alpha=0.5, label="no_chiral_restr", density=True)
    # plt.hist(vols_chiral, bins=np.linspace(-1, 1, 80), alpha=0.5, label="with_chiral_restr", density=True)
    # plt.legend()
    # plt.title("pyramidal chirality")
    # plt.xlabel("chiral volume")
    # plt.ylabel("samples")
    # plt.title("raw")
    # plt.show()

    # # time series
    # plt.plot(vols_chiral, label="with_chiral_restraint")
    # plt.show()

    # # compare filtered distributions
    # plt.hist([x for x in vols_orig if x < 0], bins=np.linspace(-1, 1, 80), alpha=0.5, label="no_chiral_restr", density=True)
    # plt.hist([x for x in vols_chiral if x < 0], bins=np.linspace(-1, 1, 80), alpha=0.5, label="with_chiral_restr", density=True)
    # plt.legend()
    # plt.title("pyramidal chirality")
    # plt.xlabel("chiral volume")
    # plt.ylabel("samples")
    # plt.title("normalized")
    # plt.show()


class BaseTopologyRescaledCharges(topology.BaseTopology):
    def __init__(self, scale, *args, **kwargs):
        self.scale = scale
        super().__init__(*args, **kwargs)

    def parameterize_nonbonded(self, ff_q_params, ff_lj_params):
        params, nb = topology.BaseTopology.parameterize_nonbonded(self, ff_q_params, ff_lj_params)
        charge_indices = jnp.index_exp[:, 0]
        new_params = jnp.asarray(params).at[charge_indices].multiply(self.scale)
        return new_params, nb

    def parameterize_nonbonded_pairlist(self, ff_q_params, ff_lj_params):
        params, nb = topology.BaseTopology.parameterize_nonbonded_pairlist(self, ff_q_params, ff_lj_params)
        charge_indices = jnp.index_exp[:, 0]
        new_params = jnp.asarray(params).at[charge_indices].multiply(self.scale)
        return new_params, nb


def test_chiral_restraints_torsion():
    """For a charge-scaled version of hydrogen peroxide, assert that:
    * without chiral bond restraints, cis/trans states are sampled ~ 30%/70%
    * with chiral bond restraints, ~ only specified state is sampled"""
    mol = Chem.MolFromMolBlock(
        """
  Mrv2202 06062222163D

  4  3  0  0  0  0            999 V2000
   -0.1757   -0.7570    0.3351 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0031   -0.0645   -0.3351 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.3069    0.0645   -0.3351 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.4857    0.7570    0.3351 H   0  0  0  0  0  0  0  0  0  0  0  0
  2  3  1  0  0  0  0
  1  2  1  0  0  0  0
  3  4  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    scale = 0.4
    s_top = BaseTopologyRescaledCharges(scale, mol, ff)
    x0 = utils.get_romol_conf(mol)

    # check initial chirality
    torsion_idxs = np.array([0, 1, 2, 3])
    vol = torsion_volume(*x0[torsion_idxs])

    assert abs(vol - 0.0) < 0.05
    system = s_top.setup_end_state()
    U_fn = system.get_U_fn()

    vols_orig = []
    frames = simulate_system(U_fn, x0)
    for f in frames:
        vols_orig.append(torsion_volume(*f[torsion_idxs]))
    vols_orig = np.array(vols_orig)

    # the 30/70 ratio is dependent on the scale defined above, which affects the repulsive
    # strength of the hydrogens.
    assert np.abs(np.mean(vols_orig > 0) - 0.3) < 0.05

    all_signs = [1, -1]  # [trans, cis]
    for sign in all_signs:

        def U_total(x):
            return U_fn(x) + U_chiral_bond(x, torsion_idxs, 1000.0, sign)

        vols_chiral = []
        frames = simulate_system(U_total, x0)
        for f in frames:
            vols_chiral.append(torsion_volume(*f[torsion_idxs]))
        vols_chiral = np.array(vols_chiral)

        # # useful for plotting
        # import matplotlib.pyplot as plt
        # plt.hist(vols_orig, bins=np.linspace(-1, 1, 80), alpha=0.5, label="no_chiral_restr", density=True)
        # plt.hist(vols_chiral, bins=np.linspace(-1, 1, 80), alpha=0.5, label="with_chiral_restr", density=True)
        # plt.legend()
        # plt.title("torsion chirality")
        # plt.xlabel("chiral volume")
        # plt.ylabel("samples")
        # plt.title("raw")
        # plt.show()

        # plt.hist([x for x in vols_orig if sign*x < 0], bins=np.linspace(-1, 1, 80), alpha=0.5, label="no_chiral_restr", density=True)
        # plt.hist([x for x in vols_chiral if sign*x < 0], bins=np.linspace(-1, 1, 80), alpha=0.5, label="with_chiral_restr", density=True)
        # plt.legend()
        # plt.title("torsion chirality")
        # plt.xlabel("chiral volume")
        # plt.ylabel("samples")
        # plt.title("normalized")
        # plt.show()

        # we should predominantly sample the correct chiral state
        # (95% of the samples)
        if sign == 1:
            assert np.abs(np.mean(vols_chiral < 0) - 0.95) < 0.05
        else:
            assert np.abs(np.mean(vols_chiral > 0) - 0.95) < 0.05

        ref_dist = [x for x in vols_orig if sign * x < 0]
        test_dist = [x for x in vols_chiral if sign * x < 0]
        # should be indistinguishable under KS-test
        ks, pv = scipy.stats.ks_2samp(ref_dist, test_dist)
        assert ks < 0.05
        assert pv > 0.10


def test_chiral_restraints_tetrahedral():
    # test that we can restrain the chirality of a tetrahedral molecule
    # to its inverted state
    mol = Chem.MolFromMolBlock(
        """
  Mrv2202 06072204223D

  5  4  0  0  0  0            999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.3633   -0.5138    0.8900 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.0900    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.3633    1.0277    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.3633   -0.5138   -0.8900 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
  1  5  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    s_top = topology.BaseTopology(mol, ff)
    x0 = utils.get_romol_conf(mol)

    perms = [
        [0, 1, 2, 3],
        [0, 1, 4, 2],
        [0, 1, 3, 4],
        [0, 2, 4, 3],
    ]

    perms = np.array(perms)

    # check initial chirality, all positive values initially
    for p in perms:
        vol = pyramidal_volume(*x0[p])
        assert vol > 0

    system = s_top.setup_end_state()
    U_fn = system.get_U_fn()

    vols_orig = []
    frames = simulate_system(U_fn, x0)
    for f in frames:
        vols_orig.append([pyramidal_volume(*f[p]) for p in perms])

    def U_total(x):
        return U_fn(x) + jnp.sum(U_chiral_atom_batch(x, perms, 1000.0))

    vols_chiral = []
    frames = simulate_system(U_total, x0)
    for f in frames:
        vols_chiral.append([pyramidal_volume(*f[p]) for p in perms])

    vols_orig = np.array(vols_orig).reshape(-1)
    vols_chiral = np.array(vols_chiral).reshape(-1)

    # ref_dist samples predominantly positive chiral volumes, and test_dist
    # pre-dominantly samples the negative chiral volumes, and the distribution
    # is symmetric about vol=0.
    ref_dist = np.array([x for x in vols_orig if x > 0])
    test_dist = np.array([x for x in vols_chiral if x < 0])
    # should be indistinguishable under KS-test
    ks, pv = scipy.stats.ks_2samp(-ref_dist, test_dist)
    assert ks < 0.05
    assert pv > 0.10

    # debugging plots
    # plt.hist(vols_orig, bins=np.linspace(-1, 1, 80), alpha=0.5, label="no_chiral_restr", density=True)
    # plt.hist(vols_chiral, bins=np.linspace(-1, 1, 80), alpha=0.5, label="with_chiral_restr", density=True)
    # plt.legend()
    # plt.title("tetrahedral chirality")
    # plt.xlabel("chiral volume")
    # plt.ylabel("samples")
    # plt.title("raw")
    # plt.show()


def test_chiral_spiro_cyclopentane():
    # fused spiro cyclopentane
    mol = Chem.MolFromMolBlock(
        """
  Mrv2202 06082219093D

 23 24  0  0  0  0            999 V2000
    0.0153    0.8682   -1.0264 C   0  0  2  0  0  0  0  0  0  0  0  0
   -1.1793    1.2423   -0.1275 C   0  0  1  0  0  0  0  0  0  0  0  0
   -1.0368    0.3687    1.1395 C   0  0  2  0  0  0  0  0  0  0  0  0
    0.3652   -0.2781    1.0734 C   0  0  2  0  0  0  0  0  0  0  0  0
    1.1394    0.4765   -0.0405 C   0  0  2  0  0  0  0  0  0  0  0  0
    1.7303    1.6686    0.5168 O   0  0  0  0  0  0  0  0  0  0  0  0
    3.1536    1.5004    0.5804 C   0  0  2  0  0  0  0  0  0  0  0  0
    3.4838    0.0421    0.2203 C   0  0  2  0  0  0  0  0  0  0  0  0
    2.3003   -0.3326   -0.6729 C   0  0  2  0  0  0  0  0  0  0  0  0
    0.3073    1.7040   -1.6701 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.2513    0.0142   -1.6581 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.1233    2.3026    0.1405 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.1297    1.0640   -0.6358 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.8062   -0.4074    1.1604 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.1399    0.9807    2.0397 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.2589   -1.3347    0.8092 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.8797   -0.2178    2.0368 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.5126    1.7577    1.5808 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.6133    2.1778   -0.1451 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.4983   -0.5783    1.1212 H   0  0  0  0  0  0  0  0  0  0  0  0
    4.4407   -0.0475   -0.3004 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.4929    0.0029   -1.6968 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.1250   -1.4110   -0.6825 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  5  1  0  0  0  0
  2  3  1  0  0  0  0
  3  4  1  0  0  0  0
  4  5  1  0  0  0  0
  7  8  1  0  0  0  0
  5  9  1  0  0  0  0
  9  8  1  0  0  0  0
  5  6  1  0  0  0  0
  6  7  1  0  0  0  0
  1 10  1  0  0  0  0
  1 11  1  0  0  0  0
  2 12  1  0  0  0  0
  2 13  1  0  0  0  0
  3 14  1  0  0  0  0
  3 15  1  0  0  0  0
  4 16  1  0  0  0  0
  4 17  1  0  0  0  0
  7 18  1  0  0  0  0
  7 19  1  0  0  0  0
  8 20  1  0  0  0  0
  8 21  1  0  0  0  0
  9 22  1  0  0  0  0
  9 23  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    x0 = utils.get_romol_conf(mol)
    # chiral center is a_idx == 4
    normal_restr_idxs = np.array([[4, 0, 3, 8], [4, 3, 0, 5], [4, 0, 8, 5], [4, 8, 3, 5]])

    inverted_restr_idxs = np.array([[4, 3, 0, 8], [4, 0, 3, 5], [4, 8, 0, 5], [4, 3, 8, 5]])

    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    s_top = topology.BaseTopology(mol, ff)

    kc = 1000.0

    assert np.all(np.asarray(U_chiral_atom_batch(x0, normal_restr_idxs, kc)) == 0)
    assert np.all(np.asarray(U_chiral_atom_batch(x0, inverted_restr_idxs, kc)) > 0)

    system = s_top.setup_end_state()
    U_fn = system.get_U_fn()

    vols_orig = []
    frames = simulate_system(U_fn, x0)
    for f in frames:
        vol_list = []
        for p in normal_restr_idxs:
            vol_list.append(pyramidal_volume(*f[p]))
        vols_orig.append(vol_list)

    def U_total(x):
        nrgs = [U_fn(x)]
        # use inverted restr_idxs
        for p in inverted_restr_idxs:
            nrgs.append(U_chiral_atom(x, p, kc))
        return jnp.sum(jnp.array(nrgs))

    vols_chiral = []
    frames = simulate_system(U_total, x0)
    for f in frames:
        vol_list = []
        for p in normal_restr_idxs:
            vol_list.append(pyramidal_volume(*f[p]))
        vols_chiral.append(vol_list)

    vols_orig = np.array(vols_orig).reshape(-1)
    vols_chiral = np.array(vols_chiral).reshape(-1)

    # debugging plots
    # import matplotlib.pyplot as plt
    # plt.hist(vols_orig, bins=np.linspace(-1, 1, 80), alpha=0.5, label="no_chiral_restr", density=True)
    # plt.hist(vols_chiral, bins=np.linspace(-1, 1, 80), alpha=0.5, label="with_chiral_restr", density=True)
    # plt.legend()
    # plt.title("tetrahedral chirality")
    # plt.xlabel("chiral volume")
    # plt.ylabel("samples")
    # plt.title("raw")
    # plt.show()

    # this should generate a symmetric distribution
    ref_dist = np.array([x for x in vols_orig if x < 0])
    test_dist = np.array([x for x in vols_chiral if x > 0])
    # should be indistinguishable under KS-test
    ks, pv = scipy.stats.ks_2samp(-ref_dist, test_dist)
    assert ks < 0.05
    assert pv > 0.10
