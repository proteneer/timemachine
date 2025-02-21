from dataclasses import replace

import numpy as np
import pytest
import scipy
from jax import numpy as jnp
from rdkit import Chem

from timemachine.constants import (
    DEFAULT_ATOM_MAPPING_KWARGS,
    DEFAULT_CHIRAL_ATOM_RESTRAINT_K,
    DEFAULT_CHIRAL_BOND_RESTRAINT_K,
)
from timemachine.fe import topology, utils
from timemachine.fe.atom_mapping import get_cores
from timemachine.fe.chiral_utils import (
    make_chiral_flip_heatmaps,
    make_chiral_restr_fxns,
    setup_all_chiral_atom_restr_idxs,
)
from timemachine.fe.free_energy import HREXParams
from timemachine.fe.rbfe import DEFAULT_HREX_PARAMS, run_solvent, run_vacuum
from timemachine.fe.single_topology import AtomMapMixin
from timemachine.fe.system import simulate_system
from timemachine.ff import Forcefield
from timemachine.potentials import chiral_restraints
from timemachine.potentials.chiral_restraints import (
    U_chiral_atom,
    U_chiral_atom_batch,
    U_chiral_bond,
    pyramidal_volume,
    torsion_volume,
)
from timemachine.utils import path_to_internal_file


@pytest.mark.nocuda
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

    # should be indistinguishable under KS-test, i.e. should not be able to reject
    ks, pv = scipy.stats.ks_2samp(ref_dist, test_dist)
    assert ks < 0.05 or pv > 0.10

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

    def parameterize_nonbonded(
        self,
        ff_q_params,
        ff_q_params_intra,
        ff_lj_params,
        ff_lj_params_intra,
        intramol_params=True,
    ):
        params, nb = topology.BaseTopology.parameterize_nonbonded(
            self,
            ff_q_params,
            ff_q_params_intra,
            ff_lj_params,
            ff_lj_params_intra,
            intramol_params=intramol_params,
        )
        charge_indices = jnp.index_exp[:, 0]
        new_params = jnp.asarray(params).at[charge_indices].multiply(self.scale)
        return new_params, nb

    def parameterize_nonbonded_pairlist(
        self, ff_q_params, ff_q_params_intra, ff_lj_params, ff_lj_params_intra, intramol_params=True
    ):
        params, nb = topology.BaseTopology.parameterize_nonbonded_pairlist(
            self, ff_q_params, ff_q_params_intra, ff_lj_params, ff_lj_params_intra, intramol_params=intramol_params
        )
        charge_indices = jnp.index_exp[:, 0]
        new_params = jnp.asarray(params).at[charge_indices].multiply(self.scale)
        return new_params, nb


@pytest.mark.nocuda
def test_chiral_restraints_torsion():
    """For a charge-scaled version of hydrogen peroxide, assert that:
    * without chiral bond restraints, cis/trans states are sampled ~ 25%/75%
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

    # the 25/75 ratio is dependent on the scale defined above, which affects the repulsive
    # strength of the hydrogens.)
    assert np.abs(np.mean(vols_orig > 0) - 0.25) < 0.05

    all_signs = [1, -1]  # [trans, cis]
    for sign in all_signs:

        def U_total(x):
            return U_fn(x) + U_chiral_bond(x, torsion_idxs, 1000.0, sign)

        vols_chiral = []
        frames = simulate_system(U_total, x0)
        for f in frames:
            vols_chiral.append(torsion_volume(*f[torsion_idxs]))
        vols_chiral = np.array(vols_chiral)

        # useful for plotting
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
        assert ks < 0.05 or pv > 0.10


@pytest.mark.nocuda
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
    # pessimize: strengthen harmonic angle force constant to max
    ff.ha_handle.params[:, 0] = np.max(ff.ha_handle.params[:, 0])

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
    # turn off minimizing to avoid accidentally swapping chiral states
    frames = simulate_system(U_fn, x0, minimize=False)
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

    # ref_dist samples predominantly positive chiral volumes, and test_dist
    # pre-dominantly samples the negative chiral volumes, and the distribution
    # is symmetric about vol=0.
    ref_dist = np.array([x for x in vols_orig if x > 0])
    test_dist = np.array([x for x in vols_chiral if x < 0])
    # should be indistinguishable under KS-test
    ks, pv = scipy.stats.ks_2samp(-ref_dist, test_dist)
    assert ks < 0.05 or pv > 0.10


@pytest.mark.nocuda
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

    # ugh basin-hopping is inverting the non-chiral endstates too, sigh
    system = s_top.setup_end_state()
    U_fn = system.get_U_fn()

    vols_orig = []

    # turn off minimization for this case to avoid inversions
    frames = simulate_system(U_fn, x0, num_samples=4000, minimize=False)
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
    frames = simulate_system(U_total, x0, num_samples=4000)
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
    assert ks < 0.05 or pv > 0.10


@pytest.mark.nocuda
@pytest.mark.parametrize(
    "check_chiral_atoms, check_chiral_bonds",
    [
        (True, False),
        pytest.param(True, True, marks=pytest.mark.xfail(reason="chiral bonds not supported yet")),
    ],
)
def test_chiral_topology(check_chiral_atoms, check_chiral_bonds):
    # test adding chiral restraints to the base topology
    # this molecule has several chiral atoms and bonds
    # 1) setup the molecule with a particular set of chiral restraints
    # 2) ensure that the chiral volumes are maintained
    # 3) flip the chiral centers in the molecule
    # 4) ensure that the chiral volumes are still maintained

    mol = Chem.MolFromMolBlock(
        """
  Mrv2202 06202217363D

 24 24  0  0  1  0            999 V2000
   -0.3734    4.7642    2.3521 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0019    3.8324    3.2247 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.4796    4.1797    4.4295 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.6670    5.5538    4.8525 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.2042    2.4457    2.8398 C   0  0  2  0  0  0  0  0  0  0  0  0
    0.9504    1.4275    2.9292 C   0  0  1  0  0  0  0  0  0  0  0  0
    0.5758    0.1691    3.3387 F   0  0  0  0  0  0  0  0  0  0  0  0
    0.4798    1.9291    1.5460 C   0  0  2  0  0  0  0  0  0  0  0  0
    1.7151    2.8906    0.7269 Cl  0  0  0  0  0  0  0  0  0  0  0  0
   -0.2499    0.9667    0.6977 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.4760    1.1343    0.4911 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.4405   -0.0544    0.1286 N   0  0  0  0  0  0  0  0  0  0  0  0
   -0.1222   -1.0583   -0.7063 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.4185    4.9284    1.7990 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.7274    3.4643    5.0757 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.0308    5.5861    5.8818 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.2770    6.1005    4.8053 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.3989    6.0481    4.2107 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.1917    2.0257    3.0495 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.8973    1.8191    3.3086 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.3905   -0.1108    0.3163 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.6896   -1.6147   -1.1775 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7243   -1.7418   -0.1044 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7418   -0.6160   -1.4890 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  2  0  0  0  0
  3  4  1  0  0  0  0
  5  2  1  0  0  0  0
  5  6  1  0  0  0  0
  6  7  1  0  0  0  0
  6  8  1  0  0  0  0
  5  8  1  0  0  0  0
  8  9  1  0  0  0  0
  8 10  1  0  0  0  0
 10 11  2  0  0  0  0
 10 12  1  0  0  0  0
 12 13  1  0  0  0  0
  1 14  1  0  0  0  0
  3 15  1  0  0  0  0
  4 16  1  0  0  0  0
  4 17  1  0  0  0  0
  4 18  1  0  0  0  0
  5 19  1  0  0  0  0
  6 20  1  0  0  0  0
 12 21  1  0  0  0  0
 13 22  1  0  0  0  0
 13 23  1  0  0  0  0
 13 24  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    top = topology.BaseTopology(mol, ff)
    system = top.setup_chiral_end_state()
    U_fn = system.get_U_fn()

    x0 = utils.get_romol_conf(mol)

    chiral_atom, chiral_bond = top.setup_chiral_restraints(
        DEFAULT_CHIRAL_ATOM_RESTRAINT_K, DEFAULT_CHIRAL_BOND_RESTRAINT_K
    )

    def get_chiral_atom_volumes(x):
        volumes = []
        for idxs in chiral_atom.potential.idxs:
            vol = chiral_restraints.pyramidal_volume(*x[idxs])
            volumes.append(vol)
        return np.array(volumes)

    ref_chiral_atom_vols = get_chiral_atom_volumes(x0)

    # should initially be all zero
    assert np.all(ref_chiral_atom_vols < 0)

    def get_chiral_bond_volumes(x):
        volumes = []
        for idxs in chiral_bond.potential.idxs:
            vol = chiral_restraints.torsion_volume(*x[idxs])
            volumes.append(vol)
        return np.array(volumes)

    ref_chiral_bond_vols = get_chiral_bond_volumes(x0)

    for bond_vol, bond_sign in zip(ref_chiral_bond_vols, chiral_bond.potential.signs):
        assert bond_vol * bond_sign < 0

    def assert_same_signs(a, b):
        np.testing.assert_array_equal(np.sign(a), np.sign(b))

    # frames = simulate_system(U_fn, x0, num_samples=1000)
    # for f in frames:
    #     frame_atom_vols = get_chiral_atom_volumes(f)
    #     frame_bond_vols = get_chiral_bond_volumes(f)
    #     assert_same_signs(frame_atom_vols, ref_chiral_atom_vols)
    #     assert_same_signs(frame_bond_vols, ref_chiral_bond_vols)

    # mol_inverted has 2 inverted chiral atoms (with halogen substituents)
    # and it has 2 inverted bonds (turned into trans as opposed to cis)
    mol_inverted = Chem.MolFromMolBlock(
        """
  Mrv2202 06212221003D

 24 24  0  0  1  0            999 V2000
   -1.0424    1.7119    1.8942 O   0  0  0  0  0  0  0  0  0  0  0  0
   -1.7608    1.0273    3.0719 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.8822    1.7709    3.8527 C   0  0  0  0  0  0  0  0  0  0  0  0
   -3.6321    1.3210    4.6802 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.3319   -0.5644    3.4121 C   0  0  2  0  0  0  0  0  0  0  0  0
   -0.8723   -1.0508    5.0681 C   0  0  2  0  0  0  0  0  0  0  0  0
   -0.1822   -0.3687    6.1461 F   0  0  0  0  0  0  0  0  0  0  0  0
    0.3854   -0.9252    3.8017 C   0  0  1  0  0  0  0  0  0  0  0  0
   -0.1560   -2.5944    3.3872 Cl  0  0  0  0  0  0  0  0  0  0  0  0
    1.7969   -0.5098    3.6952 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.9034    0.8386    2.7836 O   0  0  0  0  0  0  0  0  0  0  0  0
    2.7720   -1.3059    4.6376 N   0  0  0  0  0  0  0  0  0  0  0  0
    2.7936   -1.4907    6.1388 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.1267    2.2034    2.3497 H   0  0  0  0  0  0  0  0  0  0  0  0
   -3.6043    3.1709    3.5003 H   0  0  0  0  0  0  0  0  0  0  0  0
   -4.5912    1.6572    5.4334 H   0  0  0  0  0  0  0  0  0  0  0  0
   -4.2788    0.3280    4.5050 H   0  0  0  0  0  0  0  0  0  0  0  0
   -3.4602    0.6805    5.8790 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.9414   -1.4599    2.7562 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.4946   -1.9847    5.6352 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.7342   -2.2251    4.1323 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.3010   -0.7731    6.8337 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.8775   -1.3466    6.5793 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.2026   -2.2924    6.8143 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  2  0  0  0  0
  3  4  1  0  0  0  0
  5  2  1  0  0  0  0
  5  6  1  0  0  0  0
  6  7  1  0  0  0  0
  6  8  1  0  0  0  0
  5  8  1  0  0  0  0
  8  9  1  0  0  0  0
  8 10  1  0  0  0  0
 10 11  2  0  0  0  0
 10 12  1  0  0  0  0
 12 13  1  0  0  0  0
  1 14  1  0  0  0  0
  3 15  1  0  0  0  0
  4 16  1  0  0  0  0
  4 17  1  0  0  0  0
  4 18  1  0  0  0  0
  5 19  1  0  0  0  0
  6 20  1  0  0  0  0
 12 21  1  0  0  0  0
 13 22  1  0  0  0  0
 13 23  1  0  0  0  0
 13 24  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    x0_inverted = utils.get_romol_conf(mol_inverted)

    # don't remove - useful for visualization
    # from timemachine.fe import cif_writer
    # traj = minimize_scipy(U_fn, x0_inverted, return_traj=True)
    # writer = cif_writer.CIFWriter([mol], "chiral_inversion.cif")
    # for f in traj:
    #     f = f - np.mean(f, axis=0)
    #     writer.write_frame(f*10)
    # writer.close()

    frames = simulate_system(U_fn, x0_inverted, num_samples=1000)

    for f in frames:
        frame_atom_vols = get_chiral_atom_volumes(f)
        frame_bond_vols = get_chiral_bond_volumes(f)
        # f = f - np.mean(f, axis=0)
        # writer.write_frame(f*10)
        if check_chiral_atoms:
            assert_same_signs(frame_atom_vols, ref_chiral_atom_vols)
        if check_chiral_bonds:
            assert_same_signs(frame_bond_vols, ref_chiral_bond_vols)


def make_chiral_flip_pair(well_aligned=True):
    # mol_a, mol_b : substituted chiral cyclobutyl
    # with 2 different alignments of mol_b w.r.t. mol_a
    with path_to_internal_file("timemachine.testsystems.data", "1243_chiral_ring_confs.sdf") as path_to_sdf:
        mol_dict = utils.read_sdf_mols_by_name(path_to_sdf)
    mol_a = mol_dict["A"]
    mol_b_0 = mol_dict["B_0"]
    mol_b_1 = mol_dict["B_1"]

    # hard-coded core
    core_0 = np.array([[6, 3], [4, 4], [5, 5], [3, 6], [15, 10], [14, 11], [12, 12], [13, 13], [11, 14], [10, 15]])
    core_1 = np.array(
        [
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [12, 5],
            [6, 6],
            [7, 7],
            [9, 8],
            [8, 9],
            [10, 10],
            [11, 11],
            [5, 12],
            [15, 14],
            [14, 15],
        ]
    )

    if well_aligned:
        return AtomMapMixin(mol_a, mol_b_1, core_1)
    else:
        return AtomMapMixin(mol_a, mol_b_0, core_0)


@pytest.mark.parametrize("well_aligned", [True, False])
def test_chiral_inversion_in_single_topology_runs(well_aligned):
    """simply test that no exceptions are raised, when running vacuum hrex"""
    very_short_hrex_params = replace(
        DEFAULT_HREX_PARAMS, n_frames=2, n_eq_steps=2, steps_per_frame=2, hrex_params=HREXParams(n_frames_bisection=2)
    )
    ff = Forcefield.load_default()

    atom_map = make_chiral_flip_pair(well_aligned)
    _ = run_vacuum(atom_map.mol_a, atom_map.mol_b, atom_map.core, ff, None, very_short_hrex_params, n_windows=3)


@pytest.mark.nightly(reason="slow")
@pytest.mark.parametrize("well_aligned", [True, False])
def test_chiral_inversion_in_single_topology(well_aligned):
    """assert chiral consistency preserved through vacuum HREX"""

    ff = Forcefield.load_default()

    atom_map = make_chiral_flip_pair(well_aligned)

    vacuum_results = run_vacuum(
        atom_map.mol_a,
        atom_map.mol_b,
        atom_map.core,
        ff,
        None,
        DEFAULT_HREX_PARAMS,
        min_overlap=0.667,  # No need to have a higher overlap then 0.667
    )
    heatmap_a, heatmap_b = make_chiral_flip_heatmaps(vacuum_results, atom_map)
    assert (heatmap_a[0] == 0).all(), "chirality in end state A was not preserved"
    assert (heatmap_b[-1] == 0).all(), "chirality in end state B was not preserved"

    # from timemachine.fe.plots import plot_as_png_fxn, plot_chiral_restraint_energies

    # with open(f"vacuum_chiral_energies_aligned_{well_aligned}_a.png", "wb") as ofs:
    #     data_a = plot_as_png_fxn(plot_chiral_restraint_energies, heatmap_a)
    #     ofs.write(data_a)

    # with open(f"vacuum_chiral_energies_aligned_{well_aligned}_b.png", "wb") as ofs:
    #     data_b = plot_as_png_fxn(plot_chiral_restraint_energies, heatmap_b)
    #     ofs.write(data_b)


@pytest.mark.nocuda
def test_chiral_restraint_energies_with_no_restraints():
    """VWhen the number of chiral restraints were zero could result in an exception"""
    mol_a = Chem.MolFromMolBlock(
        """
     RDKit          3D

 12 12  0  0  0  0  0  0  0  0999 V2000
    0.9820    1.0014    0.0105 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.3692    1.3257   -0.0095 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.3213    0.3320   -0.0196 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.9782   -0.9962   -0.0105 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.3622   -1.3280    0.0094 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.3301   -0.3374    0.0197 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.7618    1.7512    0.0190 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6369    2.3883   -0.0167 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.3674    0.6147   -0.0351 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.7688   -1.7441   -0.0191 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.6281   -2.3740    0.0165 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.3775   -0.6335    0.0353 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0
  2  3  1  0
  3  4  2  0
  4  5  1  0
  5  6  2  0
  6  1  1  0
  1  7  1  0
  2  8  1  0
  3  9  1  0
  4 10  1  0
  5 11  1  0
  6 12  1  0
M  END""",
        removeHs=False,
    )

    mol_b = Chem.MolFromMolBlock(
        """
     RDKit          3D

 13 14  0  0  0  0  0  0  0  0999 V2000
    0.6509    0.6640    0.0070 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0474   -0.0518    1.1787 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7335    0.7273   -0.5626 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.7357   -0.7070   -0.6174 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6472   -0.6589   -0.0069 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.3555    1.3705    0.0097 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5448    0.5192    1.8798 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.7823   -0.7798    1.5901 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7595    0.6979   -1.6679 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.4509    1.3947   -0.0683 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.7983   -0.7612   -1.6975 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.3850   -1.3532   -0.0133 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.3319   -1.3827   -0.0161 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1  3  1  0
  1  4  1  0
  4  5  1  0
  5  2  1  0
  5  3  1  0
  1  6  1  0
  2  7  1  0
  2  8  1  0
  3  9  1  0
  3 10  1  0
  4 11  1  0
  4 12  1  0
  5 13  1  0
M  END""",
        removeHs=False,
    )

    atom_mappng_kwargs = DEFAULT_ATOM_MAPPING_KWARGS.copy()
    core = get_cores(mol_a, mol_b, **atom_mappng_kwargs)[0]

    atom_map = AtomMapMixin(mol_a, mol_b, core)

    assert len(setup_all_chiral_atom_restr_idxs(mol_a, utils.get_romol_conf(mol_a))) == 0
    assert len(setup_all_chiral_atom_restr_idxs(mol_b, utils.get_romol_conf(mol_b))) == 20
    U_a, U_b = make_chiral_restr_fxns(atom_map.mol_a, atom_map.mol_b)
    assert U_a(utils.get_romol_conf(mol_a)) == 0.0
    assert U_b(utils.get_romol_conf(mol_b)) == 0.0


@pytest.mark.nightly(reason="slow")
def test_chiral_inversion_in_single_topology_solvent():
    """assert chiral consistency preserved through solvent HREX"""

    ff = Forcefield.load_default()

    # Run the poorly aligned version, restraints get exercised
    atom_map = make_chiral_flip_pair(False)

    md_params = DEFAULT_HREX_PARAMS
    md_params = replace(md_params, n_frames=100)

    res, _ = run_solvent(atom_map.mol_a, atom_map.mol_b, atom_map.core, ff, None, md_params, n_windows=3)
    heatmap_a, heatmap_b = make_chiral_flip_heatmaps(res, atom_map)
    assert (heatmap_a[0] == 0).all(), "chirality in end state A was not preserved"
    assert (heatmap_b[-1] == 0).all(), "chirality in end state B was not preserved"

    # from timemachine.fe.plots import plot_as_png_fxn, plot_chiral_restraint_energies

    # with open("solvent_chiral_energies_a.png", "wb") as ofs:
    #     data_a = plot_as_png_fxn(plot_chiral_restraint_energies, heatmap_a)
    #     ofs.write(data_a)

    # with open("solvent_chiral_energies_b.png", "wb") as ofs:
    #     data_b = plot_as_png_fxn(plot_chiral_restraint_energies, heatmap_b)
    #     ofs.write(data_b)
