from jax.config import config

config.update("jax_enable_x64", True)

import unittest
from importlib import resources
from typing import Tuple

import jax
import numpy as np
import pytest
from rdkit import Chem

from timemachine.constants import DEFAULT_FF
from timemachine.fe import topology
from timemachine.fe.utils import get_romol_conf
from timemachine.ff import Forcefield, combine_ordered_params
from timemachine.ff.handlers import AM1CCCHandler, openmm_deserializer
from timemachine.md import builders
from timemachine.testsystems.relative import hif2a_ligand_pair


class BenzenePhenolSparseTest(unittest.TestCase):
    def setUp(self, *args, **kwargs):

        suppl = Chem.SDMolSupplier("tests/data/benzene_phenol_sparse.sdf", removeHs=False)
        all_mols = [x for x in suppl]

        self.mol_a = all_mols[0]
        self.mol_b = all_mols[1]

        # atom type free
        self.ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

        super(BenzenePhenolSparseTest, self).__init__(*args, **kwargs)

    def test_bonded(self):
        # other bonded terms use an identical protocol, so we assume they're correct if the harmonic bond tests pass.
        # leaving benzene H unmapped, and phenol OH unmapped
        core = np.array(
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
                [5, 5],
            ],
            dtype=np.int32,
        )

        st = topology.SingleTopology(self.mol_a, self.mol_b, core, self.ff)

        combined_params, vjp_fn, combined_potential = jax.vjp(
            st.parameterize_harmonic_bond, self.ff.hb_handle.params, has_aux=True
        )

        # test that vjp_fn works
        vjp_fn(np.random.rand(*combined_params.shape))

        # we expect 15 bonds in total, of which 6 are duplicated.
        assert len(combined_potential.get_idxs() == 15)

        src_idxs = set([tuple(x) for x in combined_potential.get_idxs()[:6]])
        dst_idxs = set([tuple(x) for x in combined_potential.get_idxs()[6:12]])

        np.testing.assert_equal(src_idxs, dst_idxs)

        cc = self.ff.hb_handle.lookup_smirks("[#6X3:1]:[#6X3:2]")
        cH = self.ff.hb_handle.lookup_smirks("[#6X3:1]-[#1:2]")
        cO = self.ff.hb_handle.lookup_smirks("[#6X3:1]-[#8X2H1:2]")
        OH = self.ff.hb_handle.lookup_smirks("[#8:1]-[#1:2]")

        params_src = combined_params[:6]
        params_dst = combined_params[6:12]
        params_uni = combined_params[12:]

        np.testing.assert_array_equal(params_src, [cc, cc, cc, cc, cc, cc])
        np.testing.assert_array_equal(params_dst, [cc, cc, cc, cc, cc, cc])
        np.testing.assert_array_equal(params_uni, [cH, cO, OH])

        # map H to O
        core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], dtype=np.int32)

        st = topology.SingleTopology(self.mol_a, self.mol_b, core, self.ff)

        combined_params, vjp_fn, combined_potential = jax.vjp(
            st.parameterize_harmonic_bond, self.ff.hb_handle.params, has_aux=True
        )

        assert len(combined_potential.get_idxs() == 15)

        src_idxs = set([tuple(x) for x in combined_potential.get_idxs()[:7]])
        dst_idxs = set([tuple(x) for x in combined_potential.get_idxs()[7:14]])

        params_src = combined_params[:7]
        params_dst = combined_params[7:14]
        params_uni = combined_params[14:]

        np.testing.assert_array_equal(params_src, [cc, cc, cc, cc, cc, cc, cH])
        np.testing.assert_array_equal(params_dst, [cc, cc, cc, cc, cc, cc, cO])
        np.testing.assert_array_equal(params_uni, [OH])

        # test that vjp_fn works
        vjp_fn(np.random.rand(*combined_params.shape))

    def test_nonbonded(self):

        # leaving benzene H unmapped, and phenol OH unmapped
        core = np.array(
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
                [5, 5],
            ],
            dtype=np.int32,
        )

        st = topology.SingleTopology(self.mol_a, self.mol_b, core, self.ff)
        x_a = get_romol_conf(self.mol_a)
        x_b = get_romol_conf(self.mol_b)

        # test interpolation of coordinates.
        x_src, x_dst = st.interpolate_params(x_a, x_b)
        x_avg = np.mean([x_src, x_dst], axis=0)

        assert x_avg.shape == (st.get_num_atoms(), 3)

        np.testing.assert_array_equal((x_a[:6] + x_b[:6]) / 2, x_avg[:6])  # C
        np.testing.assert_array_equal(x_a[6], x_avg[6])  # H
        np.testing.assert_array_equal(x_b[6:], x_avg[7:])  # OH

        # NOTE: unused result
        st.parameterize_nonbonded(self.ff.q_handle.params, self.ff.lj_handle.params)

        params, vjp_fn, pot_c = jax.vjp(
            st.parameterize_nonbonded, self.ff.q_handle.params, self.ff.lj_handle.params, has_aux=True
        )

        vjp_fn(np.random.rand(*params.shape))

        assert params.shape == (2 * st.get_num_atoms(), 3)  # qlj

        # test interpolation of parameters
        bt_a = topology.BaseTopology(self.mol_a, self.ff)
        qlj_a, pot_a = bt_a.parameterize_nonbonded(self.ff.q_handle.params, self.ff.lj_handle.params)
        bt_b = topology.BaseTopology(self.mol_b, self.ff)
        qlj_b, pot_b = bt_b.parameterize_nonbonded(self.ff.q_handle.params, self.ff.lj_handle.params)

        n_base_params = len(params) // 2  # params is actually interpolated, so its 2x number of base params

        # qlj_c = np.mean([params[:n_base_params], params[n_base_params:]], axis=0)

        params_src = params[:n_base_params]
        params_dst = params[n_base_params:]

        # core testing
        np.testing.assert_array_equal(qlj_a[:6], params_src[:6])
        np.testing.assert_array_equal(qlj_b[:6], params_dst[:6])

        # test r-group in A
        np.testing.assert_array_equal(qlj_a[6], params_src[6])
        np.testing.assert_array_equal(np.array([0, qlj_a[6][1], 0]), params_dst[6])

        # test r-group in B
        np.testing.assert_array_equal(qlj_b[6:], params_dst[7:])
        np.testing.assert_array_equal(np.array([[0, qlj_b[6][1], 0], [0, qlj_b[7][1], 0]]), params_src[7:])

    def test_nonbonded_optimal_map(self):
        """Similar test as test_nonbonbed, ie. assert that coordinates and nonbonded parameters
        can be averaged in benzene -> phenol transformation. However, use the maximal mapping possible."""

        # map benzene H to phenol O, leaving a dangling phenol H
        core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], dtype=np.int32)

        st = topology.SingleTopology(self.mol_a, self.mol_b, core, self.ff)
        x_a = get_romol_conf(self.mol_a)
        x_b = get_romol_conf(self.mol_b)

        # test interpolation of coordinates.
        x_src, x_dst = st.interpolate_params(x_a, x_b)
        x_avg = np.mean([x_src, x_dst], axis=0)

        assert x_avg.shape == (st.get_num_atoms(), 3)

        np.testing.assert_array_equal((x_a[:7] + x_b[:7]) / 2, x_avg[:7])  # core parts
        np.testing.assert_array_equal(x_b[-1], x_avg[7])  # dangling H

        params, vjp_fn, pot_c = jax.vjp(
            st.parameterize_nonbonded, self.ff.q_handle.params, self.ff.lj_handle.params, has_aux=True
        )

        vjp_fn(np.random.rand(*params.shape))

        assert params.shape == (2 * st.get_num_atoms(), 3)  # qlj

        # test interpolation of parameters
        bt_a = topology.BaseTopology(self.mol_a, self.ff)
        qlj_a, pot_a = bt_a.parameterize_nonbonded(self.ff.q_handle.params, self.ff.lj_handle.params)
        bt_b = topology.BaseTopology(self.mol_b, self.ff)
        qlj_b, pot_b = bt_b.parameterize_nonbonded(self.ff.q_handle.params, self.ff.lj_handle.params)

        n_base_params = len(params) // 2  # params is actually interpolated, so its 2x number of base params

        # qlj_c = np.mean([params[:n_base_params], params[n_base_params:]], axis=0)

        params_src = params[:n_base_params]
        params_dst = params[n_base_params:]

        # core testing
        np.testing.assert_array_equal(qlj_a[:7], params_src[:7])
        np.testing.assert_array_equal(qlj_b[:7], params_dst[:7])

        # r-group atoms in A are all part of the core. so no testing is needed.

        # test r-group in B
        np.testing.assert_array_equal(qlj_b[7], params_dst[8])
        np.testing.assert_array_equal(np.array([0, qlj_b[7][1], 0]), params_src[8])


class TestFactorizability(unittest.TestCase):
    def test_bad_factor(self):
        # test a bad mapping that results in a non-cancellable endpoint
        with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
            suppl = Chem.SDMolSupplier(str(path_to_ligand), removeHs=False)

        all_mols = [x for x in suppl]
        mol_a = all_mols[0]
        mol_b = all_mols[1]

        ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

        core = np.array(
            [
                [4, 1],
                [5, 2],
                [6, 3],
                [7, 4],
                [8, 5],
                [9, 6],
                [10, 7],
                [11, 8],
                [12, 9],
                [13, 10],
                [15, 11],
                [16, 12],
                [18, 14],
                [34, 31],
                [17, 13],
                [23, 23],
                [33, 30],
                [32, 28],
                [31, 27],
                [30, 26],
                [19, 15],
                [20, 16],
                [21, 17],
            ]
        )

        with self.assertRaises(topology.AtomMappingError):
            topology.SingleTopology(mol_a, mol_b, core, ff)

    def test_good_factor(self):
        # test a good mapping
        with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
            suppl = Chem.SDMolSupplier(str(path_to_ligand), removeHs=False)

        all_mols = [x for x in suppl]
        mol_a = all_mols[1]
        mol_b = all_mols[4]

        ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

        core = np.array(
            [
                [0, 0],
                [2, 2],
                [1, 1],
                [6, 6],
                [5, 5],
                [4, 4],
                [3, 3],
                [15, 16],
                [16, 17],
                [17, 18],
                [18, 19],
                [19, 20],
                [20, 21],
                [32, 30],
                [26, 25],
                [27, 26],
                [7, 7],
                [8, 8],
                [9, 9],
                [10, 10],
                [29, 11],
                [11, 12],
                [12, 13],
                [14, 15],
                [31, 29],
                [13, 14],
                [23, 24],
                [30, 28],
                [28, 27],
                [21, 22],
            ]
        )

        st = topology.SingleTopology(mol_a, mol_b, core, ff)

        # test that the vjps work
        _ = jax.vjp(st.parameterize_harmonic_bond, ff.hb_handle.params, has_aux=True)
        _ = jax.vjp(st.parameterize_harmonic_angle, ff.ha_handle.params, has_aux=True)
        _ = jax.vjp(st.parameterize_periodic_torsion, ff.pt_handle.params, ff.it_handle.params, has_aux=True)
        _ = jax.vjp(st.parameterize_nonbonded, ff.q_handle.params, ff.lj_handle.params, has_aux=True)


def test_component_idxs():
    # single topology
    # fmt: off
    mol_a_idxs = np.array([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33], dtype=np.int32)
    mol_b_idxs = np.array([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 29, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 34, 23, 26, 27, 28, 30, 31, 32], dtype=np.int32)
    # fmt: on
    np.testing.assert_equal(hif2a_ligand_pair.top.get_component_idxs(), [mol_a_idxs, mol_b_idxs])

    # single topology w/host
    solvent_system, coords, _, _ = builders.build_water_system(4.0)
    host_bps, _ = openmm_deserializer.deserialize_system(solvent_system, cutoff=1.2)
    hgt_single = topology.HostGuestTopology(host_bps, hif2a_ligand_pair.top)

    num_solvent_atoms = coords.shape[0]
    solvent_idxs = np.arange(num_solvent_atoms)

    np.testing.assert_equal(
        hgt_single.get_component_idxs(), [solvent_idxs, mol_a_idxs + num_solvent_atoms, mol_b_idxs + num_solvent_atoms]
    )

    # dual topology
    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        suppl = Chem.SDMolSupplier(str(path_to_ligand), removeHs=False)

    all_mols = [x for x in suppl]
    mol_a = all_mols[1]
    mol_b = all_mols[4]
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    dt = topology.DualTopology(mol_a, mol_b, ff)

    mol_a_idxs = np.arange(mol_a.GetNumAtoms())
    mol_b_idxs = np.arange(mol_b.GetNumAtoms()) + mol_a.GetNumAtoms()
    np.testing.assert_equal(dt.get_component_idxs(), [mol_a_idxs, mol_b_idxs])

    # dual topology w/host
    hgt_dual = topology.HostGuestTopology(host_bps, dt)
    np.testing.assert_equal(
        hgt_dual.get_component_idxs(), [solvent_idxs, mol_a_idxs + num_solvent_atoms, mol_b_idxs + num_solvent_atoms]
    )

    # base topology
    bt = topology.BaseTopology(mol_a, ff)
    np.testing.assert_equal(bt.get_component_idxs(), [mol_a_idxs])

    # base topology w/host
    hgt_base = topology.HostGuestTopology(host_bps, bt)
    np.testing.assert_equal(hgt_base.get_component_idxs(), [solvent_idxs, mol_a_idxs + num_solvent_atoms])


def test_relative_free_energy_forcefield():
    BOND, ANGLE, PROPER, IMPROPER, CHARGE, LJ = 0, 1, 2, 3, 4, 5

    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        mol = next(Chem.SDMolSupplier(str(path_to_ligand), removeHs=False))

    ff0 = Forcefield.load_from_file(DEFAULT_FF)
    ff1 = Forcefield.load_from_file(DEFAULT_FF)

    # Modify the charge parameters for ff1
    for h, p in zip(ff1.get_ordered_handles(), ff1.get_ordered_params()):
        if isinstance(h, AM1CCCHandler):
            p += 1.0

    fftop = topology.RelativeFreeEnergyForcefield(mol, ff0, ff1)
    bt0 = topology.BaseTopology(mol, ff0)
    bt1 = topology.BaseTopology(mol, ff1)

    combined_params = combine_ordered_params(ff0, ff1)
    combined_qlj_params, combined_ubp = fftop.parameterize_nonbonded(combined_params[CHARGE], combined_params[LJ])
    qlj0_params, ubp0 = bt0.parameterize_nonbonded(ff0.get_ordered_params()[CHARGE], ff0.get_ordered_params()[LJ])
    qlj1_params, ubp1 = bt1.parameterize_nonbonded(ff1.get_ordered_params()[CHARGE], ff1.get_ordered_params()[LJ])

    coords = get_romol_conf(mol)
    box = np.identity(3) * 99.0

    def u_endstates(ubp, params) -> Tuple[float, float]:
        u = ubp.bind(params).bound_impl(precision=np.float32)
        _, _, u0_val = u.execute(coords, box, lam=0)
        _, _, u1_val = u.execute(coords, box, lam=1)
        return u0_val, u1_val

    u0_combined, u1_combined = u_endstates(combined_ubp, combined_qlj_params)
    u0, _ = u_endstates(ubp0, qlj0_params)
    u1, _ = u_endstates(ubp1, qlj1_params)

    # Check that the endstate NB energies are consistent
    assert pytest.approx(u0_combined) == u0
    assert pytest.approx(u1_combined) == u1

    # Check that other terms can not be changed
    fftop.parameterize_harmonic_bond(combined_params[BOND])
    invalid = [combined_params[BOND][0], combined_params[BOND][0] + 1.0]
    with pytest.raises(AssertionError, match="changing harmonic bond"):
        fftop.parameterize_harmonic_bond(invalid)

    fftop.parameterize_harmonic_angle(combined_params[ANGLE])
    invalid = [combined_params[ANGLE][0], combined_params[ANGLE][0] + 1.0]
    with pytest.raises(AssertionError, match="changing harmonic angle"):
        fftop.parameterize_harmonic_angle(invalid)

    fftop.parameterize_periodic_torsion(combined_params[PROPER], combined_params[IMPROPER])
    invalid = [combined_params[PROPER][0], combined_params[PROPER][0] + 1.0]
    with pytest.raises(AssertionError, match="changing proper"):
        fftop.parameterize_periodic_torsion(invalid, combined_params[IMPROPER])
    invalid = [combined_params[IMPROPER][0], combined_params[IMPROPER][0] + 1.0]
    with pytest.raises(AssertionError, match="changing improper"):
        fftop.parameterize_periodic_torsion(combined_params[PROPER], invalid)
