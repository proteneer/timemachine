from tempfile import NamedTemporaryFile

import numpy as np
import pytest
from common import ligand_from_smiles
from openmm import app, unit
from rdkit import Chem

from timemachine.constants import DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF, ONE_4PI_EPS0, NBParamIdx
from timemachine.fe.utils import get_romol_conf, read_sdf, set_romol_conf
from timemachine.ff import sanitize_water_ff
from timemachine.md.barostat.utils import compute_box_volume, get_bond_list, get_group_indices
from timemachine.md.builders import build_protein_system, build_water_system
from timemachine.md.minimizer import check_force_norm
from timemachine.potentials import Nonbonded
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology
from timemachine.utils import path_to_internal_file


def test_build_water_system():
    mol_a, mol_b, _ = get_hif2a_ligand_pair_single_topology()
    host_config = build_water_system(4.0, DEFAULT_WATER_FF)
    host_with_mols_config = build_water_system(4.0, DEFAULT_WATER_FF, mols=[mol_a, mol_b])

    # No waters should be deleted, but the box will be slightly larger
    assert len(host_config.conf) == len(host_with_mols_config.conf)
    assert compute_box_volume(host_config.box) < compute_box_volume(host_with_mols_config.box)

    mol_coords = np.concatenate([get_romol_conf(mol_a), get_romol_conf(mol_b)])
    mol_centroid = np.mean(mol_coords, axis=0)

    water_centeroid = np.mean(host_config.conf, axis=0)

    # The centroid of the water particles should be near the centroid of the ligand
    mol_water_centeroid = np.mean(host_with_mols_config.conf, axis=0)
    np.testing.assert_allclose(mol_centroid, mol_water_centeroid, atol=2e-2)

    # Dependent on the molecule (where it was posed in complex), but centroid of water will not be near the ligands
    assert not np.allclose(mol_centroid, water_centeroid, atol=2e-2)

    host_config.box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes around boundary
    host_with_mols_config.box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes around boundary

    for bp in host_config.host_system.get_U_fns():
        (
            du_dx,
            _,
        ) = bp.to_gpu(np.float32).bound_impl.execute(host_config.conf, host_config.box, compute_u=False)
        check_force_norm(-du_dx)

    for bp in host_with_mols_config.host_system.get_U_fns():
        (
            du_dx,
            _,
        ) = bp.to_gpu(np.float32).bound_impl.execute(
            host_with_mols_config.conf, host_with_mols_config.box, compute_u=False
        )
        check_force_norm(-du_dx)


@pytest.mark.nocuda
@pytest.mark.parametrize("water_ff", ["amber14/tip3p", "amber14/tip4pfb", "amber14/spce", "tip5p"])
def test_build_water_system_different_water_ffs(water_ff):
    mol_a, mol_b, _ = get_hif2a_ligand_pair_single_topology()
    host_config = build_water_system(4.0, water_ff, mols=[mol_a, mol_b])
    host_config.box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes around boundary
    for bp in host_config.host_system.get_U_fns():
        # Skip the nonbonded potential, as a lot of memory is required when using the CPU JAX platform
        if isinstance(bp.potential, Nonbonded):
            continue
        assert np.isfinite(bp(host_config.conf, host_config.box))


@pytest.mark.nocuda
def test_build_protein_system_returns_correct_water_count():
    with path_to_internal_file("timemachine.testsystems.fep_benchmark.pfkfb3", "ligands.sdf") as sdf_path:
        mols = read_sdf(sdf_path)
    # Pick two arbitrary mols
    mol_a = mols[0]
    mol_b = mols[1]
    last_num_waters = None
    # Verify that even adding different molecules produces the same number of waters in the system
    for mols in (None, [mol_a], [mol_b], [mol_a, mol_b]):
        with path_to_internal_file("timemachine.testsystems.fep_benchmark.pfkfb3", "6hvi_prepared.pdb") as pdb_path:
            host_config = build_protein_system(str(pdb_path), DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF, mols=mols)
            # The builder should not modify the number of atoms in the protein at all
            # Hard coded to the number of protein atoms in the PDB, refer to 6hvi_prepared.pdb for the actual
            # number of atoms
            assert host_config.conf.shape[0] - host_config.num_water_atoms == 6748
            if last_num_waters is not None:
                assert last_num_waters == host_config.num_water_atoms
            last_num_waters = host_config.num_water_atoms


@pytest.mark.nocuda
@pytest.mark.parametrize("ionic_concentration", [0.0, 0.15])
@pytest.mark.parametrize("neutralize", [False, True])
def test_water_system_ion_concentration_and_neutralization(ionic_concentration, neutralize):
    positive_mol = ligand_from_smiles("c1cc[nH+]cc1")
    negative_mol = ligand_from_smiles("[N+](=O)([O-])[O-]")
    neutral_mol = ligand_from_smiles("c1ccccc1")

    box_size = 2.0

    host_config_no_ions = build_water_system(box_size, DEFAULT_WATER_FF, ionic_concentration=0.0)
    # Host system will have zero net charge if no ionic concentration and not neutralized
    assert np.sum(host_config_no_ions.host_system.nonbonded_all_pairs.params[:, NBParamIdx.Q_IDX]) == 0.0

    # Can't mix ligands of different charges when neutralizing the system
    if neutralize:
        with pytest.raises(AssertionError):
            build_water_system(
                box_size,
                DEFAULT_WATER_FF,
                mols=[positive_mol, negative_mol],
                ionic_concentration=ionic_concentration,
                neutralize=neutralize,
            )
    else:
        build_water_system(
            box_size,
            DEFAULT_WATER_FF,
            mols=[positive_mol, negative_mol],
            ionic_concentration=ionic_concentration,
            neutralize=neutralize,
        )
    for mol in [positive_mol, negative_mol, neutral_mol]:
        host_config = build_water_system(
            box_size, DEFAULT_WATER_FF, mols=[mol], ionic_concentration=ionic_concentration, neutralize=neutralize
        )
        expected_charge = 0
        if neutralize:
            # Since the ligand isn't in the system, should be missing the charge of the ligand
            expected_charge = -Chem.GetFormalCharge(mol)
        np.testing.assert_allclose(
            np.sum(host_config.host_system.nonbonded_all_pairs.params[:, NBParamIdx.Q_IDX]) / np.sqrt(ONE_4PI_EPS0),
            expected_charge,
            atol=1e-15,
        )
        bond_indices = get_bond_list(host_config.host_system.bond.potential)

        all_group_idxs = get_group_indices(bond_indices, host_config.conf.shape[0])
        ions = [group for group in all_group_idxs if len(group) == 1]
        num_ions = len(ions)
        if ionic_concentration > 0.0:
            assert num_ions > 0
            assert num_ions % 2 == abs(expected_charge)
        else:
            # Should have the number of ions extra to account for the charge of the ligand
            assert num_ions == abs(expected_charge)


@pytest.mark.nocuda
@pytest.mark.parametrize("ionic_concentration", [0.0, 0.15])
@pytest.mark.parametrize("neutralize", [False, True])
def test_protein_system_ion_concentration_and_neutralization(ionic_concentration, neutralize):
    # Note that none of this ligands go with the protein, but as long as we don't minimize, all is well.
    positive_mol = ligand_from_smiles("c1cc[nH+]cc1")
    negative_mol = ligand_from_smiles("[N+](=O)([O-])[O-]")
    neutral_mol = ligand_from_smiles("c1ccccc1")

    with path_to_internal_file("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as pdb_path:
        host_pdbfile = str(pdb_path)

    host_config_no_ions = build_protein_system(
        host_pdbfile, DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF, ionic_concentration=0.0, neutralize=False
    )
    # Without neutralizing the system, the protein system may have some charge
    reference_protein_charge = np.sum(
        host_config_no_ions.host_system.nonbonded_all_pairs.params[:, NBParamIdx.Q_IDX]
    ) / np.sqrt(ONE_4PI_EPS0)

    # Can't mix ligands of different charges when neutralizing the system
    if neutralize:
        with pytest.raises(AssertionError):
            build_protein_system(
                host_pdbfile,
                DEFAULT_PROTEIN_FF,
                DEFAULT_WATER_FF,
                mols=[positive_mol, negative_mol],
                ionic_concentration=ionic_concentration,
                neutralize=neutralize,
            )
    else:
        build_protein_system(
            host_pdbfile,
            DEFAULT_PROTEIN_FF,
            DEFAULT_WATER_FF,
            mols=[positive_mol, negative_mol],
            ionic_concentration=ionic_concentration,
            neutralize=neutralize,
        )
    for mol in [positive_mol, negative_mol, neutral_mol]:
        host_config = build_protein_system(
            host_pdbfile,
            DEFAULT_PROTEIN_FF,
            DEFAULT_WATER_FF,
            mols=[mol],
            ionic_concentration=ionic_concentration,
            neutralize=neutralize,
        )
        expected_charge = reference_protein_charge
        if neutralize:
            # Since the ligand isn't in the system, should be missing the charge of the ligand
            expected_charge = -Chem.GetFormalCharge(mol)
        np.testing.assert_allclose(
            np.sum(host_config.host_system.nonbonded_all_pairs.params[:, NBParamIdx.Q_IDX]) / np.sqrt(ONE_4PI_EPS0),
            expected_charge,
            atol=1.5e-15,
        )
        bond_indices = get_bond_list(host_config.host_system.bond.potential)

        all_group_idxs = get_group_indices(bond_indices, host_config.conf.shape[0])
        ions = [group for group in all_group_idxs if len(group) == 1]
        num_ions = len(ions)
        if ionic_concentration > 0.0:
            assert num_ions > 0
            if neutralize:
                assert num_ions % 2 == abs(Chem.GetFormalCharge(mol) + int(np.rint(reference_protein_charge))) % 2
            else:
                assert num_ions % 2 == 0
        elif neutralize:
            # Should have the number of ions extra to account for the charge of the ligand
            assert num_ions == abs(Chem.GetFormalCharge(mol) + int(np.rint(reference_protein_charge)))
        else:
            assert num_ions == 0


@pytest.mark.nocuda
def test_deserialize_protein_system_1_4_exclusions():
    with path_to_internal_file("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as pdb_path:
        host_pdbfile = str(pdb_path)
    host_config = build_protein_system(host_pdbfile, DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF)

    exclusion_idxs = host_config.host_system.nonbonded_all_pairs.potential.exclusion_idxs
    scale_factors = host_config.host_system.nonbonded_all_pairs.potential.scale_factors

    kvs = dict()
    for (src, dst), (q_sf, lj_sf) in zip(exclusion_idxs, scale_factors):
        kvs[(src, dst)] = (q_sf, lj_sf)

    # 1-4 torsion between H-ACE and carbonyl=O, expected behavior:
    # we should remove 1/6th of the electrostatic strength
    # we should remove 1/2 of the lennard jones strength
    np.testing.assert_almost_equal(kvs[(2, 3)][0], 0.5, decimal=4)  # TODO: differs from OFF 1/6
    np.testing.assert_almost_equal(kvs[(2, 3)][1], 0.5, decimal=4)

    np.testing.assert_almost_equal(kvs[(2, 4)][0], 0.5, decimal=4)  # TODO: differs from OFF 1/6
    np.testing.assert_almost_equal(kvs[(2, 4)][1], 0.5, decimal=4)

    np.testing.assert_almost_equal(kvs[(2, 5)][0], 0.5, decimal=4)  # TODO: differs from OFF 1/6
    np.testing.assert_almost_equal(kvs[(2, 5)][1], 0.5, decimal=4)

    # 1-3 angle term should be completely removed
    np.testing.assert_almost_equal(kvs[(3, 4)][0], 1.0, decimal=4)
    np.testing.assert_almost_equal(kvs[(3, 4)][1], 1.0, decimal=4)


@pytest.mark.nocuda
def test_build_protein_system_waters_before_protein():
    num_waters = 100
    # Construct a PDB file with the waters before the protein, should raise an exception
    with path_to_internal_file("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as pdb_path:
        host_pdbfile = host_pdb = app.PDBFile(str(pdb_path))

    host_ff = app.ForceField(f"{DEFAULT_PROTEIN_FF}.xml", f"{DEFAULT_WATER_FF}.xml")

    top = app.Topology()
    pos = unit.Quantity((), unit.angstroms)
    modeller = app.Modeller(top, pos)
    modeller.addSolvent(host_ff, numAdded=num_waters, neutralize=False, model=sanitize_water_ff(DEFAULT_WATER_FF))
    assert modeller.getTopology().getNumAtoms() == num_waters * 3

    modeller.add(host_pdbfile.topology, host_pdb.positions)

    with NamedTemporaryFile(suffix=".pdb") as temp:
        with open(temp.name, "w") as ofs:
            app.PDBFile.writeFile(modeller.getTopology(), modeller.getPositions(), file=ofs)

        with pytest.raises(AssertionError, match="Waters in PDB must be at the end of the file"):
            build_protein_system(temp.name, DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF)


def test_build_protein_system():
    rng = np.random.default_rng(2024)
    mol_a, mol_b, _ = get_hif2a_ligand_pair_single_topology()

    with path_to_internal_file("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as pdb_path:
        host_pdbfile = str(pdb_path)
    host_config = build_protein_system(host_pdbfile, DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF)
    num_host_atoms = host_config.conf.shape[0] - host_config.num_water_atoms

    host_with_mols_config = build_protein_system(
        host_pdbfile, DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF, mols=[mol_a, mol_b]
    )
    num_host_atoms_with_mol = host_with_mols_config.conf.shape[0] - host_with_mols_config.num_water_atoms

    assert num_host_atoms == num_host_atoms_with_mol
    # Waters won't be deleted since the pocket has no waters
    assert host_config.num_water_atoms == host_with_mols_config.num_water_atoms
    np.testing.assert_equal(compute_box_volume(host_config.box), compute_box_volume(host_with_mols_config.box))

    host_config.box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
    host_with_mols_config.box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes

    for bp in host_config.host_system.get_U_fns():
        (
            du_dx,
            _,
        ) = bp.to_gpu(np.float32).bound_impl.execute(host_config.conf, host_config.box, compute_u=False)
        check_force_norm(-du_dx)

    for bp in host_with_mols_config.host_system.get_U_fns():
        (
            du_dx,
            _,
        ) = bp.to_gpu(np.float32).bound_impl.execute(
            host_with_mols_config.conf, host_with_mols_config.box, compute_u=False
        )
        check_force_norm(-du_dx)

    # Pick a random water atom, will center the ligands on the atom and verify that the box is slightly
    # larger
    water_atom_idx = rng.choice(host_with_mols_config.num_water_atoms)
    new_ligand_center = host_with_mols_config.conf[num_host_atoms_with_mol + water_atom_idx]
    for mol in [mol_a, mol_b]:
        conf = get_romol_conf(mol)
        centroid = np.mean(conf, axis=0)
        conf = conf - centroid + new_ligand_center
        set_romol_conf(mol, conf)

    moved_host_config = build_protein_system(host_pdbfile, DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF, mols=[mol_a, mol_b])
    assert moved_host_config.num_water_atoms == host_with_mols_config.num_water_atoms
    host_atoms_with_moved_ligands = moved_host_config.conf.shape[0] - moved_host_config.num_water_atoms
    assert num_host_atoms == host_atoms_with_moved_ligands
    assert compute_box_volume(host_config.box) < compute_box_volume(moved_host_config.box)
