from importlib import resources

import numpy as np
from rdkit import Chem

from timemachine.constants import DEFAULT_FF
from timemachine.fe.utils import to_md_units
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.md import builders, minimizer


def test_minimizer():
    ff = Forcefield.load_from_file(DEFAULT_FF)

    with resources.path("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as path_to_pdb:
        complex_system, complex_coords, _, _, complex_box, _ = builders.build_protein_system(str(path_to_pdb), ff)

    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        suppl = Chem.SDMolSupplier(str(path_to_ligand), removeHs=False)

    all_mols = [x for x in suppl]
    mol_a = all_mols[1]
    mol_b = all_mols[4]

    # these methods will throw if the minimization failed
    minimizer.minimize_host_4d([mol_a, mol_b], complex_system, complex_coords, ff, complex_box)
    minimizer.minimize_host_4d([mol_a], complex_system, complex_coords, ff, complex_box)
    minimizer.minimize_host_4d([mol_b], complex_system, complex_coords, ff, complex_box)


def test_equilibrate_host():
    ff = Forcefield.load_from_file(DEFAULT_FF)
    host_system, host_coords, host_box, _ = builders.build_water_system(4.0, ff)

    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        suppl = Chem.SDMolSupplier(str(path_to_ligand), removeHs=False)

    mol = next(suppl)

    coords, box = minimizer.equilibrate_host(mol, host_system, host_coords, 300, 1.0, ff, host_box, 25, seed=2022)
    assert coords.shape[0] == host_coords.shape[0] + mol.GetNumAtoms()
    assert coords.shape[1] == host_coords.shape[1]
    assert box.shape == host_box.shape


def test_local_minimize_water_box():
    """
    Test that we can locally relax a box of water by selecting some random indices.
    """
    ff = Forcefield.load_from_file(DEFAULT_FF)

    system, x0, box0, _ = builders.build_water_system(4.0, ff)
    x0 = to_md_units(x0)
    lamb = 0.0
    bps, _ = openmm_deserializer.deserialize_system(system, cutoff=1.2)
    box0 += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes at the boundary

    val_and_grad_fn = minimizer.get_val_and_grad_fn(bps, box0, lamb)

    free_idxs = [0, 2, 3, 6, 7, 9, 15, 16]
    frozen_idxs = set(range(len(x0))).difference(set(free_idxs))
    frozen_idxs = list(frozen_idxs)

    x_opt = minimizer.local_minimize(x0, val_and_grad_fn, free_idxs)

    np.testing.assert_array_equal(x0[frozen_idxs], x_opt[frozen_idxs])
    assert np.linalg.norm(x0[free_idxs] - x_opt[free_idxs]) > 0.01
