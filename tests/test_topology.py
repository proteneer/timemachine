from importlib import resources

import numpy as np
import pytest
from rdkit import Chem

from timemachine.constants import DEFAULT_FF
from timemachine.fe import topology
from timemachine.fe.utils import get_romol_conf
from timemachine.ff import Forcefield, combine_params


def test_relative_free_energy_forcefield():

    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        mol = next(Chem.SDMolSupplier(str(path_to_ligand), removeHs=False))

    ff0 = Forcefield.load_from_file(DEFAULT_FF)
    ff1 = Forcefield.load_from_file(DEFAULT_FF)

    # Modify the charge parameters for ff1
    ff1.q_handle.params += 1.0

    fftop = topology.RelativeFreeEnergyForcefield(mol, ff0, ff1)
    bt0 = topology.BaseTopology(mol, ff0)
    bt1 = topology.BaseTopology(mol, ff1)

    ff0_params = ff0.get_params()
    ff1_params = ff1.get_params()

    combined_params = combine_params(ff0_params, ff1_params)
    combined_qlj_params, combined_ubp = fftop.parameterize_nonbonded(
        combined_params.q_params, combined_params.lj_params
    )
    qlj0_params, ubp0 = bt0.parameterize_nonbonded(ff0_params.q_params, ff0_params.lj_params)
    qlj1_params, ubp1 = bt1.parameterize_nonbonded(ff1_params.q_params, ff1_params.lj_params)

    coords = get_romol_conf(mol)
    box = np.identity(3) * 99.0

    combined_impl = combined_ubp.bind(combined_qlj_params).bound_impl(precision=np.float32)
    _, _, u0_combined = combined_impl.execute(coords, box, lam=0)
    _, _, u1_combined = combined_impl.execute(coords, box, lam=1)

    u0_impl = ubp0.bind(qlj0_params).bound_impl(precision=np.float32)
    _, _, u0 = u0_impl.execute(coords, box, lam=0)

    u1_impl = ubp1.bind(qlj1_params).bound_impl(precision=np.float32)
    _, _, u1 = u1_impl.execute(coords, box, lam=0)  # lam=0 for the fully interacting state here

    # Check that the endstate NB energies are consistent
    assert pytest.approx(u0_combined) == u0
    assert pytest.approx(u1_combined) == u1

    # Check that other terms can not be changed
    fftop.parameterize_harmonic_bond(combined_params.hb_params)
    invalid = [ff0_params.hb_params, ff0_params.hb_params + 1.0]
    with pytest.raises(AssertionError, match="changing harmonic bond"):
        fftop.parameterize_harmonic_bond(invalid)

    fftop.parameterize_harmonic_angle(combined_params.ha_params)
    invalid = [ff0_params.ha_params, ff0_params.ha_params + 1.0]
    with pytest.raises(AssertionError, match="changing harmonic angle"):
        fftop.parameterize_harmonic_angle(invalid)

    fftop.parameterize_periodic_torsion(combined_params.pt_params, combined_params.it_params)
    invalid = [ff0_params.pt_params, ff0_params.pt_params + 1.0]
    with pytest.raises(AssertionError, match="changing proper"):
        fftop.parameterize_periodic_torsion(invalid, combined_params.it_params)

    invalid = [ff0_params.it_params, ff0_params.it_params + 1.0]
    with pytest.raises(AssertionError, match="changing improper"):
        fftop.parameterize_periodic_torsion(combined_params.pt_params, invalid)


def test_dual_topology_nonbonded_pairlist():
    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        suppl = Chem.SDMolSupplier(str(path_to_ligand), removeHs=False)

    all_mols = [x for x in suppl]
    mol_a = all_mols[1]
    mol_b = all_mols[4]
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    dt = topology.DualTopology(mol_a, mol_b, ff)

    nb_params, nb = dt.parameterize_nonbonded(ff.q_handle.params, ff.lj_handle.params)

    nb_pairlist_params, nb_pairlist = dt.parameterize_nonbonded_pairlist(ff.q_handle.params, ff.lj_handle.params)

    x0 = np.concatenate([get_romol_conf(mol_a), get_romol_conf(mol_b)])
    box = np.eye(3) * 4.0

    for precision, rtol, atol in [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)]:

        nb_unbound = nb.unbound_impl(precision)
        nb_pairlist_unbound = nb_pairlist.unbound_impl(precision)

        for lamb in [0.0, 1.0]:

            du_dx, du_dp, du_dl, u = nb_unbound.execute(x0, nb_params, box, lamb)

            pairlist_du_dx, pairlist_du_dp, pairlist_du_dl, pairlist_u = nb_pairlist_unbound.execute(
                x0, nb_pairlist_params, box, lamb
            )

            np.testing.assert_allclose(du_dx, pairlist_du_dx, atol=atol, rtol=rtol)

            # Different parameters, and so no expectation of shapes agreeing
            assert du_dp.shape != pairlist_du_dp.shape

            np.testing.assert_allclose(du_dl, pairlist_du_dl, atol=atol, rtol=rtol)
            np.testing.assert_allclose(u, pairlist_u, atol=atol, rtol=rtol)
