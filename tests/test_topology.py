from importlib import resources

import numpy as np

from timemachine.fe import topology
from timemachine.fe.utils import get_romol_conf, read_sdf
from timemachine.ff import Forcefield


def test_dual_topology_nonbonded_pairlist():
    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        all_mols = read_sdf(path_to_ligand)

    mol_a = all_mols[1]
    mol_b = all_mols[4]
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    dt = topology.DualTopology(mol_a, mol_b, ff)

    nb_params, nb = dt.parameterize_nonbonded(ff.q_handle.params, ff.lj_handle.params, 0.0)

    nb_pairlist_params, nb_pairlist = dt.parameterize_nonbonded_pairlist(ff.q_handle.params, ff.lj_handle.params)

    x0 = np.concatenate([get_romol_conf(mol_a), get_romol_conf(mol_b)])
    box = np.eye(3) * 4.0

    for precision, rtol, atol in [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)]:

        nb_unbound = nb.unbound_impl(precision)
        nb_pairlist_unbound = nb_pairlist.unbound_impl(precision)

        du_dx, du_dp, u = nb_unbound.execute(x0, nb_params, box)

        pairlist_du_dx, pairlist_du_dp, pairlist_u = nb_pairlist_unbound.execute(x0, nb_pairlist_params, box)

        np.testing.assert_allclose(du_dx, pairlist_du_dx, atol=atol, rtol=rtol)

        # Different parameters, and so no expectation of shapes agreeing
        assert du_dp.shape != pairlist_du_dp.shape

        np.testing.assert_allclose(u, pairlist_u, atol=atol, rtol=rtol)
