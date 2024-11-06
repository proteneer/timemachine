import numpy as np

from timemachine.constants import DEFAULT_ATOM_MAPPING_KWARGS
from timemachine.fe import atom_mapping
from timemachine.fe.single_topology import SingleTopology
from timemachine.fe.utils import get_romol_conf, read_sdf_mols_by_name
from timemachine.ff import Forcefield

ff = Forcefield.load_default()


def get_vacuum_system_and_conf(mol_a, mol_b, core, lamb):
    st = SingleTopology(mol_a, mol_b, core, ff)
    conf_a = get_romol_conf(mol_a)
    conf_b = get_romol_conf(mol_b)
    conf = st.combine_confs(conf_a, conf_b, lamb)
    return st.setup_intermediate_state(lamb), conf


if __name__ == "__main__":
    mols_by_name = read_sdf_mols_by_name("timemachine/datasets/fep_benchmark/hif2a/ligands.sdf")
    mol_a = mols_by_name["338"]
    mol_b = mols_by_name["43"]
    core = atom_mapping.get_cores(mol_a, mol_b, **DEFAULT_ATOM_MAPPING_KWARGS)[0]
    lamb = 0.1
    sys_fwd, conf_fwd = get_vacuum_system_and_conf(mol_a, mol_b, core, lamb)
    sys_rev, conf_rev = get_vacuum_system_and_conf(mol_b, mol_a, core[:, ::-1], 1.0 - lamb)

    box = 100.0 * np.eye(3)

    assert sys_fwd.chiral_atom
    assert sys_rev.chiral_atom
    assert sys_fwd.torsion
    assert sys_rev.torsion

    # These pass
    np.testing.assert_allclose(sys_fwd.bond(conf_fwd, box), sys_rev.bond(conf_rev, box))
    np.testing.assert_allclose(sys_fwd.angle(conf_fwd, box), sys_rev.angle(conf_rev, box))
    np.testing.assert_allclose(sys_fwd.nonbonded(conf_fwd, box), sys_rev.nonbonded(conf_rev, box), atol=1e-4)
    np.testing.assert_allclose(sys_fwd.chiral_atom(conf_fwd, box), sys_rev.chiral_atom(conf_rev, box))

    # This fails
    np.testing.assert_allclose(sys_fwd.torsion(conf_fwd, box), sys_rev.torsion(conf_rev, box))
