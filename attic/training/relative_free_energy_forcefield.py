import jax.numpy as jnp
import numpy as np

from timemachine.fe.topology import BaseTopology


class RelativeFreeEnergyForcefield(BaseTopology):
    """
    Used to run the same molecule under different forcefields.
    Currently only changing the nonbonded parameters is supported.

    The parameterize_* methods should be passed a list of parameters
    corresponding to forcefield0 and forcefield1.
    """

    def __init__(self, mol, forcefield0, forcefield1):
        """
        Utility for working with a single ligand.

        Parameter
        ---------
        mol: Chem.Mol
            Ligand to be parameterized

        forcefield0: ff.Forcefield
            A convenience wrapper for forcefield lists.

        forcefield1: ff.Forcefield
            A convenience wrapper for forcefield lists.

        """
        self.mol = mol
        self.ff = forcefield0
        self.ff1 = forcefield1
        self.bt1 = BaseTopology(mol, forcefield1)

    def parameterize_nonbonded(self, ff_q_params, ff_lj_params):
        assert self.ff.lj_handle.smirks == self.ff1.lj_handle.smirks, "changing lj smirks is not supported"
        assert self.ff.q_handle.smirks == self.ff1.q_handle.smirks, "changing charge smirks is not supported"
        src_qlj_params, nb_potential = super().parameterize_nonbonded(ff_q_params[0], ff_lj_params[0])
        dst_qlj_params, _ = self.bt1.parameterize_nonbonded(ff_q_params[1], ff_lj_params[1])

        combined_qlj_params = jnp.concatenate([src_qlj_params, dst_qlj_params])
        lambda_plane_idxs = np.zeros(self.mol.GetNumAtoms(), dtype=np.int32)
        lambda_offset_idxs = np.zeros(self.mol.GetNumAtoms(), dtype=np.int32)

        interpolated_potential = nb_potential.interpolate()
        interpolated_potential.set_lambda_plane_idxs(lambda_plane_idxs)
        interpolated_potential.set_lambda_offset_idxs(lambda_offset_idxs)

        return combined_qlj_params, interpolated_potential

    def parameterize_harmonic_bond(self, ff_params):
        assert np.allclose(ff_params[0], ff_params[1]), "changing harmonic bond parameters is not supported"
        assert self.ff.hb_handle.smirks == self.ff1.hb_handle.smirks, "changing harmonic bond smirks is not supported"
        return super().parameterize_harmonic_bond(ff_params[0])

    def parameterize_harmonic_angle(self, ff_params):
        assert np.allclose(ff_params[0], ff_params[1]), "changing harmonic angle parameters is not supported"
        assert self.ff.ha_handle.smirks == self.ff1.ha_handle.smirks, "changing harmonic angle smirks is not supported"
        return super().parameterize_harmonic_angle(ff_params[0])

    def parameterize_periodic_torsion(self, proper_params, improper_params):
        assert np.allclose(proper_params[0], proper_params[1]), "changing proper torsion parameters is not supported"
        assert self.ff.pt_handle.smirks == self.ff1.pt_handle.smirks, "changing proper torsion smirks is not supported"
        assert np.allclose(
            improper_params[0], improper_params[1]
        ), "changing improper torsion parameters is not supported"
        assert (
            self.ff.it_handle.smirks == self.ff1.it_handle.smirks
        ), "changing improper torsion smirks is not supported"
        return super().parameterize_periodic_torsion(proper_params[0], improper_params[0])
