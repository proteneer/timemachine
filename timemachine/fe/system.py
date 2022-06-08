import functools

import jax.numpy as jnp
import numpy as np

from timemachine.potentials import bonded, nonbonded


class VacuumSystem:

    # utility system container

    def __init__(self, bond, angle, torsion, nonbonded):
        self.bond = bond
        self.angle = angle
        self.torsion = torsion
        self.nonbonded = nonbonded

    def get_U_fn(self):
        """
        Return a jax function that evaluates the potential energy of a set of coordinates.
        """
        bond_U = functools.partial(
            bonded.harmonic_bond,
            params=np.array(self.bond.params),
            box=None,
            lamb=0.0,
            bond_idxs=np.array(self.bond.get_idxs()),
        )
        angle_U = functools.partial(
            bonded.harmonic_angle,
            params=np.array(self.angle.params),
            box=None,
            lamb=0.0,
            angle_idxs=np.array(self.angle.get_idxs()),
        )
        torsion_U = functools.partial(
            bonded.periodic_torsion,
            params=np.array(self.torsion.params),
            box=None,
            lamb=0.0,
            torsion_idxs=np.array(self.torsion.get_idxs()),
        )

        nbpl_U = functools.partial(
            nonbonded.nonbonded_v3_on_specific_pairs,
            pairs=np.array(self.nonbonded.get_idxs()),
            params=np.array(self.nonbonded.params),
            box=None,
            beta=self.nonbonded.get_beta(),
            cutoff=self.nonbonded.get_cutoff(),
            rescale_mask=np.array(self.nonbonded.get_rescale_mask()),
        )

        def U_fn(x):
            Us_vdw, Us_coulomb = nbpl_U(x)
            return bond_U(x) + angle_U(x) + torsion_U(x) + jnp.sum(Us_vdw) + jnp.sum(Us_coulomb)

        return U_fn
