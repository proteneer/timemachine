from typing import List

import jax.numpy as jnp
import numpy as np

from timemachine.fe.topology import BaseTopology, DualTopology



class BaseTopologyConversion(BaseTopology):
    """
    Decharges the ligand and reduces the LJ epsilon by half.
    """

    def parameterize_nonbonded(self, ff_q_params, ff_lj_params):

        qlj_params, nb_potential = super().parameterize_nonbonded(ff_q_params, ff_lj_params)
        charge_indices = jnp.index_exp[:, 0]
        epsilon_indices = jnp.index_exp[:, 2]

        src_qlj_params = qlj_params
        dst_qlj_params = jnp.asarray(qlj_params).at[charge_indices].set(0.0)
        dst_qlj_params = dst_qlj_params.at[epsilon_indices].multiply(0.5)

        combined_qlj_params = jnp.concatenate([src_qlj_params, dst_qlj_params])
        lambda_plane_idxs = np.zeros(self.mol.GetNumAtoms(), dtype=np.int32)
        lambda_offset_idxs = np.zeros(self.mol.GetNumAtoms(), dtype=np.int32)

        interpolated_potential = nb_potential.interpolate()
        interpolated_potential.set_lambda_plane_idxs(lambda_plane_idxs)
        interpolated_potential.set_lambda_offset_idxs(lambda_offset_idxs)

        return combined_qlj_params, interpolated_potential

class BaseTopologyDecoupling(BaseTopology):
    """
    Decouple a ligand from the environment. The ligand has its charges set to zero
    and lennard jones epsilon halved.

    lambda=0 is the fully interacting state.
    lambda=1 is the non-interacting state.
    """

    def parameterize_nonbonded(self, ff_q_params, ff_lj_params):
        qlj_params, nb_potential = super().parameterize_nonbonded(ff_q_params, ff_lj_params)
        charge_indices = jnp.index_exp[:, 0]
        epsilon_indices = jnp.index_exp[:, 2]
        qlj_params = jnp.asarray(qlj_params).at[charge_indices].set(0.0)
        qlj_params = qlj_params.at[epsilon_indices].multiply(0.5)

        return qlj_params, nb_potential


class BaseTopologyRHFE(BaseTopology):
    pass


# non-ring torsions are just always turned off at the end-states in the hydration
# free energy test
class DualTopologyRHFE(DualTopology):

    """
    Utility class used for relative hydration free energies. Ligand B is decoupled as lambda goes
    from 0 to 1, while ligand A is fully coupled. At the same time, at lambda=0, ligand B and ligand A
    have their charges and epsilons reduced by half.
    """

    def parameterize_nonbonded(self, ff_q_params, ff_lj_params):

        qlj_params, nb_potential = super().parameterize_nonbonded(ff_q_params, ff_lj_params)

        # halve the strength of the charge and the epsilon parameters
        charge_indices = jnp.index_exp[:, 0]
        epsilon_indices = jnp.index_exp[:, 2]

        src_qlj_params = jnp.asarray(qlj_params).at[charge_indices].multiply(0.5)
        src_qlj_params = jnp.asarray(src_qlj_params).at[epsilon_indices].multiply(0.5)

        dst_qlj_params = qlj_params
        combined_qlj_params = jnp.concatenate([src_qlj_params, dst_qlj_params])

        combined_lambda_plane_idxs = np.zeros(self.mol_a.GetNumAtoms() + self.mol_b.GetNumAtoms(), dtype=np.int32)
        combined_lambda_offset_idxs = np.concatenate(
            [np.zeros(self.mol_a.GetNumAtoms(), dtype=np.int32), np.ones(self.mol_b.GetNumAtoms(), dtype=np.int32)]
        )

        nb_potential.set_lambda_plane_idxs(combined_lambda_plane_idxs)
        nb_potential.set_lambda_offset_idxs(combined_lambda_offset_idxs)

        return combined_qlj_params, nb_potential.interpolate()


class DualTopologyChargeConversion(DualTopology):
    """
    Let A and B be the two ligands of interest (typically both occupying the binding pocket)
    Assume that exclusions are already defined between atoms in A and atoms in B.

    Let this topology class compute the free energy associated with transferring the charge
    from molecule A onto molecule B.

    lambda=0: ligand A (charged), ligand B (decharged)
    lambda=1: ligand A (decharged), ligand B (charged)

    """

    def parameterize_nonbonded(self, ff_q_params, ff_lj_params):
        qlj_params_ab, nb_potential = super().parameterize_nonbonded(ff_q_params, ff_lj_params)
        num_a_atoms = self.mol_a.GetNumAtoms()
        charge_indices_b = jnp.index_exp[num_a_atoms:, 0]
        epsilon_indices_b = jnp.index_exp[num_a_atoms:, 2]

        charge_indices_a = jnp.index_exp[:num_a_atoms, 0]
        epsilon_indices_a = jnp.index_exp[:num_a_atoms, 2]

        qlj_params_src = qlj_params_ab.at[charge_indices_b].set(0.0)
        qlj_params_src = qlj_params_src.at[epsilon_indices_b].multiply(0.5)
        qlj_params_dst = qlj_params_ab.at[charge_indices_a].set(0.0)
        qlj_params_dst = qlj_params_dst.at[epsilon_indices_a].multiply(0.5)
        combined_qlj_params = jnp.concatenate([qlj_params_src, qlj_params_dst])
        interpolated_potential = nb_potential.interpolate()

        total_atoms = num_a_atoms + self.mol_b.GetNumAtoms()

        # probably already set to zeros by default
        combined_lambda_plane_idxs = np.zeros(total_atoms, dtype=np.int32)
        combined_lambda_offset_idxs = np.zeros(total_atoms, dtype=np.int32)
        interpolated_potential.set_lambda_plane_idxs(combined_lambda_plane_idxs)
        interpolated_potential.set_lambda_offset_idxs(combined_lambda_offset_idxs)

        return combined_qlj_params, interpolated_potential


class DualTopologyDecoupling(DualTopology):
    """
    Given two ligands A and B, let the end-states be:

    lambda=0 A is fully charged, and interacting with the environment.
             B is decharged, but "inserted"/interacting with the environment.

    lambda=1 A is fully charged, and interacting with the environment.
             B is decharged, and non-interacting with the environment.

    """

    def parameterize_nonbonded(self, ff_q_params, ff_lj_params):
        qlj_params_combined, nb_potential = super().parameterize_nonbonded(ff_q_params, ff_lj_params)
        num_a_atoms = self.mol_a.GetNumAtoms()
        charge_indices_b = jnp.index_exp[num_a_atoms:, 0]
        epsilon_indices_b = jnp.index_exp[num_a_atoms:, 2]

        qlj_params_combined = jnp.asarray(qlj_params_combined).at[charge_indices_b].set(0.0)
        qlj_params_combined = qlj_params_combined.at[epsilon_indices_b].multiply(0.5)

        num_b_atoms = self.mol_b.GetNumAtoms()
        combined_lambda_plane_idxs = np.zeros(num_a_atoms + num_b_atoms, dtype=np.int32)
        combined_lambda_offset_idxs = np.concatenate(
            [np.zeros(num_a_atoms, dtype=np.int32), np.ones(num_b_atoms, dtype=np.int32)]
        )
        nb_potential.set_lambda_plane_idxs(combined_lambda_plane_idxs)
        nb_potential.set_lambda_offset_idxs(combined_lambda_offset_idxs)

        return qlj_params_combined, nb_potential
