from typing import List

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from timemachine.fe import chiral_utils
from timemachine.fe.system import VacuumSystem
from timemachine.fe.utils import get_romol_conf
from timemachine.ff.handlers import nonbonded
from timemachine.lib import potentials
from timemachine.potentials.nonbonded import combining_rule_epsilon, combining_rule_sigma

_SCALE_12 = 1.0
_SCALE_13 = 1.0
_SCALE_14 = 0.5
_BETA = 2.0
_CUTOFF = 1.2


class AtomMappingError(Exception):
    pass


class UnsupportedPotential(Exception):
    pass


class HostGuestTopology:
    def __init__(self, host_potentials, guest_topology):
        """
        Utility tool for combining host with a guest, in that order. host_potentials must be comprised
        exclusively of supported potentials (currently: bonds, angles, torsions, nonbonded).

        Parameters
        ----------
        host_potentials:
            Bound potentials for the host.

        guest_topology:
            Guest's Topology {Base, Dual, Single}Topology.

        """
        self.guest_topology = guest_topology

        self.host_nonbonded = None
        self.host_harmonic_bond = None
        self.host_harmonic_angle = None
        self.host_periodic_torsion = None

        # (ytz): extra assertions inside are to ensure we don't have duplicate terms
        for bp in host_potentials:
            if isinstance(bp, potentials.HarmonicBond):
                assert self.host_harmonic_bond is None
                self.host_harmonic_bond = bp
            elif isinstance(bp, potentials.HarmonicAngle):
                assert self.host_harmonic_angle is None
                self.host_harmonic_angle = bp
            elif isinstance(bp, potentials.PeriodicTorsion):
                assert self.host_periodic_torsion is None
                self.host_periodic_torsion = bp
            elif isinstance(bp, potentials.Nonbonded):
                assert self.host_nonbonded is None
                self.host_nonbonded = bp
            else:
                raise UnsupportedPotential("Unsupported host potential")

        assert self.host_nonbonded is not None
        self.num_host_atoms = self.host_nonbonded.get_num_atoms()

    def get_num_atoms(self):
        return self.num_host_atoms + self.guest_topology.get_num_atoms()

    def get_component_idxs(self) -> List[NDArray]:
        """
        Return the atom indices for each component of
        this topology as a list of NDArray. If the host is
        not present, this will just be the result from the guest topology.
        Otherwise, the result is in the order host atom idxs then guest
        component atom idxs.
        """
        host_idxs = [np.arange(self.num_host_atoms)] if self.num_host_atoms else []
        guest_idxs = [
            guest_component_idxs + self.num_host_atoms
            for guest_component_idxs in self.guest_topology.get_component_idxs()
        ]
        return host_idxs + guest_idxs

    # tbd: just merge the hamiltonians here
    def _parameterize_bonded_term(self, guest_params, guest_potential, host_potential):

        if guest_potential is None:
            raise UnsupportedPotential("Mismatch in guest_potential")

        # (ytz): corner case exists if the guest_potential is None
        if host_potential is not None:
            assert type(host_potential) == type(guest_potential)

        guest_idxs = guest_potential.get_idxs() + self.num_host_atoms

        if host_potential is not None:
            # the host is always on.
            host_params = host_potential.params
            host_idxs = host_potential.get_idxs()
        else:
            # (ytz): this extra jank is to work around jnp.concatenate not supporting empty lists.
            host_params = np.array([], dtype=guest_params.dtype).reshape((-1, guest_params.shape[1]))
            host_idxs = np.array([], dtype=guest_idxs.dtype).reshape((-1, guest_idxs.shape[1]))

        combined_params = jnp.concatenate([host_params, guest_params])
        combined_idxs = np.concatenate([host_idxs, guest_idxs])
        ctor = type(guest_potential)

        return combined_params, ctor(combined_idxs)

    def parameterize_harmonic_bond(self, ff_params):
        guest_params, guest_potential = self.guest_topology.parameterize_harmonic_bond(ff_params)
        return self._parameterize_bonded_term(guest_params, guest_potential, self.host_harmonic_bond)

    def parameterize_harmonic_angle(self, ff_params):
        guest_params, guest_potential = self.guest_topology.parameterize_harmonic_angle(ff_params)
        return self._parameterize_bonded_term(guest_params, guest_potential, self.host_harmonic_angle)

    def parameterize_periodic_torsion(self, proper_params, improper_params):
        guest_params, guest_potential = self.guest_topology.parameterize_periodic_torsion(
            proper_params, improper_params
        )
        return self._parameterize_bonded_term(guest_params, guest_potential, self.host_periodic_torsion)

    def parameterize_nonbonded(self, ff_q_params, ff_lj_params, lamb: float):
        num_guest_atoms = self.guest_topology.get_num_atoms()
        guest_params, guest_pot = self.guest_topology.parameterize_nonbonded(ff_q_params, ff_lj_params, lamb)

        assert guest_params.shape == (num_guest_atoms, 4)
        assert self.host_nonbonded is not None
        assert guest_pot.get_beta() == self.host_nonbonded.get_beta()
        assert guest_pot.get_cutoff() == self.host_nonbonded.get_cutoff()

        hg_exclusion_idxs = np.concatenate(
            [self.host_nonbonded.get_exclusion_idxs(), guest_pot.get_exclusion_idxs() + self.num_host_atoms]
        )
        hg_scale_factors = np.concatenate([self.host_nonbonded.get_scale_factors(), guest_pot.get_scale_factors()])

        hg_nb_params = jnp.concatenate([self.host_nonbonded.params, guest_params])

        return hg_nb_params, potentials.Nonbonded(
            self.num_host_atoms + num_guest_atoms,
            hg_exclusion_idxs,
            hg_scale_factors,
            guest_pot.get_beta(),
            guest_pot.get_cutoff(),
        )


class BaseTopology:
    def __init__(self, mol, forcefield):
        """
        Utility for working with a single ligand.

        Parameter
        ---------
        mol: ROMol
            Ligand to be parameterized

        forcefield: ff.Forcefield
            A convenience wrapper for forcefield lists.

        """
        self.mol = mol
        self.ff = forcefield

    def get_num_atoms(self):
        return self.mol.GetNumAtoms()

    def get_component_idxs(self) -> List[NDArray]:
        """
        Return the atom indices for the molecule in
        this topology as a list of NDArray.
        """
        return [np.arange(self.get_num_atoms())]

    def parameterize_nonbonded(self, ff_q_params, ff_lj_params, lamb: float):
        q_params = self.ff.q_handle.partial_parameterize(ff_q_params, self.mol)
        lj_params = self.ff.lj_handle.partial_parameterize(ff_lj_params, self.mol)

        exclusion_idxs, scale_factors = nonbonded.generate_exclusion_idxs(
            self.mol, scale12=_SCALE_12, scale13=_SCALE_13, scale14=_SCALE_14
        )

        scale_factors = np.stack([scale_factors, scale_factors], axis=1)

        beta = _BETA
        cutoff = _CUTOFF  # solve for this analytically later

        N = len(q_params)

        nb = potentials.Nonbonded(N, exclusion_idxs, scale_factors, beta, cutoff)

        w_coords = lamb * cutoff * jnp.ones((N, 1))
        params = jnp.concatenate([jnp.reshape(q_params, (-1, 1)), jnp.reshape(lj_params, (-1, 2)), w_coords], axis=1)

        return params, nb

    def parameterize_nonbonded_pairlist(self, ff_q_params, ff_lj_params):
        """
        Generate intramolecular nonbonded pairlist, and is mostly identical to the above
        except implemented as a pairlist.
        """
        # use same scale factors for electrostatics and vdWs
        exclusion_idxs, scale_factors = nonbonded.generate_exclusion_idxs(
            self.mol, scale12=_SCALE_12, scale13=_SCALE_13, scale14=_SCALE_14
        )

        # note: use same scale factor for electrostatics and vdw
        # typically in protein ffs, gaff, the 1-4 ixns use different scale factors between vdw and electrostatics
        exclusions_kv = dict()
        for (i, j), sf in zip(exclusion_idxs, scale_factors):
            assert i < j
            exclusions_kv[(i, j)] = sf

        # loop over all pairs
        inclusion_idxs, rescale_mask = [], []
        for i in range(self.mol.GetNumAtoms()):
            for j in range(i + 1, self.mol.GetNumAtoms()):
                scale_factor = exclusions_kv.get((i, j), 0.0)
                rescale_factor = 1 - scale_factor
                # keep this ixn
                if rescale_factor > 0:
                    rescale_mask.append([rescale_factor, rescale_factor])
                    inclusion_idxs.append([i, j])

        inclusion_idxs = np.array(inclusion_idxs).reshape(-1, 2).astype(np.int32)

        q_params = self.ff.q_handle.partial_parameterize(ff_q_params, self.mol)
        lj_params = self.ff.lj_handle.partial_parameterize(ff_lj_params, self.mol)

        sig_params = lj_params[:, 0]
        eps_params = lj_params[:, 1]

        l_idxs = inclusion_idxs[:, 0]
        r_idxs = inclusion_idxs[:, 1]

        q_ij = q_params[l_idxs] * q_params[r_idxs]
        sig_ij = combining_rule_sigma(sig_params[l_idxs], sig_params[r_idxs])
        eps_ij = combining_rule_epsilon(eps_params[l_idxs], eps_params[r_idxs])

        params = []
        for q, s, e, (sf_q, sf_lj) in zip(q_ij, sig_ij, eps_ij, rescale_mask):
            params.append(
                (
                    q * sf_q,
                    s,
                    e * sf_lj,
                    0.0,  # w offset
                )
            )

        params = np.array(params)

        beta = _BETA
        cutoff = _CUTOFF  # solve for this analytically later

        return params, potentials.NonbondedPairListPrecomputed(inclusion_idxs, beta, cutoff)

    def parameterize_harmonic_bond(self, ff_params):
        params, idxs = self.ff.hb_handle.partial_parameterize(ff_params, self.mol)
        return params, potentials.HarmonicBond(idxs)

    def parameterize_harmonic_angle(self, ff_params):
        params, idxs = self.ff.ha_handle.partial_parameterize(ff_params, self.mol)
        return params, potentials.HarmonicAngle(idxs)

    def parameterize_proper_torsion(self, ff_params):
        params, idxs = self.ff.pt_handle.partial_parameterize(ff_params, self.mol)
        return params, potentials.PeriodicTorsion(idxs)

    def parameterize_improper_torsion(self, ff_params):
        params, idxs = self.ff.it_handle.partial_parameterize(ff_params, self.mol)
        return params, potentials.PeriodicTorsion(idxs)

    def parameterize_periodic_torsion(self, proper_params, improper_params):
        """
        Parameterize all periodic torsions in the system.
        """
        proper_params, proper_potential = self.parameterize_proper_torsion(proper_params)
        improper_params, improper_potential = self.parameterize_improper_torsion(improper_params)
        combined_params = jnp.concatenate([proper_params, improper_params])
        combined_idxs = np.concatenate([proper_potential.get_idxs(), improper_potential.get_idxs()])
        combined_potential = potentials.PeriodicTorsion(combined_idxs)
        return combined_params, combined_potential

    # def setup_chiral_restraints(self, restraint_k=1000.0):
    def setup_chiral_restraints(self, restraint_k=1000.0):
        """
        Create chiral atom and bond potentials.

        Parameters
        ----------
        restraint_k: float
            Force constant of the restraints

        Returns
        -------
        2-tuple
            Returns a ChiralAtomRestraint and a ChiralBondRestraint

        """
        chiral_atoms = chiral_utils.find_chiral_atoms(self.mol)
        chiral_bonds = chiral_utils.find_chiral_bonds(self.mol)

        chiral_atom_restr_idxs = []
        chiral_atom_params = []
        for a_idx in chiral_atoms:
            idxs = chiral_utils.setup_chiral_atom_restraints(self.mol, get_romol_conf(self.mol), a_idx)
            for ii in idxs:
                assert ii not in chiral_atom_restr_idxs
            chiral_atom_restr_idxs.extend(idxs)
            chiral_atom_params.extend(restraint_k for _ in idxs)

        chiral_atom_params = np.array(chiral_atom_params)
        chiral_atom_restr_idxs = np.array(chiral_atom_restr_idxs)
        chiral_atom_potential = potentials.ChiralAtomRestraint(chiral_atom_restr_idxs).bind(chiral_atom_params)

        chiral_bond_restr_idxs = []
        chiral_bond_restr_signs = []
        chiral_bond_params = []
        for src_idx, dst_idx in chiral_bonds:
            idxs, signs = chiral_utils.setup_chiral_bond_restraints(
                self.mol, get_romol_conf(self.mol), src_idx, dst_idx
            )
            for ii in idxs:
                assert ii not in chiral_bond_restr_idxs
            chiral_bond_restr_idxs.extend(idxs)
            chiral_bond_restr_signs.extend(signs)
            chiral_bond_params.extend(restraint_k for _ in idxs)

        chiral_bond_restr_idxs = np.array(chiral_bond_restr_idxs)
        chiral_bond_restr_signs = np.array(chiral_bond_restr_signs)
        chiral_bond_params = np.array(chiral_bond_params)
        chiral_bond_potential = potentials.ChiralBondRestraint(chiral_bond_restr_idxs, chiral_bond_restr_signs).bind(
            chiral_bond_params
        )

        return chiral_atom_potential, chiral_bond_potential

    def setup_chiral_end_state(self):
        """
        Setup an end-state with chiral restraints attached.
        """
        system = self.setup_end_state()
        chiral_atom_potential, chiral_bond_potential = self.setup_chiral_restraints()
        system.chiral_atom = chiral_atom_potential
        system.chiral_bond = chiral_bond_potential
        return system

    def setup_end_state(self):
        mol_bond_params, mol_hb = self.parameterize_harmonic_bond(self.ff.hb_handle.params)
        mol_angle_params, mol_ha = self.parameterize_harmonic_angle(self.ff.ha_handle.params)
        mol_proper_params, mol_pt = self.parameterize_proper_torsion(self.ff.pt_handle.params)
        mol_improper_params, mol_it = self.parameterize_improper_torsion(self.ff.it_handle.params)
        mol_nbpl_params, mol_nbpl = self.parameterize_nonbonded_pairlist(
            self.ff.q_handle.params, self.ff.lj_handle.params
        )
        bond_potential = mol_hb.bind(mol_bond_params)
        angle_potential = mol_ha.bind(mol_angle_params)

        torsion_params = np.concatenate([mol_proper_params, mol_improper_params])
        torsion_idxs = np.concatenate([mol_pt.get_idxs(), mol_it.get_idxs()])
        torsion_potential = potentials.PeriodicTorsion(torsion_idxs).bind(torsion_params)
        nonbonded_potential = mol_nbpl.bind(mol_nbpl_params)

        system = VacuumSystem(bond_potential, angle_potential, torsion_potential, nonbonded_potential, None, None)

        return system


class DualTopology(BaseTopology):
    def __init__(self, mol_a, mol_b, forcefield):
        """
        Utility for working with two ligands via dual topology. Both copies of the ligand
        will be present after merging.

        Parameter
        ---------
        mol_a: ROMol
            First ligand to be parameterized

        mol_b: ROMol
            Second ligand to be parameterized

        forcefield: ff.Forcefield
            A convenience wrapper for forcefield lists.

        """
        self.mol_a = mol_a
        self.mol_b = mol_b
        self.ff = forcefield

    def get_num_atoms(self):
        return self.mol_a.GetNumAtoms() + self.mol_b.GetNumAtoms()

    def get_component_idxs(self) -> List[NDArray]:
        """
        Return the atom indices for the two ligands in
        this topology as a list of NDArray.
        """
        num_a_atoms = self.mol_a.GetNumAtoms()
        num_b_atoms = self.mol_b.GetNumAtoms()
        return [np.arange(num_a_atoms), num_a_atoms + np.arange(num_b_atoms)]

    def parameterize_nonbonded(self, ff_q_params, ff_lj_params, lamb: float):
        # NOTE: lamb is unused here, but is used by the subclass DualTopologyMinimization
        del lamb

        # dummy is either "a or "b"
        q_params_a = self.ff.q_handle.partial_parameterize(ff_q_params, self.mol_a)
        q_params_b = self.ff.q_handle.partial_parameterize(ff_q_params, self.mol_b)
        lj_params_a = self.ff.lj_handle.partial_parameterize(ff_lj_params, self.mol_a)
        lj_params_b = self.ff.lj_handle.partial_parameterize(ff_lj_params, self.mol_b)

        q_params = jnp.concatenate([q_params_a, q_params_b])
        lj_params = jnp.concatenate([lj_params_a, lj_params_b])

        exclusion_idxs_a, scale_factors_a = nonbonded.generate_exclusion_idxs(
            self.mol_a, scale12=_SCALE_12, scale13=_SCALE_13, scale14=_SCALE_14
        )

        exclusion_idxs_b, scale_factors_b = nonbonded.generate_exclusion_idxs(
            self.mol_b, scale12=_SCALE_12, scale13=_SCALE_13, scale14=_SCALE_14
        )

        mutual_exclusions_ = []
        mutual_scale_factors_ = []

        NA = self.mol_a.GetNumAtoms()
        NB = self.mol_b.GetNumAtoms()

        for i in range(NA):
            for j in range(NB):
                mutual_exclusions_.append([i, j + NA])
                mutual_scale_factors_.append([1.0, 1.0])

        mutual_exclusions = np.array(mutual_exclusions_)
        mutual_scale_factors = np.array(mutual_scale_factors_)

        combined_exclusion_idxs = np.concatenate([exclusion_idxs_a, exclusion_idxs_b + NA, mutual_exclusions]).astype(
            np.int32
        )

        combined_scale_factors = np.concatenate(
            [
                np.stack([scale_factors_a, scale_factors_a], axis=1),
                np.stack([scale_factors_b, scale_factors_b], axis=1),
                mutual_scale_factors,
            ]
        ).astype(np.float64)

        N = NA + NB
        w_coords = jnp.zeros((N, 1))

        beta = _BETA
        cutoff = _CUTOFF  # solve for this analytically later

        qlj_params = jnp.concatenate(
            [jnp.reshape(q_params, (-1, 1)), jnp.reshape(lj_params, (-1, 2)), w_coords], axis=1
        )

        return qlj_params, potentials.Nonbonded(
            N,
            combined_exclusion_idxs,
            combined_scale_factors,
            beta,
            cutoff,
        )

    def parameterize_nonbonded_pairlist(self, ff_q_params, ff_lj_params):
        """
        Generate intramolecular nonbonded pairlist, and is mostly identical to the above
        except implemented as a pairlist.
        """
        NA = self.mol_a.GetNumAtoms()

        params_a, pairlist_a = BaseTopology(self.mol_a, self.ff).parameterize_nonbonded_pairlist(
            ff_q_params, ff_lj_params
        )
        params_b, pairlist_b = BaseTopology(self.mol_b, self.ff).parameterize_nonbonded_pairlist(
            ff_q_params, ff_lj_params
        )

        params = np.concatenate([params_a, params_b])

        inclusions_a = pairlist_a.get_idxs()
        inclusions_b = pairlist_b.get_idxs()
        inclusions_b += NA
        inclusion_idxs = np.concatenate([inclusions_a, inclusions_b])

        assert pairlist_a.get_beta() == pairlist_b.get_beta()
        assert pairlist_a.get_cutoff() == pairlist_b.get_cutoff()

        return params, potentials.NonbondedPairListPrecomputed(
            inclusion_idxs, pairlist_a.get_beta(), pairlist_a.get_beta()
        )

    def _parameterize_bonded_term(self, ff_params, bonded_handle, potential):
        offset = self.mol_a.GetNumAtoms()
        params_a, idxs_a = bonded_handle.partial_parameterize(ff_params, self.mol_a)
        params_b, idxs_b = bonded_handle.partial_parameterize(ff_params, self.mol_b)
        params_c = jnp.concatenate([params_a, params_b])
        idxs_c = np.concatenate([idxs_a, idxs_b + offset])
        return params_c, potential(idxs_c)

    def parameterize_harmonic_bond(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.hb_handle, potentials.HarmonicBond)

    def parameterize_harmonic_angle(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.ha_handle, potentials.HarmonicAngle)

    def parameterize_periodic_torsion(self, proper_params, improper_params):
        """
        Parameterize all periodic torsions in the system.
        """
        proper_params, proper_potential = self.parameterize_proper_torsion(proper_params)
        improper_params, improper_potential = self.parameterize_improper_torsion(improper_params)

        combined_params = jnp.concatenate([proper_params, improper_params])
        combined_idxs = np.concatenate([proper_potential.get_idxs(), improper_potential.get_idxs()])

        combined_potential = potentials.PeriodicTorsion(combined_idxs)
        return combined_params, combined_potential

    def parameterize_proper_torsion(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.pt_handle, potentials.PeriodicTorsion)

    def parameterize_improper_torsion(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.it_handle, potentials.PeriodicTorsion)


class DualTopologyMinimization(DualTopology):
    def parameterize_nonbonded(self, ff_q_params, ff_lj_params, lamb: float):

        # both mol_a and mol_b are standardized.
        # we don't actually need derivatives for this stage.

        params, nb_potential = super().parameterize_nonbonded(ff_q_params, ff_lj_params, lamb)
        cutoff = nb_potential.get_cutoff()
        params_with_offsets = jnp.asarray(params).at[:, 3].set(lamb * cutoff)

        return params_with_offsets, nb_potential
