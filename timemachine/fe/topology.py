from typing import List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from timemachine import potentials
from timemachine.constants import DEFAULT_CHIRAL_ATOM_RESTRAINT_K, DEFAULT_CHIRAL_BOND_RESTRAINT_K
from timemachine.fe import chiral_utils
from timemachine.fe.system import VacuumSystem
from timemachine.fe.utils import get_romol_conf
from timemachine.ff.handlers import nonbonded
from timemachine.potentials.nonbonded import combining_rule_epsilon, combining_rule_sigma
from timemachine.potentials.types import Params

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
    def __init__(self, host_potentials, guest_topology, num_water_atoms: int):
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
            if isinstance(bp.potential, potentials.HarmonicBond):
                assert self.host_harmonic_bond is None
                self.host_harmonic_bond = bp
            elif isinstance(bp.potential, potentials.HarmonicAngle):
                assert self.host_harmonic_angle is None
                self.host_harmonic_angle = bp
            elif isinstance(bp.potential, potentials.PeriodicTorsion):
                assert self.host_periodic_torsion is None
                self.host_periodic_torsion = bp
            elif isinstance(bp.potential, potentials.Nonbonded):
                assert self.host_nonbonded is None
                self.host_nonbonded = bp
            else:
                raise UnsupportedPotential("Unsupported host potential")

        assert self.host_nonbonded is not None
        self.num_host_atoms = self.host_nonbonded.potential.num_atoms
        self.num_water_atoms = num_water_atoms
        self.num_other_atoms = self.num_host_atoms - num_water_atoms

    def get_water_idxs(self) -> NDArray:
        return np.arange(self.num_water_atoms, dtype=np.int32) + self.num_other_atoms

    def get_other_idxs(self) -> NDArray:
        return np.arange(self.num_other_atoms, dtype=np.int32)

    def get_num_atoms(self) -> int:
        return self.num_host_atoms + self.guest_topology.get_num_atoms()

    def get_lig_idxs(self) -> NDArray:
        def to_np(a):
            return np.concatenate([np.array(v, dtype=np.int32) for v in a])

        if self.num_host_atoms:
            return to_np(self.get_component_idxs()[1:])
        else:
            return to_np(self.get_component_idxs())

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
            assert isinstance(host_potential.potential, type(guest_potential))

        guest_idxs = guest_potential.idxs + self.num_host_atoms

        # If the host has no parameters, treat it as empty to handle concatenation of empty lists
        if host_potential is not None and host_potential.params.size > 0:
            # the host is always on.
            host_params = host_potential.params
            host_idxs = host_potential.potential.idxs
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

    def parameterize_nonbonded(
        self,
        ff_q_params,
        ff_q_params_intra,
        ff_q_params_solv,
        ff_lj_params,
        ff_lj_params_intra,
        ff_lj_params_solv,
        lamb: float,
    ):
        num_guest_atoms = self.guest_topology.get_num_atoms()
        # ligand other interactions
        # NOTE: None is passed for the other params to indicate they are not used.
        guest_ixn_other_params, _ = self.guest_topology.parameterize_nonbonded(
            ff_q_params, None, None, ff_lj_params, None, None, lamb, intramol_params=False
        )

        # ligand water interactions
        # NOTE: We assume the handlers of the solvent terms are the same as the
        # non-solvent terms (so we can pass ff_q_params_solv in place of ff_q_params).
        # If this changes, this code will need to be updated.
        guest_ixn_water_params, _ = self.guest_topology.parameterize_nonbonded(
            ff_q_params_solv, None, None, ff_lj_params_solv, None, None, lamb, intramol_params=False
        )
        # ligand intramolecular interactions
        guest_intra_params, guest_intra_pot = self.guest_topology.parameterize_nonbonded_pairlist(
            None, ff_q_params_intra, None, ff_lj_params_intra, intramol_params=True
        )

        # shift idxs because of the host
        beta = guest_intra_pot.beta
        cutoff = guest_intra_pot.cutoff
        guest_intra_pot.idxs = guest_intra_pot.idxs + self.num_host_atoms
        assert guest_ixn_water_params.shape == (num_guest_atoms, 4)
        assert self.host_nonbonded is not None
        assert beta == self.host_nonbonded.potential.beta
        assert cutoff == self.host_nonbonded.potential.cutoff

        exclusion_idxs = self.host_nonbonded.potential.exclusion_idxs
        scale_factors = self.host_nonbonded.potential.scale_factors

        # Note: The choice of zeros here is arbitrary. It doesn't affect the
        # potentials or grads, but anything that looks at the parameters
        # (such as hashing for the seed) could depend on these values
        hg_nb_params = jnp.concatenate([self.host_nonbonded.params, np.zeros(guest_ixn_water_params.shape)])

        host_guest_pot = potentials.Nonbonded(
            self.num_host_atoms + num_guest_atoms,
            exclusion_idxs,
            scale_factors,
            beta,
            cutoff,
            atom_idxs=np.arange(self.num_host_atoms, dtype=np.int32),
        )

        ixn_pots, ixn_params = get_ligand_ixn_pots_params(
            self.get_lig_idxs(),
            self.get_water_idxs(),
            self.get_other_idxs(),
            self.host_nonbonded.params,
            guest_ixn_water_params,
            guest_ixn_other_params,
            beta=beta,
            cutoff=cutoff,
        )

        hg_total_pot = [host_guest_pot] + ixn_pots
        hg_total_params = [hg_nb_params] + ixn_params

        # If the molecule has < 4 atoms there may not be any intramolecular terms
        # so they should be ignored here
        has_intra_terms = guest_intra_params.shape[0] > 0
        if has_intra_terms:
            hg_total_pot += [guest_intra_pot]
            hg_total_params += [guest_intra_params]

        # SummedPotential requires flattened params
        sum_pot = potentials.SummedPotential(hg_total_pot, hg_total_params)
        sum_params = jnp.concatenate(hg_total_params).reshape((-1,))

        return sum_params, sum_pot


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

    def parameterize_nonbonded(
        self,
        ff_q_params,
        ff_q_params_intra,
        ff_q_params_solv,
        ff_lj_params,
        ff_lj_params_intra,
        ff_lj_params_solv,
        lamb: float,
        intramol_params=True,
    ):
        if intramol_params:
            q_params = self.ff.q_handle_intra.partial_parameterize(ff_q_params_intra, self.mol)
            lj_params = self.ff.lj_handle_intra.partial_parameterize(ff_lj_params_intra, self.mol)
        else:
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

    def parameterize_nonbonded_pairlist(
        self, ff_q_params, ff_q_params_intra, ff_lj_params, ff_lj_params_intra, intramol_params=True
    ):
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

        if intramol_params:
            q_params = self.ff.q_handle_intra.partial_parameterize(ff_q_params_intra, self.mol)
            lj_params = self.ff.lj_handle_intra.partial_parameterize(ff_lj_params_intra, self.mol)
        else:
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
        for q, sig, eps, (sf_q, sf_lj) in zip(q_ij, sig_ij, eps_ij, rescale_mask):
            params.append(
                (
                    q * sf_q,
                    sig,
                    eps * sf_lj,
                    0.0,  # w offset for intramolecular term
                )
            )
        params = np.array(params)

        # corner case for molecule without nb terms (everything excluded)
        if params.shape[0] == 0:
            params = np.reshape(params, (0, 4))

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
        combined_idxs = np.concatenate([proper_potential.idxs, improper_potential.idxs])
        combined_potential = potentials.PeriodicTorsion(combined_idxs)
        return combined_params, combined_potential

    def setup_chiral_restraints(self, chiral_atom_restraint_k, chiral_bond_restraint_k):
        """
        Create chiral atom and bond potentials.

        Parameters
        ----------
        chiral_atom_restraint_k: float
            Force constant of the restraints

        chiral_bond_restraint_k: float
            Force constant of the restraints

        Returns
        -------
        2-tuple
            Returns a ChiralAtomRestraint and a ChiralBondRestraint

        """
        mol = self.mol
        conf = get_romol_conf(mol)

        # chiral atoms
        chiral_atom_restr_idxs = np.array(chiral_utils.setup_all_chiral_atom_restr_idxs(mol, conf))

        chiral_atom_params = chiral_atom_restraint_k * np.ones(len(chiral_atom_restr_idxs))
        assert len(chiral_atom_params) == len(chiral_atom_restr_idxs)  # TODO: can this be checked in Potential::bind ?
        chiral_atom_potential = potentials.ChiralAtomRestraint(chiral_atom_restr_idxs).bind(chiral_atom_params)

        # chiral bonds
        chiral_bonds = chiral_utils.find_chiral_bonds(mol)
        chiral_bond_restr_idxs = []
        chiral_bond_restr_signs = []
        chiral_bond_params = []
        for src_idx, dst_idx in chiral_bonds:
            idxs, signs = chiral_utils.setup_chiral_bond_restraints(mol, conf, src_idx, dst_idx)
            for ii in idxs:
                assert ii not in chiral_bond_restr_idxs
            chiral_bond_restr_idxs.extend(idxs)
            chiral_bond_restr_signs.extend(signs)
            chiral_bond_params.extend(chiral_bond_restraint_k for _ in idxs)  # TODO: double-check this

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
        chiral_atom_potential, chiral_bond_potential = self.setup_chiral_restraints(
            chiral_atom_restraint_k=DEFAULT_CHIRAL_ATOM_RESTRAINT_K,
            chiral_bond_restraint_k=DEFAULT_CHIRAL_BOND_RESTRAINT_K,
        )
        system.chiral_atom = chiral_atom_potential
        system.chiral_bond = chiral_bond_potential
        return system

    def setup_end_state(self):
        mol_bond_params, mol_hb = self.parameterize_harmonic_bond(self.ff.hb_handle.params)
        mol_angle_params, mol_ha = self.parameterize_harmonic_angle(self.ff.ha_handle.params)
        mol_proper_params, mol_pt = self.parameterize_proper_torsion(self.ff.pt_handle.params)
        mol_improper_params, mol_it = self.parameterize_improper_torsion(self.ff.it_handle.params)
        mol_nbpl_params, mol_nbpl = self.parameterize_nonbonded_pairlist(
            self.ff.q_handle.params,
            self.ff.q_handle_intra.params,
            self.ff.lj_handle.params,
            self.ff.lj_handle_intra.params,
            intramol_params=True,
        )
        bond_potential = mol_hb.bind(mol_bond_params)
        angle_potential = mol_ha.bind(mol_angle_params)

        torsion_params = np.concatenate([mol_proper_params, mol_improper_params])
        torsion_idxs = np.concatenate([mol_pt.idxs, mol_it.idxs])
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

    def parameterize_nonbonded(
        self,
        ff_q_params,
        ff_q_params_intra,
        ff_q_params_solv,
        ff_lj_params,
        ff_lj_params_intra,
        ff_lj_params_solv,
        lamb: float,
        intramol_params=True,
    ):
        # NOTE: lamb is unused here, but is used by the subclass DualTopologyMinimization
        del lamb

        # dummy is either "a or "b"
        if intramol_params:
            q_params_a = self.ff.q_handle_intra.partial_parameterize(ff_q_params_intra, self.mol_a)
            q_params_b = self.ff.q_handle_intra.partial_parameterize(ff_q_params_intra, self.mol_b)
            lj_params_a = self.ff.lj_handle_intra.partial_parameterize(ff_lj_params_intra, self.mol_a)
            lj_params_b = self.ff.lj_handle_intra.partial_parameterize(ff_lj_params_intra, self.mol_b)
        else:
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

    def parameterize_nonbonded_pairlist(
        self, ff_q_params, ff_q_params_intra, ff_lj_params, ff_lj_params_intra, intramol_params=True
    ):
        """
        Generate intramolecular nonbonded pairlist, and is mostly identical to the above
        except implemented as a pairlist.
        """
        NA = self.mol_a.GetNumAtoms()

        params_a, pairlist_a = BaseTopology(self.mol_a, self.ff).parameterize_nonbonded_pairlist(
            ff_q_params, ff_q_params_intra, ff_lj_params, ff_lj_params_intra, intramol_params=intramol_params
        )
        params_b, pairlist_b = BaseTopology(self.mol_b, self.ff).parameterize_nonbonded_pairlist(
            ff_q_params, ff_q_params_intra, ff_lj_params, ff_lj_params_intra, intramol_params=intramol_params
        )

        params = np.concatenate([params_a, params_b])

        inclusions_a = pairlist_a.idxs
        inclusions_b = pairlist_b.idxs
        inclusions_b += NA
        inclusion_idxs = np.concatenate([inclusions_a, inclusions_b])

        assert pairlist_a.beta == pairlist_b.beta
        assert pairlist_a.cutoff == pairlist_b.cutoff

        return params, potentials.NonbondedPairListPrecomputed(inclusion_idxs, pairlist_a.beta, pairlist_a.cutoff)

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
        combined_idxs = np.concatenate([proper_potential.idxs, improper_potential.idxs])

        combined_potential = potentials.PeriodicTorsion(combined_idxs)
        return combined_params, combined_potential

    def parameterize_proper_torsion(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.pt_handle, potentials.PeriodicTorsion)

    def parameterize_improper_torsion(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.it_handle, potentials.PeriodicTorsion)


class DualTopologyMinimization(DualTopology):
    def parameterize_nonbonded(
        self,
        ff_q_params,
        ff_q_params_intra,
        ff_q_params_solv,
        ff_lj_params,
        ff_lj_params_intra,
        ff_lj_params_solv,
        lamb: float,
        intramol_params=True,
    ):
        # both mol_a and mol_b are standardized.
        # we don't actually need derivatives for this stage.

        params, nb_potential = super().parameterize_nonbonded(
            ff_q_params,
            ff_q_params_intra,
            ff_q_params_solv,
            ff_lj_params,
            ff_lj_params_intra,
            ff_lj_params_solv,
            lamb,
            intramol_params=intramol_params,
        )
        cutoff = nb_potential.cutoff
        params_with_offsets = jnp.asarray(params).at[:, 3].set(lamb * cutoff)

        return params_with_offsets, nb_potential


def exclude_all_ligand_ligand_ixns(num_host_atoms: int, num_guest_atoms: int) -> Tuple[NDArray, NDArray]:
    """
    Return a tuple of the ligand exclusions and scale factors which exclude
    all ligand-ligand interactions. This is done to mask out these interactions
    so they can be calculated using the pairlist.
    """
    guest_exclusions_ = []
    guest_scale_factors_ = []

    for i in range(num_guest_atoms):
        for j in range(i + 1, num_guest_atoms):
            guest_exclusions_.append((i, j))
            guest_scale_factors_.append((1.0, 1.0))

    guest_exclusions = np.array(guest_exclusions_, dtype=np.int32) + num_host_atoms
    guest_scale_factors = np.array(guest_scale_factors_, dtype=np.float64)
    return guest_exclusions, guest_scale_factors


def get_ligand_ixn_pots_params(
    lig_idxs: NDArray,
    water_idxs: NDArray,
    other_idxs: Optional[NDArray],
    host_nb_params: Params,
    guest_params_ixn_water: Params,
    guest_params_ixn_other: Params,
    beta=2.0,
    cutoff=1.2,
):
    """
    Return the interaction group potentials and corresponding parameters
    for the ligand-water and ligand-protein interaction terms.

    Parameters
    ----------
    lig_idxs_list:
        List of ligand indexes (dtype, np.int32), one for each ligand.

    water_idxs:
        Indexes for the water atoms.

    other_idxs:
        Indexes for the other environment atoms that are not waters.
        May be None if there are no other atoms.

    host_nb_params:
        Nonbonded parameters for the host (environment) atoms.

    guest_params_ixn_water:
        Parameters for the guest (ligand) NB interactions with water.

    guest_params_ixn_other:
        Parameters for the guest (ligand) NB interactions with the
        non-water environment atoms.
    """

    # Init
    other_idxs = other_idxs if other_idxs is not None else np.array([])

    # Ligand-Water terms
    num_lig_atoms = len(lig_idxs)
    num_total_atoms = num_lig_atoms + len(water_idxs) + len(other_idxs)

    hg_water_pot = potentials.NonbondedInteractionGroup(
        num_total_atoms,
        lig_idxs,
        beta,
        cutoff,
        col_atom_idxs=water_idxs,
    )
    hg_water_params = jnp.concatenate([host_nb_params, guest_params_ixn_water])

    # Ligand-Other (protein) terms
    hg_other_pots = []
    hg_other_params = []
    if len(other_idxs):
        hg_other_pots.append(
            potentials.NonbondedInteractionGroup(
                num_total_atoms,
                lig_idxs,
                beta,
                cutoff,
                col_atom_idxs=other_idxs,
            )
        )
        hg_other_params.append(jnp.concatenate([host_nb_params, guest_params_ixn_other]))

    # If the ordering of these parameters change, will need to change
    # timemachine.fe.free_energy::get_water_sampler_params
    # total potential = host_guest_pot + guest_intra_pot + lw_ixn_pots + lp_ixn_pots
    hg_ixn_pots = [hg_water_pot] + hg_other_pots
    hg_ixn_params = [hg_water_params] + hg_other_params

    return hg_ixn_pots, hg_ixn_params
