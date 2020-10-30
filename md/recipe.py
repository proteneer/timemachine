import copy
import numpy as np
import functools

import jax
import jax.numpy as jnp

from ff.handlers import bonded, nonbonded, openmm_deserializer
from timemachine.lib import potentials

from rdkit import Chem

class Recipe():

    def __init__(self, masses, bound_potentials, vjp_fns):

        assert len(bound_potentials) == len(vjp_fns)

        self.masses = masses
        self.vjp_fns = vjp_fns
        self.bound_potentials = bound_potentials

    @classmethod
    def from_rdkit(cls, mol, ff_handlers):
        """
        Initialize a system from an RDKit ROMol. 
    
        Parameters
        ----------
        mol: Chem.ROMol
            RDKit ROMol. Should have graphical hydrogens in the topology.

        ff_handlers: list of forcefield handlers.
            openforcefield small molecule handlers.

        """
        masses = np.array([a.GetMass() for a in mol.GetAtoms()], dtype=np.float64)

        bound_potentials = []
        vjp_fns = []

        for handle in ff_handlers:
            results = handle.parameterize(mol)
            if isinstance(handle, bonded.HarmonicBondHandler):
                bond_idxs, (bond_params, vjp_fn) = results
                bound_potentials.append(potentials.HarmonicBond(bond_idxs).bind(bond_params))
                vjp_fns.append([(handle, vjp_fn)])
            elif isinstance(handle, bonded.HarmonicAngleHandler):
                angle_idxs, (angle_params, vjp_fn) = results
                bound_potentials.append(potentials.HarmonicAngle(angle_idxs).bind(angle_params))
                vjp_fns.append([(handle, vjp_fn)])
            elif isinstance(handle, bonded.ProperTorsionHandler):
                torsion_idxs, (torsion_params, vjp_fn) = results
                bound_potentials.append(potentials.PeriodicTorsion(torsion_idxs).bind(torsion_params))
                vjp_fns.append([(handle, vjp_fn)])
            elif isinstance(handle, bonded.ImproperTorsionHandler):
                torsion_idxs, (torsion_params, vjp_fn) = results
                bound_potentials.append(potentials.PeriodicTorsion(torsion_idxs).bind(torsion_params))
                vjp_fns.append([(handle, vjp_fn)])
            elif isinstance(handle, nonbonded.AM1CCCHandler):
                charge_handle = handle
                charge_params, charge_vjp_fn = results
            elif isinstance(handle, nonbonded.LennardJonesHandler):
                lj_params, lj_vjp_fn = results
                lj_handle = handle
            else:
                print("WARNING: skipping handler", handle)
                pass

        lambda_plane_idxs = np.zeros(len(masses), dtype=np.int32)
        lambda_offset_idxs = np.zeros(len(masses), dtype=np.int32)

        exclusion_idxs, scale_factors = nonbonded.generate_exclusion_idxs(
            mol,
            scale12=1.0,
            scale13=1.0,
            scale14=0.5
        )

        # use same scale factors until we modify 1-4s for electrostatics
        scale_factors = np.stack([scale_factors, scale_factors], axis=1)

        # (ytz) fix this later to not be so hard coded
        alpha = 2.0 # same as ewald alpha
        cutoff = 1.0 # nonbonded cutoff

        def combine_qlj(guest_q, guest_lj):
            return jnp.concatenate([
                jnp.reshape(guest_q, (-1, 1)),
                jnp.reshape(guest_lj, (-1, 2))
            ], axis=1)

        qlj_params, qlj_vjp_fn = jax.vjp(combine_qlj, charge_params, lj_params)

        def chain_q_fn(x):
            return charge_vjp_fn(qlj_vjp_fn(x)[0])

        def chain_lj_fn(x):
            return lj_vjp_fn(qlj_vjp_fn(x)[1])

        bound_potentials.append(potentials.Nonbonded(
            exclusion_idxs,
            scale_factors,
            lambda_plane_idxs,
            lambda_offset_idxs,
            alpha,
            cutoff).bind(qlj_params))

        vjp_fns.append([(charge_handle, chain_q_fn), (lj_handle, chain_lj_fn)])

        return cls(masses, bound_potentials, vjp_fns)

    @classmethod
    def from_openmm(cls, omm_system):
        """ Initialize a system from an OpenMM System.

        Parameters
        ----------
        omm_system: openm.System
            OpenMM system object

        """

        bound_potentials, masses = openmm_deserializer.deserialize_system(
            omm_system,
            cutoff=1.0
        )

        # add dummy vjp_fns
        vjp_fns = []
        for _ in bound_potentials:
            vjp_fns.append([])

        return cls(masses, bound_potentials, vjp_fns)

    # def combine(self, other, nb_lambda_offset, nb_lambda_plane, nb_exclude_other):
    def combine(self, other):
        """
        Combine two recipes together. self will keep its original indexing,
        while other will be incremented.

        other: Recipe

        # nb_exclude_other: boolean
            # Whether we want the two systems to be excluded.

        """
        self_num_atoms = len(self.masses)
        combined_masses = np.concatenate([self.masses, other.masses])
        combined_bound_potentials = []
        combined_vjp_fns = []

        for bp, vps in zip(self.bound_potentials, self.vjp_fns):
            if isinstance(bp, potentials.Nonbonded):
                self_nb_params = bp.params
                self_nb_exclusions = bp.get_exclusion_idxs()
                self_nb_scale_factors = bp.get_scale_factors()
                self_nb_cutoff = bp.get_cutoff()
                self_nb_beta = bp.get_beta()
                self_nb_lambda_plane_idxs = bp.get_lambda_plane_idxs()
                self_nb_lambda_offset_idxs = bp.get_lambda_offset_idxs()
                self_nb_vjp_fns = vps
            else:
                combined_bound_potentials.append(bp)
                combined_vjp_fns.append(vps)

        for obp, other_vjp_fns in zip(other.bound_potentials, other.vjp_fns):
            # deepcopy to prevent modifying original copy
            obp = copy.deepcopy(obp)
            if isinstance(obp, potentials.HarmonicBond):
                idxs = obp.get_bond_idxs()
                idxs += self_num_atoms # modify inplace
                combined_bound_potentials.append(obp)
                combined_vjp_fns.append(other_vjp_fns)
            elif isinstance(obp, potentials.HarmonicAngle):
                idxs = obp.get_angle_idxs()
                idxs += self_num_atoms # modify inplace
                combined_bound_potentials.append(obp)
                combined_vjp_fns.append(other_vjp_fns)
            elif isinstance(obp, potentials.PeriodicTorsion):
                idxs = obp.get_torsion_idxs()
                idxs += self_num_atoms # modify inplace
                combined_bound_potentials.append(obp)
                combined_vjp_fns.append(other_vjp_fns)
            elif isinstance(obp, potentials.CentroidRestraint):
                assert 0
            elif isinstance(obp, potentials.Nonbonded):
                assert self_nb_cutoff == obp.get_cutoff()

                assert self_nb_beta == obp.get_beta()

                combined_nb_params, combined_vjp_fn = jax.vjp(jnp.concatenate, [self_nb_params, obp.params])
                combined_exclusion_idxs = np.concatenate([self_nb_exclusions, obp.get_exclusion_idxs() + self_num_atoms])
                combined_scale_factors = np.concatenate([self_nb_scale_factors, obp.get_scale_factors()])
                combined_lambda_offset_idxs = np.concatenate([self_nb_lambda_offset_idxs, obp.get_lambda_offset_idxs()])
                combined_lambda_plane_idxs = np.concatenate([self_nb_lambda_plane_idxs, obp.get_lambda_plane_idxs()])

                # (ytz): leave this in for now
                # sanity check to ensure that the chain rules are working
                dummy = np.ones_like(combined_nb_params)

                total_vjp_fns = []

                def chain_former(x, vjp_fn):
                    former_adjoint, _ = combined_vjp_fn(x)[0]
                    return vjp_fn(former_adjoint)

                def chain_latter(x, vjp_fn):
                    _, latter_adjoint = combined_vjp_fn(x)[0]
                    return vjp_fn(latter_adjoint)

                for other_handle, other_vjp_fn in other_vjp_fns:
                    # (ytz): careful, closure rules can really screw with you
                    # if you define a function inside a loop
                    other_chain = functools.partial(chain_latter, vjp_fn=other_vjp_fn)
                    assert other_handle.params.shape == other_chain(dummy)[0].shape
                    total_vjp_fns.append((other_handle, other_chain))

                for self_handle, self_vjp_fn in self_nb_vjp_fns:
                    self_chain = functools.partial(chain_former, vjp_fn=self_vjp_fn)
                    assert self_handle.params.shape == self_chain(dummy)[0].shape
                    total_vjp_fns.append((self_handle, self_chain))

                combined_bound_potentials.append(potentials.Nonbonded(
                    combined_exclusion_idxs,
                    combined_scale_factors,
                    combined_lambda_plane_idxs,
                    combined_lambda_offset_idxs,
                    self_nb_beta,
                    self_nb_cutoff).bind(combined_nb_params)
                )
                combined_vjp_fns.append(total_vjp_fns)
            else:
                raise Exception("Unknown functional form")


        assert len(combined_bound_potentials) == len(combined_vjp_fns)

        return Recipe(combined_masses, combined_bound_potentials, combined_vjp_fns)