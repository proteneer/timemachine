import copy
import numpy as np
import functools

import jax
import jax.numpy as jnp

from ff.handlers import bonded, nonbonded, openmm_deserializer
from timemachine.lib import potentials

from rdkit import Chem

class Recipe():

    def __init__(self, masses, bound_potentials):
        """
        Recipes detail how to create a simulation system. They are similar to OpenMM's System
        class. Recipes can be converted from either rdkit ROMols or OpenMM System objects. Note
        that due to the use of vjp_fns and their closures, recipes are not easily serializable.
        You should only serialize the masses and bound_potentials.


        Parameters
        ----------
        masses: np.ndarray float64 [N]
            masses of each particle

        bound_potentials: [lib.potentials]
            list of potential energies that have been bound to parameteters

        """
        self.masses = masses
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

        for handle in ff_handlers:
            results = handle.parameterize(mol)
            if isinstance(handle, bonded.HarmonicBondHandler):
                bond_params, bond_idxs = results
                bound_potentials.append(potentials.HarmonicBond(bond_idxs).bind(bond_params))
            elif isinstance(handle, bonded.HarmonicAngleHandler):
                angle_params, angle_idxs = results
                bound_potentials.append(potentials.HarmonicAngle(angle_idxs).bind(angle_params))
            elif isinstance(handle, bonded.ProperTorsionHandler):
                torsion_params, torsion_idxs = results
                bound_potentials.append(potentials.PeriodicTorsion(torsion_idxs).bind(torsion_params))
            elif isinstance(handle, bonded.ImproperTorsionHandler):
                torsion_params, torsion_idxs = results
                bound_potentials.append(potentials.PeriodicTorsion(torsion_idxs).bind(torsion_params))
            elif isinstance(handle, nonbonded.AM1CCCHandler):
                charge_handle = handle
                charge_params = results
            elif isinstance(handle, nonbonded.LennardJonesHandler):
                lj_params = results
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

        qlj_params = jnp.concatenate([
            jnp.reshape(charge_params, (-1, 1)),
            jnp.reshape(lj_params, (-1, 2))
        ], axis=1)


        bound_potentials.append(potentials.Nonbonded(
            exclusion_idxs,
            scale_factors,
            lambda_plane_idxs,
            lambda_offset_idxs,
            alpha,
            cutoff).bind(qlj_params))

        return cls(masses, bound_potentials)

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

        return cls(masses, bound_potentials)

    def combine(self, other):
        """
        Combine two recipes together. self will keep its original indexing,
        while other will be incremented. This method automatically increments
        indices in the potential functions accordingly. For nonbonded terms, the recipe
        does a straight forward concatenation of the lambda idxs.

        Parameters
        ----------
        other: Recipe
            the right hand side recipe to combine with

        Returns
        -------
        Recipe
            combined recipe

        """
        self_num_atoms = len(self.masses)
        combined_masses = np.concatenate([self.masses, other.masses])
        combined_bound_potentials = []

        for bp in self.bound_potentials:
            if isinstance(bp, potentials.Nonbonded):
                # save these parameters for the merge part.
                self_nb_params = bp.params
                self_nb_exclusions = bp.get_exclusion_idxs()
                self_nb_scale_factors = bp.get_scale_factors()
                self_nb_cutoff = bp.get_cutoff()
                self_nb_beta = bp.get_beta()
                self_nb_lambda_plane_idxs = bp.get_lambda_plane_idxs()
                self_nb_lambda_offset_idxs = bp.get_lambda_offset_idxs()
            else:
                combined_bound_potentials.append(bp)


        for full_obp in other.bound_potentials:
            # always deepcopy to prevent modifying original copy
            full_obp = copy.deepcopy(full_obp)

            # if this is a lambda potential we replace only the .u_fn part
            if isinstance(full_obp, potentials.LambdaPotential):
                full_obp.set_N(full_obp.get_N() + self_num_atoms)
                obp = full_obp.get_u_fn()
            else:
                obp = full_obp

            if isinstance(obp, potentials.HarmonicBond) or isinstance(obp, potentials.CoreRestraint):
                idxs = obp.get_idxs()
                idxs += self_num_atoms # modify inplace
            elif isinstance(obp, potentials.HarmonicAngle):
                idxs = obp.get_idxs()
                idxs += self_num_atoms # modify inplace
            elif isinstance(obp, potentials.PeriodicTorsion):
                idxs = obp.get_idxs()
                idxs += self_num_atoms # modify inplace
            elif isinstance(obp, potentials.CentroidRestraint):
                a_idxs = obp.get_a_idxs()
                a_idxs += self_num_atoms # modify inplace
                b_idxs = obp.get_b_idxs()
                b_idxs += self_num_atoms # modify inplace
                # adjust masses
                obp.set_masses(np.concatenate([np.zeros(self_num_atoms), obp.get_masses()]))
            elif isinstance(obp, potentials.Nonbonded):
                assert self_nb_cutoff == obp.get_cutoff()

                assert self_nb_beta == obp.get_beta()

                combined_nb_params = jnp.concatenate([self_nb_params, obp.params])
                combined_exclusion_idxs = np.concatenate([self_nb_exclusions, obp.get_exclusion_idxs() + self_num_atoms])
                combined_scale_factors = np.concatenate([self_nb_scale_factors, obp.get_scale_factors()])
                combined_lambda_offset_idxs = np.concatenate([self_nb_lambda_offset_idxs, obp.get_lambda_offset_idxs()])
                combined_lambda_plane_idxs = np.concatenate([self_nb_lambda_plane_idxs, obp.get_lambda_plane_idxs()])

                # (ytz): leave this in for now
                # sanity check to ensure that the chain rules are working
                dummy = np.ones_like(combined_nb_params)

                obp = potentials.Nonbonded(
                    combined_exclusion_idxs,
                    combined_scale_factors,
                    combined_lambda_plane_idxs,
                    combined_lambda_offset_idxs,
                    self_nb_beta,
                    self_nb_cutoff).bind(combined_nb_params)

            else:
                raise Exception("Unknown functional form")

            if isinstance(full_obp, potentials.LambdaPotential):
                combined_bound_potentials.append(full_obp)
            else:
                combined_bound_potentials.append(obp)

        return Recipe(combined_masses, combined_bound_potentials)