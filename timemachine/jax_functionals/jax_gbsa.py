import numpy as onp
import jax.numpy as np
from jax.scipy.special import erf, erfc

from timemachine.constants import ONE_4PI_EPS0
from timemachine.jax_functionals import Energy
from timemachine.jax_functionals.jax_utils import delta_r, distance

class GBSA(Energy):

    def __init__(self,
        param_idxs, 
        dielectric_offset=0.009,
        cutoff=2.0,
        alpha_obc=1.0,
        beta_obc=0.8,
        gamma_obc=4.85,
        solute_dielectric=1.0,
        solvent_dielectric=78.3,
        electric_constant=-69.467728,
        probe_radius=0.14,
        surface_area_energy=2.25936):
        """
        Implements the Generalized Born Surface Area implicit solvation model.
        
        Note that this is an O(N^2) as we don't use cutoffs.
        """
        self.param_idxs = param_idxs 

        self.dielectric_offset = dielectric_offset
        self.alpha_obc = alpha_obc
        self.beta_obc = beta_obc
        self.gamma_obc = gamma_obc
        self.solute_dielectric = solute_dielectric
        self.solvent_dielectric = solvent_dielectric
        self.electric_constant = electric_constant

        self.probe_radius = probe_radius
        self.pi4Asolv = 4*np.pi*surface_area_energy

    def non_polar_ace(self, born_radii, atomic_radii):
        """
        Compute the non-polar contribution to the GBSA energy.

        Parameters
        ----------
        born_radii: np.array
            shape [N,] effective compute born radii

        atomic_radii: np.array
            shape [N,] atomic radii

        Returns
        -------
        scalar
            non-polar energy

        """
        r = atomic_radii + self.probe_radius
        ratio6 = np.power(atomic_radii/born_radii, 6)
        sa_term = self.pi4Asolv*r*r*ratio6
        return np.sum(sa_term)

    def born_radii(self,
            conf,
            atomic_radii,
            scaled_radius_factor):
        """
        Compute the adjusted born radii of each atom. This is the first part of the GBSA calculation.

        Parameters
        ----------
        conf: np.array
            shape Nx3 matrix of geometric coordinates

        atomic_radii: np.array
            shape [N,] array of radius of each atom

        scaled_radius_factor: np.array
            shape [N,] array of adjusted shape factors for each atom.

        Returns
        -------
        np.array
            shape [N,] np.array of atomic radiis

        """

        num_atoms = conf.shape[0]
        dielectric_offset = self.dielectric_offset
        alpha_obc = self.alpha_obc
        beta_obc = self.beta_obc
        gamma_obc = self.gamma_obc

        r_i = np.expand_dims(conf, axis=0)
        r_j = np.expand_dims(conf, axis=1)
        d_ij = distance(r_i, r_j)

        oR = atomic_radii - dielectric_offset
        oRI = np.expand_dims(oR, axis=1) # rows
        oRJ = np.expand_dims(oR, axis=0) # columns
        sRJ = oRJ * scaled_radius_factor
        rSRJ = d_ij + sRJ

        # along the diagonal rSRJ < oRI, resulting in a mask whose
        # diagonals are strictly false.
        mask_final = np.less(oRI, rSRJ)

        d_ij_inv = 1/d_ij 
        # 1/d_ij has NaNs along diagonals so we need to zero it out
        keep_mask = 1 - np.eye(conf.shape[0])
        d_ij_inv = np.where(keep_mask, d_ij_inv, np.zeros_like(d_ij_inv))

        rfs = np.abs(d_ij - sRJ)
        l_ij = np.maximum(oRI, rfs)
        l_ij = 1/l_ij
        u_ij = 1/rSRJ
        l_ij2 = l_ij * l_ij
        u_ij2 = u_ij * u_ij
        ratio = np.log(u_ij/l_ij)
        term = l_ij - u_ij + 0.25*d_ij*(u_ij2 - l_ij2)  + (0.5*d_ij_inv*ratio) + (0.25*sRJ*sRJ*d_ij_inv)*(l_ij2 - u_ij2);
        term_masked = np.where(mask_final, term, np.zeros_like(term))

        summ = np.sum(term_masked, axis=-1)

        summ *= 0.5 * oR
        sum2 = summ*summ
        sum3 = summ*sum2
        tanhSum = np.tanh(alpha_obc*summ - beta_obc*sum2 + gamma_obc*sum3)

        return 1.0/(1.0/oR - tanhSum/atomic_radii)

    def energy(self, conf, params, box=None):
        """
        Compute the GBSA energy.

        Parameters
        ----------
        conf: np.array
            shape Nx3 matrix of geometric coordinates

        params: np.array
            shape [n_unique_params, 3] where:
                params[param_idxs[:, 0]] == charges
                params[param_idxs[:, 1]] == atomic radii
                params[param_idxs[:, 2]] == scaled radius factors
    
        Returns
        -------
        scalar
            total gbsa energy

        """
        if box is not None:
            raise ValueError("Periodic GBSA is not supported.")

        num_atoms = conf.shape[0]

        if self.solute_dielectric != 0.0 and self.solvent_dielectric != 0.0:
            prefactor = 2.0 * self.electric_constant * (1.0/self.solute_dielectric - 1.0/self.solvent_dielectric)
        else:
            prefactor = 0.0


        # (ytz): The rough sketch of the algorithm is as follows:
        # 1. Compute the adjusted GB radii
        # 2. Use the adjusted radiis to compute the shielded electrostatic potential
        # 3. Compute the non-polar contribution using the GB radii

        atomic_radii = params[self.param_idxs[:, 1]]
        scaled_factors = params[self.param_idxs[:, 2]]
        born_radii = self.born_radii(conf, atomic_radii, scaled_factors)

        charges = params[self.param_idxs[:, 0]]

        r_i = np.expand_dims(conf, axis=0)
        r_j = np.expand_dims(conf, axis=1)

        q_i = np.expand_dims(charges, axis=0)
        q_j = np.expand_dims(charges, axis=1)
        q_ij = q_i*q_j

        br_i = np.expand_dims(born_radii, axis=0)
        br_j = np.expand_dims(born_radii, axis=1)

        r2 = np.sum(np.power(r_i - r_j, 2), axis=-1)
        alpha2_ij = br_i * br_j
        D_ij = r2/(4.0*alpha2_ij)
        expTerm = np.exp(-D_ij)
        denom2 = r2 + alpha2_ij*expTerm
        denom = np.sqrt(denom2)
        pq_ij = prefactor*q_ij

        Gpol = pq_ij/denom
        energy = Gpol

        nonpolar_nrg = self.non_polar_ace(born_radii, atomic_radii)

        return np.sum(np.triu(energy)) + np.diagonal(energy)/2.0 + nonpolar_nrg
