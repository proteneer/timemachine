import numpy as np
import tensorflow as tf
from timemachine.functionals import Energy
from timemachine.constants import ONE_4PI_EPS0


# (TODO): generalize exclusions so we can support fudge factors.
def generate_exclusion_masks(exclusions):
    ones = tf.ones_like(exclusions, dtype=tf.int32)
    mask_a = tf.matrix_band_part(ones, 0, -1) # Upper triangular matrix of 0s and 1s
    mask_b = tf.matrix_band_part(ones, 0, 0) # Diagonal matrix

    # only use upper triangular part
    exclusions = tf.cast(tf.matrix_band_part(exclusions, 0, -1), dtype=tf.int32)
    return (mask_a - mask_b) - exclusions


def generate_inclusion_exclusion_masks(exclusions):
    ones = tf.ones_like(exclusions, dtype=tf.int32)
    mask_a = tf.matrix_band_part(ones, 0, -1) # Upper triangular matrix of 0s and 1s
    mask_b = tf.matrix_band_part(ones, 0, 0) # Diagonal matrix

    # only use upper triangular part
    exclusions = tf.cast(tf.matrix_band_part(exclusions, 0, -1), dtype=tf.int32)
    return (mask_a - mask_b) - exclusions, exclusions

class LeonnardJones(Energy):

    def __init__(self, params, param_idxs, scale_matrix, cutoff=None):
        """
        Implements a non-periodic LJ612 potential using the Lorentz−Berthelot terms,
        where sig_ij = sig_i + sig_j and eps_ij = eps_i * eps_j.

        Parameters:
        -----------
        params: tf.float64 variable
            list of parameters for sig and eps in LB combining rules

        param_idxs: tf.int32 (N,2)
            each tuple (sig, eps) is used as part of the combining rules

        scale_matrix: tf.bool (N, N)
            boolean mask denoting if interaction e[i,j] should be
            excluded or not. If e[i,j] is 1 then the interaction
            is excluded, 0 implies it is kept. Note that only the upper
            right triangular portion of this is used.

        box: np.array (3,)
            Rectangular box definitions. If None then we assume the system
            to be aperiodic. Otherwise we enforce periodic boundary conditions

        """
        self.params = params
        self.param_idxs = param_idxs
        self.scale_matrix = scale_matrix
        self.cutoff = cutoff

    def energy(self, conf, box=None):
        sig = tf.gather(self.params, self.param_idxs[:, 0])
        eps = tf.gather(self.params, self.param_idxs[:, 1])

        sig_i = tf.expand_dims(sig, 0)
        sig_j = tf.expand_dims(sig, 1)
        sig_ij = (sig_i + sig_j)/2

        sig_ij_raw = sig_ij

        eps_i = tf.expand_dims(eps, 0)
        eps_j = tf.expand_dims(eps, 1)
        eps_ij = self.scale_matrix * tf.sqrt(eps_i * eps_j)

        eps_ij_raw = eps_ij

        ri = tf.expand_dims(conf, 0)
        rj = tf.expand_dims(conf, 1)

        if box is not None:
            # periodic distance
            rij = ri - rj
            base = tf.floor(rij/box + 0.5)*box # (ytz): can we differentiate through this?
            dxdydz = tf.pow(rij-base, 2)
            d2ij = tf.reduce_sum(dxdydz, axis=-1)
        else:
            # nonperiodic distance
            d2ij = tf.reduce_sum(tf.pow(ri-rj, 2), axis=-1)

        if self.cutoff is not None:
            eps_ij = tf.where(d2ij < self.cutoff*self.cutoff, eps_ij, tf.zeros_like(eps_ij))

        keep_mask = self.scale_matrix > 0

        sig_ij = tf.boolean_mask(sig_ij, keep_mask)
        eps_ij = tf.boolean_mask(eps_ij, keep_mask)
        d2ij = tf.boolean_mask(d2ij, keep_mask)

        dij = tf.sqrt(d2ij)

        sig2 = sig_ij/dij
        sig2 *= sig2
        sig6 = sig2*sig2*sig2

        energy = 4*eps_ij*(sig6-1.0)*sig6
        # divide by two to deal with symmetry
        return tf.reduce_sum(energy, axis=-1)/2, 4*eps_ij_raw, sig_ij_raw, energy

class Electrostatic(Energy):

    def __init__(self, params, param_idxs, scale_matrix, cutoff=None, crf=1.0, kmax=10):
        self.params = params
        self.param_idxs = param_idxs # length N
        self.num_atoms = len(self.param_idxs)
        self.charges = tf.gather(self.params, self.param_idxs)
        self.charges = tf.reshape(self.charges, shape=(1, -1))
        self.scale_matrix = scale_matrix
        self.alphaEwald = 1.0
        self.cutoff = cutoff
        self.kmax = kmax
        self.crf = crf

    def energy(self, conf, box=None):
        direct_nrg, exclusion_nrg = self.direct_and_exclusion_energy(conf, box)
        if box is None:
            return direct_nrg
        else:
            return self.reciprocal_energy(conf, box) + direct_nrg - exclusion_nrg - self.self_energy(conf)

    def self_energy(self, conf):
        return tf.reduce_sum(ONE_4PI_EPS0 * tf.pow(self.charges, 2) * self.alphaEwald/np.sqrt(np.pi))

    def direct_and_exclusion_energy(self, conf, box):
        charges = tf.gather(self.params, self.param_idxs)
        qi = tf.expand_dims(charges, 0)
        qj = tf.expand_dims(charges, 1)
        qij = self.scale_matrix * tf.multiply(qi, qj)

        ri = tf.expand_dims(conf, 0)
        rj = tf.expand_dims(conf, 1)

        if box is not None:
            rij = ri - rj
            base = tf.floor(rij/box + 0.5)*box # (ytz): can we differentiate through this?
            dxdydz = tf.pow(rij-base, 2)
            d2ij = tf.reduce_sum(dxdydz, axis=-1)
        else:
            d2ij = tf.reduce_sum(tf.pow(ri-rj, 2), axis=-1)

        d2ij = tf.where(d2ij != 0.0, d2ij, tf.zeros_like(d2ij))
        dij_inverse = 1/tf.sqrt(d2ij)

        if self.cutoff is not None:
            # apply only to fully non-excepted terms
            dij_inverse = tf.where(self.scale_matrix == 1.0, dij_inverse - self.crf, dij_inverse)
            qij = tf.where(d2ij < self.cutoff*self.cutoff, qij, tf.zeros_like(qij))

        # (ytz): we move the sqrt to outside of the mask
        # to avoid nans in autograd.
        direct_mask = self.scale_matrix > 0
        qij_direct_mask = tf.boolean_mask(qij, direct_mask)
        dij_inverse_mask =  tf.boolean_mask(dij_inverse, direct_mask)
        eij_direct = qij_direct_mask * dij_inverse_mask

        if box is not None:
            # We adjust direct by the erfc, and adjust the reciprocal space's
            # exclusionary contribution by the direction space weighted by erf
            eij_direct *= tf.erfc(self.alphaEwald*r_direct)
            # exclusions to subtract from reciprocal space
            qij_exclusion_mask = tf.boolean_mask(qij, self.exclusion_mask)
            d2ij_exclusion_mask = tf.boolean_mask(d2ij, self.exclusion_mask)
            r_exclusion = tf.sqrt(d2ij_exclusion_mask)
            eij_exclusion = qij_exclusion_mask/r_exclusion
            eij_exclusion *= tf.erf(self.alphaEwald*r_exclusion)

            # extra factor of 2 is to deal with the fact that we compute the full matrix as opposed to the upper right
            return ONE_4PI_EPS0*tf.reduce_sum(eij_direct, axis=-1)/2, ONE_4PI_EPS0*tf.reduce_sum(eij_exclusion, axis=-1)
        else:
            return ONE_4PI_EPS0*tf.reduce_sum(eij_direct, axis=-1)/2, None

    def reciprocal_energy(self, conf, box):
        assert box is not None

        recipBoxSize = (2*np.pi)/box

        mg = []
        lowry = 0
        lowrz = 1

        numRx, numRy, numRz = self.kmax, self.kmax, self.kmax

        for rx in range(numRx):
            for ry in range(lowry, numRy):
                for rz in range(lowrz, numRz):
                    mg.append((rx, ry, rz))
                    lowrz = 1 - numRz
                lowry = 1 - numRy

        # lattice vectors
        ki = tf.expand_dims(recipBoxSize, axis=0) * mg # [nk, 3]
        ri = tf.expand_dims(conf, axis=0) # [1, N, 3]
        rik = tf.reduce_sum(tf.multiply(ri, tf.expand_dims(ki, axis=1)), axis=-1) # [nk, N]
        real = tf.cos(rik)
        imag = tf.sin(rik)
        eikr = tf.complex(real, imag) # [nk, N]
        qi = tf.complex(self.charges, np.float64(0.0))
        Sk = tf.reduce_sum(qi*eikr, axis=-1)  # [nk]
        n2Sk = tf.pow(tf.abs(Sk), 2)
        k2 = tf.reduce_sum(tf.multiply(ki, ki), axis=-1) # [nk]
        factorEwald = -1/(4*self.alphaEwald*self.alphaEwald)
        ak = tf.exp(k2*factorEwald)/k2 # [nk]
        nrg = tf.reduce_sum(ak * n2Sk)
        recipCoeff = (ONE_4PI_EPS0*4*np.pi)/(box[0]*box[1]*box[2])

        return recipCoeff * nrg
