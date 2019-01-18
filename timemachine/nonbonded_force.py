import numpy as np
import tensorflow as tf
from timemachine.force import Force
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

class LeonnardJones(Force):

    def __init__(self, params, param_idxs, exclusions):
        """
        Implements a non-periodic LJ612 potential using the Lorentz−Berthelot terms,
        where sig_ij = sig_i + sig_j and eps_ij = eps_i * eps_j.

        Parameters:
        -----------
        params: tf.float64 variable
            list of parameters for Ai and Ci used in the OPLS combining
            rules.

        param_idxs: tf.int32 (N,2)
            each tuple (Ai, Ci) is used as part of the combining rules

        exclusions: tf.bool (N, N)
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
        self.keep_mask = generate_exclusion_masks(exclusions)
        self.box = box

    def energy(self, conf, box=None):
        A = tf.gather(self.params, self.param_idxs[:, 0])
        C = tf.gather(self.params, self.param_idxs[:, 1])

        sig_i = tf.expand_dims(A, 0)
        sig_j = tf.expand_dims(A, 1)
        sig_ij = sig_i + sig_j

        eps_i = tf.expand_dims(C, 0)
        eps_j = tf.expand_dims(C, 1)
        eps_ij = eps_i * eps_j

        ri = tf.expand_dims(conf, 0)
        rj = tf.expand_dims(conf, 1)

        if self.box is not None:
            # periodic distance
            rij = ri - rj
            base = tf.floor(rij/self.box + 0.5)*self.box # (ytz): can we differentiate through this?
            dxdydz = tf.pow(rij-base, 2)
            d2ij = tf.reduce_sum(dxdydz, axis=-1)
        else:
            # nonperiodic distance
            d2ij = tf.reduce_sum(tf.pow(ri-rj, 2), axis=-1)

        sig = tf.boolean_mask(sig_ij, self.keep_mask)
        eps = tf.boolean_mask(eps_ij, self.keep_mask)
        d2ij = tf.boolean_mask(d2ij, self.keep_mask)
        dij = tf.sqrt(d2ij)

        sig2 = sig/dij
        sig2 *= sig2
        sig6 = sig2*sig2*sig2

        energy = eps*(sig6-1.0)*sig6

        return tf.reduce_sum(energy, axis=-1)

class Electrostatic(Force):

    def __init__(self, params, param_idxs, exclusions, box=None, kmax=10):
        self.params = params
        self.param_idxs = param_idxs # length N
        self.num_atoms = len(self.param_idxs)
        self.charges = tf.gather(self.params, self.param_idxs)
        self.charges = tf.reshape(self.charges, shape=(1, -1))
        self.exclusions = exclusions
        self.alphaEwald = 1.0
        self.direct_mask, self.exclusion_mask = generate_inclusion_exclusion_masks(exclusions)
        self.box = box # we probably want this to be variable when we start to work on barostats
        self.kmax = kmax

    def energy(self, conf, box=None):
        direct_nrg, exclusion_nrg = self.direct_and_exclusion_energy(conf)
        if box is None:
            return direct_nrg
        else:
            print(self.reciprocal_energy(conf, box).shape, direct_nrg.shape, exclusion_nrg.shape, self.self_energy(conf).shape)
            return self.reciprocal_energy(conf, box) + direct_nrg - exclusion_nrg - self.self_energy(conf)

    def self_energy(self, conf):
        return tf.reduce_sum(ONE_4PI_EPS0 * tf.pow(self.charges, 2) * self.alphaEwald/np.sqrt(np.pi))

    def direct_and_exclusion_energy(self, conf):
        charges = tf.gather(self.params, self.param_idxs)
        qi = tf.expand_dims(charges, 0)
        qj = tf.expand_dims(charges, 1)
        qij = tf.multiply(qi, qj)

        ri = tf.expand_dims(conf, 0)
        rj = tf.expand_dims(conf, 1)

        if self.box is not None:
            rij = ri - rj
            base = tf.floor(rij/self.box + 0.5)*self.box # (ytz): can we differentiate through this?
            dxdydz = tf.pow(rij-base, 2)
            d2ij = tf.reduce_sum(dxdydz, axis=-1)
        else:
            d2ij = tf.reduce_sum(tf.pow(ri-rj, 2), axis=-1)

        # (ytz): we move the sqrt to outside of the mask
        # to avoid nans in autograd.

        # direct
        qij_direct_mask = tf.boolean_mask(qij, self.direct_mask)
        d2ij_direct_mask = tf.boolean_mask(d2ij, self.direct_mask)
        r_direct = tf.sqrt(d2ij_direct_mask)
        eij_direct = qij_direct_mask/r_direct

        if self.box is not None:
            # We adjust direct by the erfc, and adjust the reciprocal space's
            # exclusionary contribution by the direction space weighted by erf
            eij_direct *= tf.erfc(self.alphaEwald*r_direct)
            # exclusions to subtract from reciprocal space
            qij_exclusion_mask = tf.boolean_mask(qij, self.exclusion_mask)
            d2ij_exclusion_mask = tf.boolean_mask(d2ij, self.exclusion_mask)
            r_exclusion = tf.sqrt(d2ij_exclusion_mask)
            eij_exclusion = qij_exclusion_mask/r_exclusion
            eij_exclusion *= tf.erf(self.alphaEwald*r_exclusion)

            return ONE_4PI_EPS0*tf.reduce_sum(eij_direct, axis=-1), ONE_4PI_EPS0*tf.reduce_sum(eij_exclusion, axis=-1)
        else:
            return ONE_4PI_EPS0*tf.reduce_sum(eij_direct, axis=-1), None

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

# class ElectrostaticForce(ConservativeForce):


#     def __init__(self, params, param_idxs, exclusions):
#         """
#         Implements a non-periodic point charge electrostatic potential.

#         Parameters:
#         -----------
#         params: tf.float64 variable
#             charge list, eg. [q_C, q_C+, q_C-, q_H-]

#         param_idxs: tf.int32 (N,)    
#             table of indexes into each charge type.

#         exclusions: tf.bool (N, N)
#             boolean mask denoting if interaction e[i,j] should be
#             excluded or not. If e[i,j] is 1 then the interaction
#             is excluded, 0 implies it is kept. Note that only the upper
#             right triangular portion of this is used.

#         """
#         # (ytz) TODO: implement Ewald/PME
#         self.params = params
#         self.param_idxs = param_idxs
#         self.keep_mask = generate_exclusion_masks(exclusions)


#     def energy(self, conf):

#         charges = tf.gather(self.params, self.param_idxs)
#         qi = tf.expand_dims(charges, 0)
#         qj = tf.expand_dims(charges, 1)
#         qij = tf.multiply(qi, qj)

#         ri = tf.expand_dims(conf, 0)
#         rj = tf.expand_dims(conf, 1)
#         d2ij = tf.reduce_sum(tf.pow(ri-rj, 2), axis=-1)

#         qij_mask = tf.boolean_mask(qij, self.keep_mask)
#         d2ij_mask = tf.boolean_mask(d2ij, self.keep_mask)

#         # (ytz): we move the sqrt to outside of the mask
#         # to avoid nans in autograd.
#         eij = qij_mask/tf.sqrt(d2ij_mask)

#         return ONE_4PI_EPS0*tf.reduce_sum(eij, axis=-1)

if __name__ == "__main__":

    # phi_dir
    box_vectors = np.array([6.0, 5.5, 4.5], dtype=np.float64)

    coords = np.array([1.9, 3.4, 2.2], dtype=np.float64)

    beta = 0.5

    tf.erfc(beta*tf.norm(r+n))/tf.norm(r+n)


    # phi_rec