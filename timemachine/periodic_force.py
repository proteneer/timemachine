import numpy as np
import tensorflow as tf
from timemachine.force import ConservativeForce
from timemachine.constants import ONE_4PI_EPS0

def generate_inclusion_exclusion_masks(exclusions):
    ones = tf.ones_like(exclusions, dtype=tf.int32)
    mask_a = tf.matrix_band_part(ones, 0, -1) # Upper triangular matrix of 0s and 1s
    mask_b = tf.matrix_band_part(ones, 0, 0) # Diagonal matrix

    # only use upper triangular part
    exclusions = tf.cast(tf.matrix_band_part(exclusions, 0, -1), dtype=tf.int32)
    return (mask_a - mask_b) - exclusions, exclusions

class EwaldElectrostaticForce(ConservativeForce):

    def __init__(self, params, param_idxs, box, exclusions, kmax=10):
        self.params = params
        self.param_idxs = param_idxs # length N
        self.num_atoms = len(self.param_idxs)
        self.charges = tf.gather(self.params, self.param_idxs)
        self.charges = tf.reshape(self.charges, shape=(1, -1))
        self.box = box # we probably want this to be variable when we start to work on barostats
        self.exclusions = exclusions
        self.alphaEwald = 1.0
        self.kmax = kmax
        self.direct_mask, self.exclusion_mask = generate_inclusion_exclusion_masks(exclusions)

        self.recipBoxSize = np.array([
            (2*np.pi)/self.box[0],
            (2*np.pi)/self.box[1],
            (2*np.pi)/self.box[2]]
        )

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
        self.ki = np.expand_dims(self.recipBoxSize, axis=0) * mg # [nk, 3]

    def energy(self, conf):
        direct_nrg, exclusion_nrg = self.direct_and_exclusion_energy(conf)
        return self.reciprocal_energy(conf) + direct_nrg - exclusion_nrg

    def direct_and_exclusion_energy(self, conf):
        charges = tf.gather(self.params, self.param_idxs)
        qi = tf.expand_dims(charges, 0)
        qj = tf.expand_dims(charges, 1)
        qij = tf.multiply(qi, qj)

        ri = tf.expand_dims(conf, 0)
        rj = tf.expand_dims(conf, 1)
        # compute periodic distance
        rij = ri - rj
        base = tf.floor(rij/self.box + 0.5)*self.box # (ytz): can we differentiate through this?
        dxdydz = tf.pow(rij-base, 2)
        d2ij = tf.reduce_sum(dxdydz, axis=-1)

        # (ytz): we move the sqrt to outside of the mask
        # to avoid nans in autograd.

        # direct
        qij_direct_mask = tf.boolean_mask(qij, self.direct_mask)
        d2ij_direct_mask = tf.boolean_mask(d2ij, self.direct_mask)
        r_direct = tf.sqrt(d2ij_direct_mask)
        eij_direct = (qij_direct_mask/r_direct)*tf.erfc(self.alphaEwald*r_direct)

        # exclusions
        qij_exclusion_mask = tf.boolean_mask(qij, self.exclusion_mask)
        d2ij_exclusion_mask = tf.boolean_mask(d2ij, self.exclusion_mask)
        r_exclusion = tf.sqrt(d2ij_exclusion_mask)
        eij_exclusion = (qij_exclusion_mask/r_exclusion)*tf.erf(self.alphaEwald*r_exclusion)

        return ONE_4PI_EPS0*tf.reduce_sum(eij_direct, axis=-1), ONE_4PI_EPS0*tf.reduce_sum(eij_exclusion, axis=-1)

    def reciprocal_energy(self, conf):
        # stack with box vectors
        ki = self.ki
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
        recipCoeff = (ONE_4PI_EPS0*4*np.pi)/(self.box[0]*self.box[1]*self.box[2])

        return recipCoeff * nrg