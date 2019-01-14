import tensorflow as tf
from timemachine.force import ConservativeForce
from timemachine.constants import ONE_4PI_EPS0


# (TODO): generalize exclusions so we can support fudge factors.
def generate_exclusion_masks(exclusions):
    ones = tf.ones_like(exclusions, dtype=tf.int32)
    mask_a = tf.matrix_band_part(ones, 0, -1) # Upper triangular matrix of 0s and 1s
    mask_b = tf.matrix_band_part(ones, 0, 0) # Diagonal matrix

    # only use upper triangular part
    exclusions = tf.cast(tf.matrix_band_part(exclusions, 0, -1), dtype=tf.int32)
    return (mask_a - mask_b) - exclusions


class LJ612Force(ConservativeForce):


    def __init__(self, params, param_idxs, exclusions):
        """
        Implements a non-periodic LJ612 potential using the Jorgensen
        combining rules for sigma/epsilon.

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

        """
        self.params = params
        self.param_idxs = param_idxs
        self.keep_mask = generate_exclusion_masks(exclusions)

    def energy(self, conf):

        A = tf.gather(self.params, self.param_idxs[:, 0])
        C = tf.gather(self.params, self.param_idxs[:, 1])

        Ai = tf.expand_dims(A, 0)
        Aj = tf.expand_dims(A, 1)
        Aij = tf.sqrt(tf.multiply(Ai, Aj))

        Ci = tf.expand_dims(C, 0)
        Cj = tf.expand_dims(C, 1)
        Cij = tf.sqrt(tf.multiply(Ci, Cj))

        ri = tf.expand_dims(conf, 0)
        rj = tf.expand_dims(conf, 1)
        d2ij = tf.reduce_sum(tf.pow(ri-rj, 2), axis=-1)

        Aij_mask = tf.boolean_mask(Aij, self.keep_mask)
        Cij_mask = tf.boolean_mask(Cij, self.keep_mask)
        d2ij_mask = tf.boolean_mask(d2ij, self.keep_mask)       
        d6ij_mask = d2ij_mask*d2ij_mask*d2ij_mask
        d12ij_mask = d6ij_mask*d6ij_mask

        energy = Aij_mask/d12ij_mask - Cij_mask/d6ij_mask
        return tf.reduce_sum(energy, axis=-1)

class ElectrostaticForce(ConservativeForce):


    def __init__(self, params, param_idxs, exclusions):
        """
        Implements a non-periodic point charge electrostatic potential.

        Parameters:
        -----------
        params: tf.float64 variable
            charge list, eg. [q_C, q_C+, q_C-, q_H-]

        param_idxs: tf.int32 (N,)    
            table of indexes into each charge type.

        exclusions: tf.bool (N, N)
            boolean mask denoting if interaction e[i,j] should be
            excluded or not. If e[i,j] is 1 then the interaction
            is excluded, 0 implies it is kept. Note that only the upper
            right triangular portion of this is used.

        """
        # (ytz) TODO: implement Ewald/PME
        self.params = params
        self.param_idxs = param_idxs
        self.keep_mask = generate_exclusion_masks(exclusions)


    def energy(self, conf):

        charges = tf.gather(self.params, self.param_idxs)
        qi = tf.expand_dims(charges, 0)
        qj = tf.expand_dims(charges, 1)
        qij = tf.multiply(qi, qj)

        ri = tf.expand_dims(conf, 0)
        rj = tf.expand_dims(conf, 1)
        d2ij = tf.reduce_sum(tf.pow(ri-rj, 2), axis=-1)

        qij_mask = tf.boolean_mask(qij, self.keep_mask)
        d2ij_mask = tf.boolean_mask(d2ij, self.keep_mask)

        # (ytz): we move the sqrt to outside of the mask
        # to avoid nans in autograd.
        eij = qij_mask/tf.sqrt(d2ij_mask)

        return ONE_4PI_EPS0*tf.reduce_sum(eij, axis=-1)

if __name__ == "__main__":

    # phi_dir
    box_vectors = np.array([6.0, 5.5, 4.5], dtype=np.float64)

    coords = np.array([1.9, 3.4, 2.2], dtype=np.float64)

    beta = 0.5

    tf.erfc(beta*tf.norm(r+n))/tf.norm(r+n)


    # phi_rec