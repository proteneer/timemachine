import tensorflow as tf
from timemachine.force import ConservativeForce
from timemachine.constants import ONE_4PI_EPS0

class ElectrostaticForce(ConservativeForce):


    def __init__(self, params, param_idxs, exclusions):
        """
        Implements a non-periodic point charge electrostatic potential: sum qiqj

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

        # pre-build mask
        num_atoms = self.param_idxs.shape[0]

        ones = tf.ones(shape=(num_atoms, num_atoms), dtype=tf.int32) # avoid ones like
        mask_a = tf.matrix_band_part(ones, 0, -1) # Upper triangular matrix of 0s and 1s
        mask_b = tf.matrix_band_part(ones, 0, 0) # Diagonal matrix

        # only use upper triangular part
        exclusions = tf.cast(tf.matrix_band_part(exclusions, 0, -1), dtype=tf.int32)
        self.keep_mask = (mask_a - mask_b) - exclusions


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