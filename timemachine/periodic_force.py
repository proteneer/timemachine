import numpy as np
import tensorflow as tf
from timemachine.force import ConservativeForce

class EwaldElectrostaticForce(ConservativeForce):

    def __init__(self, params, param_idxs, box, exclusions):
        self.params = params
        self.param_idxs = param_idxs # length N
        self.charges = [self.params[p_idx] for p_idx in self.param_idxs]
        self.num_atoms = len(self.charges)
        self.box = box # we probably want this to be variable when we start to work on barostats
        self.exclusions = exclusions
        self.alphaEwald = 1.0
        self.kmax = 10

    def _construct_eir(self, conf):


        # generate a K, N, 3



        recipBoxSize = tf.expand_dims(
            np.array([(2*np.pi)/self.box[0], (2*np.pi)/self.box[1], (2*np.pi)/self.box[2]]),
            axis=0
        )

        # [N,3] by [1,3]
        reals = tf.expand_dims(tf.cos(tf.multiply(conf, recipBoxSize)), axis=0)
        imags = tf.expand_dims(tf.sin(tf.multiply(conf, recipBoxSize)), axis=0)

        # [1, N,3] of complex numbers
        eir = tf.reshape(tf.complex(reals, imags), shape=(1, self.num_atoms, 3))

        exponents = np.arange(0, self.kmax, dtype=np.complex128)
        exponents = tf.reshape(exponents, (-1,1,1)) # (K, 1, 1)


        return tf.pow(eir, exponents)