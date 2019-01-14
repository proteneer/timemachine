import numpy as np
import tensorflow as tf
from timemachine.force import ConservativeForce
from timemachine.constants import ONE_4PI_EPS0

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

        return self.reciprocal_energy(conf)

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