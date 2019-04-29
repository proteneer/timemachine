import time
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
# import jax
# import jax.numpy as np
import autograd
import autograd.numpy as np
import tensorflow as tf

BOLTZMANN = 1.380658e-23
AVOGADRO = 6.0221367e23
RGAS = BOLTZMANN*AVOGADRO
BOLTZ = RGAS/1000
ONE_4PI_EPS0 = 138.935456
VIBRATIONAL_CONSTANT = 1302.79 # http://openmopac.net/manual/Hessian_Matrix.html

class EwaldEnergy():

    def __init__(self, kmax, charges, alphaEwald, box):
        self.kmax = kmax
        self.charges = charges
        self.alphaEwald = alphaEwald
        self.box = box

        self.recipBoxSize = (2*np.pi)/box

        self.mg = []
        lowry = 0
        lowrz = 1

        numRx, numRy, numRz = self.kmax, self.kmax, self.kmax

        for rx in range(numRx):
            for ry in range(lowry, numRy):
                for rz in range(lowrz, numRz):
                    self.mg.append((rx, ry, rz))
                    lowrz = 1 - numRz
                lowry = 1 - numRy

        self.mg = onp.array(self.mg)

    def jax_reciprocal_energy(self, conf):

        # lattice vectors
        ki = np.expand_dims(self.recipBoxSize, axis=0) * self.mg # [nk, 3]
        ri = np.expand_dims(conf, axis=0) # [1, N, 3]
        rik = np.sum(np.multiply(ri, np.expand_dims(ki, axis=1)), axis=-1) # [nk, N]
        real = np.cos(rik)
        imag = np.sin(rik)
        # eikr = np.complex(real, imag) # [nk, N]
        eikr = real + 1j*imag
        # qi = np.complex(self.charges, np.float64(0.0))
        qi = self.charges + 1j*0
        Sk = np.sum(qi*eikr, axis=-1)  # [nk]
        n2Sk = np.power(np.real(Sk), 2)
        k2 = np.sum(np.multiply(ki, ki), axis=-1) # [nk]
        factorEwald = -1/(4*self.alphaEwald*self.alphaEwald)
        ak = np.exp(k2*factorEwald)/k2 # [nk]
        nrg = np.sum(ak * n2Sk)
        recipCoeff = (ONE_4PI_EPS0*4*np.pi)/(self.box[0]*self.box[1]*self.box[2])

        return recipCoeff * nrg


    def tf_reciprocal_energy(self, conf):

        # lattice vectors
        ki = tf.expand_dims(self.recipBoxSize, axis=0) * self.mg # [nk, 3]
        ri = tf.expand_dims(conf, axis=0) # [1, N, 3]
        rik = tf.reduce_sum(tf.multiply(ri, tf.expand_dims(ki, axis=1)), axis=-1) # [nk, N]
        real = tf.cos(rik)
        imag = tf.sin(rik)
        eikr = tf.complex(real, imag) # [nk, N]
        qi = tf.complex(self.charges, np.float64(0.0))
        Sk = tf.reduce_sum(qi*eikr, axis=-1)  # [nk]
        n2Sk = tf.pow(tf.real(Sk), 2)
        k2 = tf.reduce_sum(tf.multiply(ki, ki), axis=-1) # [nk]
        factorEwald = -1/(4*self.alphaEwald*self.alphaEwald)
        ak = tf.exp(k2*factorEwald)/k2 # [nk]
        nrg = tf.reduce_sum(ak * n2Sk)
        recipCoeff = (ONE_4PI_EPS0*4*np.pi)/(self.box[0]*self.box[1]*self.box[2])

        return recipCoeff * nrg

if __name__ == "__main__":

    charges = onp.array([
        0.1,
        -0.1,
        0.3,
        0.15,
        -0.4
    ], dtype=np.float64)

    ee = EwaldEnergy(
        kmax=4, 
        charges=charges, 
        alphaEwald=1.0,
        box=onp.array([4.0, 4.0, 4.0], dtype=np.float64))

    x0 = onp.array([
        [ 0.0637,   0.0126,   0.2203],
        [ 1.0573,  -0.2011,   1.2864],
        [ 2.3928,   1.2209,  -0.2230],
        [-0.6891,   1.6983,   0.0780],
        [-0.6312,  -1.6261,  -0.2601]
    ], dtype=np.float64)

    xt = tf.convert_to_tensor(x0)

    nrg_op = ee.tf_reciprocal_energy(xt)
    grad_op = tf.gradients(nrg_op, xt)[0]
    hess_op = tf.hessians(nrg_op, xt)

    sess = tf.Session()
    nrg_tf = sess.run([nrg_op])

    nrg_jax = ee.jax_reciprocal_energy(x0)
    onp.testing.assert_almost_equal(nrg_tf, nrg_jax)

    grad_tf = sess.run([grad_op])[0]

    grad_jax_rev_fn = jax.jacrev(ee.jax_reciprocal_energy)
    grad_jax_rev = grad_jax_rev_fn(x0)

    # grad_rev passes
    onp.testing.assert_almost_equal(grad_tf, grad_jax_rev)

    grad_jax_fwd_fn = jax.jacfwd(ee.jax_reciprocal_energy)
    grad_jax_fwd = grad_jax_fwd_fn(x0)

    # grad_fwd passes
    onp.testing.assert_almost_equal(grad_tf, grad_jax_fwd)

    hess_jax_fwd_rev_fn = jax.jacfwd(jax.jacrev(ee.jax_reciprocal_energy))
    hess_jax_fwd_rev = hess_jax_fwd_rev_fn(x0)

    hess_tf = sess.run([hess_op])[0][0]

    # hessian fails
    print(hess_jax_fwd_rev)

    assert 0
    onp.testing.assert_almost_equal(hess_tf, hess_jax_fwd_rev)
