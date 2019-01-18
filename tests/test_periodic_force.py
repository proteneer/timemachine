import unittest
import numpy as np
import tensorflow as tf

from scipy.special import erf, erfc
from timemachine.constants import ONE_4PI_EPS0
from timemachine.nonbonded_force import LeonnardJones, Electrostatic
from timemachine import derivatives

def periodic_difference(val1, val2, period):
    diff = val1-val2
    base = np.floor(diff/period+0.5)*period
    return diff-base;


def periodic_dij(ri, rj, boxSize):
    dx = periodic_difference(ri[0], rj[0], boxSize[0])
    dy = periodic_difference(ri[1], rj[1], boxSize[1])
    dz = periodic_difference(ri[2], rj[2], boxSize[2])
    return np.sqrt(dx*dx + dy*dy + dz*dz)

class ReferenceLeonnardJonesEnergy():

    def __init__(self, params, param_idxs, exclusions, box, kmax=0):
        self.params = params # variadic
        self.param_idxs = param_idxs # (N, 2), last rank is (sig_idx, eps_idx)
        self.num_atoms = len(self.param_idxs)
        self.box = box
        self.exclusions = exclusions

    def openmm_energy(self, conf):

        # minimum image convention
        direct_vdw_nrg = 0

        for i in range(self.num_atoms):
            sig_i = self.params[self.param_idxs[i, 0]]
            eps_i = self.params[self.param_idxs[i, 1]]
            ri = conf[i]
            for j in range(i+1, self.num_atoms):
                
                if self.exclusions[i][j]:
                    continue

                rj = conf[j]
                r = periodic_dij(ri, rj, self.box)

                sig_j = self.params[self.param_idxs[j, 0]]
                eps_j = self.params[self.param_idxs[j, 1]]

                sig = sig_i + sig_j
                sig2 = sig/r
                sig2 *= sig2
                sig6 = sig2*sig2*sig2
                eps = eps_i * eps_j

                vdwEnergy = eps*(sig6-1.0)*sig6

                direct_vdw_nrg += vdwEnergy

        return direct_vdw_nrg

class ReferenceEwaldEnergy():

    def __init__(self, params, param_idxs, exclusions, box, kmax=10):
        self.params = params
        self.param_idxs = param_idxs # length N
        self.charges = [self.params[p_idx] for p_idx in self.param_idxs]
        self.num_atoms = len(self.param_idxs)
        self.box = box # we probably want this to be variable when we start to work on barostats
        self.exclusions = exclusions
        self.alphaEwald = 1.0 # 1/(sqrt(2)*sigma)
        self.kmax = kmax
        self.recipBoxSize = np.array([
            (2*np.pi)/self.box[0],
            (2*np.pi)/self.box[1],
            (2*np.pi)/self.box[2]]
        )

    def energy(self, conf):
        return self.openmm_reciprocal_energy(conf)

    def _construct_eir(self, conf):

        eir = np.zeros((self.kmax, self.num_atoms, 3), dtype=np.complex128)
        for i in range(self.num_atoms):
            coords = conf[i, :]

            # j == 0
            for m in range(3):
                eir[0, i, m] = np.complex(1., 0.)

            # j == 1
            for m in range(3):
                eir[1, i, m] = np.complex(
                    np.cos(coords[m]*self.recipBoxSize[m]), # real
                    np.sin(coords[m]*self.recipBoxSize[m])  # imag
                )
            # j == 2
            for j in range(2, self.kmax):
                for m in range(3):
                    # (ytz): this is complex multiplication, not 
                    # element-wise multiplication
                    eir[j, i, m] = eir[j-1, i, m] * eir[1, i, m]

        return eir

    def openmm_exclusion_energy(self, conf):
        # minimum image convention, add back in the erf part
        energy = 0
        for i in range(self.num_atoms):
            qi = self.charges[i]
            for j in range(i+1, self.num_atoms):
                if self.exclusions[i][j]:
                    continue

                    qj = self.charges[j]
                    ri = conf[i]
                    rj = conf[j]
                    r = periodic_dij(ri, rj, self.box)
                    alphaR = self.alphaEwald*r
                    energy += (qi*qj/r)*erf(alphaR);

        return ONE_4PI_EPS0*energy

    def openmm_direct_and_exclusion_energy(self, conf):

        # minimum image convention
        direct_nrg = 0
        exclusion_nrg = 0
        for i in range(self.num_atoms):
            qi = self.charges[i]
            for j in range(i+1, self.num_atoms):

                qj = self.charges[j]
                ri = conf[i]
                rj = conf[j]
                r = periodic_dij(ri, rj, self.box)
                alphaR = self.alphaEwald*r
                ixn_nrg = (qi*qj)/r

                if self.exclusions[i][j]:
                    exclusion_nrg += ixn_nrg*erf(alphaR)
                else:
                    direct_nrg += ixn_nrg*erfc(alphaR)

        return ONE_4PI_EPS0*direct_nrg, ONE_4PI_EPS0*exclusion_nrg

    def reference_reciprocal_energy(self, conf):

        # this generates an energy thats 2x the expected due to the double counting
        # x_tiles = [0]
        # y_tiles = [0]
        # z_tiles = [0]

        # for i in range(1, self.kmax):
        #     x_tiles.extend([i, -i])
        #     y_tiles.extend([i, -i])
        #     z_tiles.extend([i, -i])
        # # we discard the k=[0,0,0] cells.
        # mg = np.stack(np.meshgrid(x_tiles, y_tiles, z_tiles), -1).reshape(-1, 3)[1:] # [nk, 3]

        # a faster implementation, saves by factor of two by exploiting anti-symmetry
        # generates indices using the same method as OpenMM
        mg = []
        lowry = 0
        lowrz = 1

        numRx, numRy, numRz = self.kmax, self.kmax, self.kmax

        for rx in range(self.kmax):
            for ry in range(lowry, self.kmax):
                for rz in range(lowrz, self.kmax):
                    mg.append((rx, ry, rz))
                    lowrz = 1 - numRz
                lowry = 1 - numRy

        ki = np.expand_dims(self.recipBoxSize, axis=0) * mg # [nk, 3]

        # stack with box vectors

        ri = np.expand_dims(conf, axis=0) # [1, N, 3]
        rik = np.sum(np.multiply(ri, np.expand_dims(ki, axis=1)), axis=-1) # [nk, N]

        real = np.cos(rik)
        imag = np.sin(rik)

        eikr = real + 1j*imag # [nk, N]

        qi = np.reshape(np.array(self.charges), (1, -1)) # [1, N]

        Sk = np.sum(qi*eikr, axis=-1)  # [nk]
        n2Sk = np.power(np.absolute(Sk), 2)
        k2 = np.sum(np.multiply(ki, ki), axis=-1) # [nk]

        factorEwald = -1/(4*self.alphaEwald*self.alphaEwald)
        ak = np.exp(k2*factorEwald)/k2 # [nk]
        nrg = np.sum(ak * n2Sk)

        recipCoeff = (ONE_4PI_EPS0*4*np.pi)/(self.box[0]*self.box[1]*self.box[2])

        return recipCoeff * nrg

    def openmm_self_energy(self, conf):
        # self-energy actually doesn't contribute to the forces since it doesn't depend
        # on the geometry
        nrg = 0
        for n in range(self.num_atoms):
            nrg += ONE_4PI_EPS0*self.charges[n] * self.charges[n]*self.alphaEwald/np.sqrt(np.pi)
        return nrg

    def openmm_reciprocal_energy(self, conf):
        # Reference implementation taken from ReferenceLJCoulombIxn.cpp in OpenMM
        N = self.num_atoms

        eir = self._construct_eir(conf)
        tab_xy = np.zeros(N, np.complex128)
        tab_qxyz = np.zeros(N, np.complex128)

        totalRecipEnergy = 0
        lowry = 0
        lowrz = 1

        numRx = self.kmax
        numRy = self.kmax
        numRz = self.kmax

        factorEwald = -1 / (4*self.alphaEwald*self.alphaEwald)
        epsilon = 1.0
        recipCoeff = (ONE_4PI_EPS0*4*np.pi)/(self.box[0]*self.box[1]*self.box[2])/epsilon;

        all_idxs = []

        for rx in range(numRx):

            kx = rx * self.recipBoxSize[0]

            for ry in range(lowry, numRy):

                ky = ry * self.recipBoxSize[1]

                if ry >= 0:
                    for n in range(N):
                        tab_xy[n] = eir[rx, n, 0] * eir[ry, n, 1]
                else:
                    for n in range(N):
                        tab_xy[n] = eir[rx, n, 0] * np.conj(eir[-ry, n, 1])

                for rz in range(lowrz, numRz):
                    all_idxs.append((rx, ry, rz))
                    if rz >= 0:
                        for n in range(N):
                            tab_qxyz[n] = self.charges[n] * tab_xy[n] * eir[rz, n, 2]
                    else:
                        for n in range(N):
                            tab_qxyz[n] = self.charges[n] * tab_xy[n] * np.conj(eir[-rz, n, 2])

                    cs = 0
                    ss = 0

                    for n in range(N):
                        cs += tab_qxyz[n].real
                        ss += tab_qxyz[n].imag

                    kz = rz * self.recipBoxSize[2]
                    k2 = kx * kx + ky*ky + kz*kz
                    ak = np.exp(k2*factorEwald) / k2
                    recipEnergy = ak * (cs * cs + ss * ss)
                    totalRecipEnergy += recipEnergy

                    lowrz = 1 - numRz

                lowry = 1 - numRy

        return recipCoeff * totalRecipEnergy


class TestPeriodicForce(unittest.TestCase):

    def setUp(self):
        self.x0 = np.array([
            [ 0.0637,   0.0126,   0.2203],
            [ 1.0573,  -0.2011,   1.2864],
            [ 2.3928,   1.2209,  -0.2230],
            [-0.6891,   1.6983,   0.0780],
            [-0.6312,  -1.6261,  -0.2601]
        ], dtype=np.float64)

        self.exclusions = np.array([
            [0,0,1,0,0],
            [0,0,0,1,1],
            [1,0,0,0,0],
            [0,1,0,0,1],
            [0,1,0,1,0],
            ], dtype=np.bool)

        self.box = [10.0, 10.0, 10.0]

    def tearDown(self):
        tf.reset_default_graph()

    def test_reference_leonnard_jones(self):
        x0 = self.x0
        exclusions = self.exclusions

        params_np = np.array([3.0, 2.0, 1.0, 1.4], dtype=np.float64)
        params = tf.convert_to_tensor(params_np)
        # Ai, Ci
        param_idxs = np.array([
            [0, 3],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2]], dtype=np.int32)

        rlj = ReferenceLeonnardJonesEnergy(params_np, param_idxs, self.exclusions, self.box)
        ref_nrg = rlj.openmm_energy(x0)

        box_ph = tf.placeholder(shape=(3), dtype=tf.float64)

        tlj = LeonnardJones(params, param_idxs, exclusions)
        test_nrg_op = tlj.energy(x0, box_ph)

        sess = tf.Session()
        np.testing.assert_almost_equal(ref_nrg, sess.run(test_nrg_op, feed_dict={box_ph: self.box}))

        dEdbox = tf.gradients(test_nrg_op, box_ph)
        box_grads_val = sess.run(dEdbox, feed_dict={box_ph: self.box})
        assert not np.any(np.isnan(box_grads_val))

    def test_reference_ewald_electrostatic(self):

        x0 = self.x0
        exclusions = self.exclusions
        box = self.box

        params = np.array([1.3, 0.3], dtype=np.float64)
        params_tf = tf.convert_to_tensor(params)
        param_idxs = np.array([0, 1, 1, 1, 1], dtype=np.int32)

        kmax = 10

        ref = ReferenceEwaldEnergy(params, param_idxs, exclusions, box, kmax)
        ref_recip_nrg = ref.reference_reciprocal_energy(x0)
        omm_recip_nrg = ref.openmm_reciprocal_energy(x0)

        box_ph = tf.placeholder(shape=(3), dtype=tf.float64)

        esf = Electrostatic(params_tf, param_idxs, exclusions, kmax)
        x_ph = tf.placeholder(shape=(5, 3), dtype=np.float64)
        test_recip_nrg_op = esf.reciprocal_energy(x_ph, box_ph)

        sess = tf.Session()

        # reciprocal
        np.testing.assert_almost_equal(ref_recip_nrg, omm_recip_nrg)
        np.testing.assert_almost_equal(ref_recip_nrg, sess.run(test_recip_nrg_op, feed_dict={x_ph: x0, box_ph: self.box}))

        # direct and exclusions
        omm_direct_nrg, omm_exc_nrg = ref.openmm_direct_and_exclusion_energy(x0)
        test_direct_nrg_op, test_exc_nrg_op = esf.direct_and_exclusion_energy(x_ph, box)

        np.testing.assert_almost_equal(omm_direct_nrg, sess.run(test_direct_nrg_op, feed_dict={x_ph: x0, box_ph: self.box}))
        np.testing.assert_almost_equal(omm_exc_nrg, sess.run(test_exc_nrg_op, feed_dict={x_ph: x0, box_ph: self.box}))

        # self
        omm_self_nrg = ref.openmm_self_energy(x0)
        test_self_nrg_op = esf.self_energy(x0)
        np.testing.assert_almost_equal(omm_self_nrg, sess.run(test_self_nrg_op, feed_dict={x_ph: x0, box_ph: self.box}))

        total_nrg = esf.energy(x_ph, box_ph)

        grads, hessians, mixed = derivatives.compute_ghm(total_nrg, x_ph, [params_tf])

        g,h,m = sess.run([grads, hessians, mixed], feed_dict={x_ph: x0, box_ph: self.box})
        assert not np.any(np.isnan(g))
        assert not np.any(np.isnan(h))
        assert not np.any(np.isnan(m))

        box_grads_op = tf.gradients(total_nrg, box_ph)

        assert not np.any(np.isnan(m))
        box_grads_val = sess.run(box_grads_op, feed_dict={x_ph: x0, box_ph: self.box})
        assert not np.any(np.isnan(box_grads_val))

if __name__ == "__main__":
    unittest.main()


