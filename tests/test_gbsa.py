import unittest
import numpy as np
import tensorflow as tf

from timemachine.functionals.gbsa import GBSAOBC

class ReferenceGBSAOBCEnergy():
    """
    Taken from ReferenceObc.cpp in OpenMM
    """
    def __init__(self,
        params, # 
        param_idxs, # (N, 3), (q, r, s)
        dielectricOffset=0.009,
        cutoffDistance=2.0,
        alphaObc=1.0,
        betaObc=0.8,
        gammaObc=4.85,
        soluteDielectric=1.0,
        solventDielectric=78.3,
        electricConstant=-69.467728):

        self.params = params
        self.param_idxs = param_idxs 

        self.dielectricOffset = dielectricOffset
        self.cutoffDistance = cutoffDistance
        self.alphaObc = alphaObc
        self.betaObc = betaObc
        self.gammaObc = gammaObc
        self.soluteDielectric = soluteDielectric
        self.solventDielectric = solventDielectric
        self.electricConstant = electricConstant

    def openmm_born_radii(self, conf):
        num_atoms = conf.shape[0]

        atomicRadii = self.params[self.param_idxs[:, 1]]
        scaledRadiusFactor = self.params[self.param_idxs[:, 2]]

        dielectricOffset = self.dielectricOffset
        alphaObc = self.alphaObc
        betaObc = self.betaObc
        gammaObc = self.gammaObc

        bornRadii = np.zeros(shape=(num_atoms,))

        # (ytz): how the fuck am I supposed to tensorflow this?
        for a_idx in range(num_atoms):
            radiusI = atomicRadii[a_idx]
            oRI = radiusI - dielectricOffset
            radiusIInverse = 1.0/oRI
            summ = 0.0

            for b_idx in range(num_atoms):

                if a_idx != b_idx:

                    # vary depending on PBCs
                    r = np.linalg.norm(conf[a_idx] - conf[b_idx])

                    if self.cutoffDistance is not None and r > self.cutoffDistance:
                        continue

                    oRJ = atomicRadii[b_idx] - dielectricOffset
                    sRJ = oRJ * scaledRadiusFactor[b_idx]
                    rSRJ = r + sRJ

                    # conditional that we can use a boolean mask for?
                    if oRI < rSRJ:
                        rInverse = 1/r
                        rfs = np.abs(r - sRJ)

                        # this is just a #max
                        if oRI > rfs:
                            l_ij = oRI
                        else:
                            l_ij = rfs

                        l_ij = 1/l_ij

                        u_ij = 1/rSRJ

                        l_ij2 = l_ij * l_ij
                        u_ij2 = u_ij * u_ij

                        ratio = np.log((u_ij/l_ij))

                        term = l_ij - u_ij + 0.25*r*(u_ij2 - l_ij2)  + (0.5*rInverse*ratio) + (0.25*sRJ*sRJ*rInverse)*(l_ij2 - u_ij2);

                        # fix later 
                        if oRI < (sRJ - r):
                            assert 0
                            term += 2.0*(radiusIInverse - l_ij);

                        summ += term;
                    else:
                        print("skipping", a_idx, b_idx)

            summ *= 0.5*oRI
            sum2 = summ*summ
            sum3 = summ*sum2

            tanhSum = np.tanh(alphaObc*summ - betaObc*sum2 + gammaObc*sum3)
            bornRadii[a_idx] = 1.0/(1.0/oRI - tanhSum/radiusI)

        return bornRadii

    def openmm_energy(self, conf):
        num_atoms = conf.shape[0]

        cutoffDistance = self.cutoffDistance

        charges = self.params[self.param_idxs[:, 0]]

        if self.soluteDielectric != 0.0 and self.solventDielectric != 0.0:
            prefactor = 2.0 * self.electricConstant * (1.0/self.soluteDielectric) - 1/self.solventDielectric
        else:
            prefactor = 0.0

        bornRadii = self.openmm_born_radii(conf)

        total_nrg = 0

        for a_idx in range(num_atoms):
            partialChargeI = prefactor * charges[a_idx]

            for b_idx in range(num_atoms):
                partialChargeJ = charges[b_idx]

                r2 = np.sum(np.power(conf[a_idx] - conf[b_idx], 2), axis=-1)
                alpha2_ij = bornRadii[a_idx]*bornRadii[b_idx];
                D_ij = r2/(4.0*alpha2_ij);
                expTerm = np.exp(-D_ij);
                denominator2 = r2 + alpha2_ij*expTerm; 
                denominator = np.sqrt(denominator2); 
                Gpol = (partialChargeI*partialChargeJ)/denominator;           
                energy = Gpol;

                if a_idx != b_idx:
                    if self.cutoffDistance is not None:
                        energy -= (partialChargeI * partialChargeJ)/cutoffDistance
                else:
                    energy *= 0.5
                total_nrg += energy

        return total_nrg


class TestGBSA(unittest.TestCase):

    def test_gbsa(self):

        masses = np.array([6.0, 1.0, 1.0, 1.0, 1.0])
        x0 = np.array([
            [ 0.0637,   0.0126,   0.2203],
            [ 1.0573,  -0.2011,   1.2864],
            [ 2.3928,   1.2209,  -0.2230],
            [-0.6891,   1.6983,   0.0780],
            [-0.6312,  -1.6261,  -0.2601]
        ], dtype=np.float64)

        params = np.array([
            .1984, .115, .85, # H
            0.0221, .19, .72  # C
        ])

        param_idxs = np.array([
            [3, 4, 5],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
        ])

        ref_nrg = ReferenceGBSAOBCEnergy(
            params,
            param_idxs,
        )

        ref_radii = ref_nrg.openmm_born_radii(x0)
        nrg = GBSAOBC(params, param_idxs)

        x_ph = tf.placeholder(shape=(5, 3), dtype=np.float64)

        test_radii_op = nrg.compute_born_radii(x_ph)
        test_grad_op = tf.gradients(test_radii_op, x_ph)
        test_hess_op = tf.hessians(test_radii_op, x_ph)

        sess = tf.Session()
        sess.run(tf.initializers.global_variables())

        test_radii_val, test_grad_val, test_hess_val = sess.run([test_radii_op, test_grad_op, test_hess_op], feed_dict={x_ph: x0})
        np.testing.assert_array_almost_equal(ref_radii, test_radii_val, decimal=13)

        assert not np.any(np.isnan(test_grad_val))
        assert not np.any(np.isnan(test_hess_val))

        ref_nrg = ref_nrg.openmm_energy(x0)
        print(ref_nrg)


if __name__ == '__main__':
    unittest.main()
