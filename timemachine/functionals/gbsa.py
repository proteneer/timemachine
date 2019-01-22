import numpy as np
import tensorflow as tf

from timemachine.functionals import Energy

class GBSAOBC(Energy):

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

    def compute_born_radii(self, conf):

        num_atoms = conf.shape[0]
        atomicRadii = self.params[self.param_idxs[:, 1]]
        scaledRadiusFactor = self.params[self.param_idxs[:, 2]]

        dielectricOffset = self.dielectricOffset
        alphaObc = self.alphaObc
        betaObc = self.betaObc
        gammaObc = self.gammaObc

        r_i = tf.expand_dims(conf, axis=0)
        r_j = tf.expand_dims(conf, axis=1)
        r_ij = r_i - r_j

        d_ij = tf.norm(r_ij, axis=-1)
        d2_ij = tf.reduce_sum(tf.pow(r_ij, 2), axis=-1)

        # (ytz): This is a trick used to remove the diagonal elements that would
        # otherwise introduce nans into the calculation.
        mask = tf.ones(shape=[num_atoms, num_atoms], dtype=tf.int32)
        mask = tf.cast(mask - tf.matrix_band_part(mask, 0, 0), dtype=tf.bool)
        d2_ij = tf.where(mask, d2_ij, tf.zeros_like(d2_ij))

        d_ij = tf.sqrt(d2_ij)

        oRI = atomicRadii - dielectricOffset
        oRJ = oRI
        sRJ = oRJ * scaledRadiusFactor
        rSRJ = d_ij + sRJ

        mask0 = tf.less(d_ij, self.cutoffDistance)
        mask1 = tf.less(oRI, rSRJ)
        mask_final = tf.logical_and(mask0, mask1)

        d_ij_inv = 1/d_ij # has NaNs
        rfs = tf.abs(d_ij - sRJ)
        l_ij = tf.maximum(oRI, rfs)
        l_ij = 1/l_ij
        u_ij = 1/rSRJ

        l_ij2 = l_ij * l_ij
        u_ij2 = u_ij * u_ij

        ratio = tf.log(u_ij/l_ij)
        term = l_ij - u_ij + 0.25*d_ij*(u_ij2 - l_ij2)  + (0.5*d_ij_inv*ratio) + (0.25*sRJ*sRJ*d_ij_inv)*(l_ij2 - u_ij2);
        term_masked = tf.where(mask_final, term, tf.zeros_like(term))

        summ = tf.reduce_sum(term_masked, axis=-1) # need to keep one of the dimension

        summ *= 0.5 * oRI
        sum2 = summ*summ
        sum3 = summ*sum2
        tanhSum = tf.tanh(alphaObc*summ - betaObc*sum2 + gammaObc*sum3)

        bornRadii = 1.0/(1.0/oRI - tanhSum/atomicRadii)

        return bornRadii

    def energy(self, conf):

        num_atoms = conf.shape[0]

        if self.soluteDielectric != 0.0 and self.solventDielectric != 0.0:
            prefactor = 2.0 * self.electricConstant * (1.0/self.soluteDielectric) - 1/self.solventDielectric
        else:
            prefactor = 0.0

        bornRadii = self.compute_born_radii(conf)

        charges = self.params[self.param_idxs[:, 0]]

        r_i = tf.expand_dims(conf, axis=0)
        r_j = tf.expand_dims(conf, axis=1)

        q_i = tf.expand_dims(charges, axis=0)
        q_j = tf.expand_dims(charges, axis=1)
        q_ij = q_i*q_j

        br_i = tf.expand_dims(bornRadii, axis=0)
        br_j = tf.expand_dims(bornRadii, axis=1)

        r2 = tf.reduce_sum(tf.pow(r_i - r_j, 2), axis=-1)
        alpha2_ij = br_i * br_j
        D_ij = r2/(4.0*alpha2_ij)
        expTerm = tf.exp(-D_ij)
        denom2 = r2 + alpha2_ij*expTerm
        denom = tf.sqrt(denom2)
        pq_ij = prefactor*q_ij
        Gpol = pq_ij/denom
        energy = Gpol

        # separate in diag and off diag parts
        ones_mask = tf.ones(shape=[num_atoms, num_atoms], dtype=tf.int32)
        on_diag_mask = tf.matrix_band_part(ones_mask, 0, 0)
        off_diag_mask = tf.cast(ones_mask - on_diag_mask, dtype=tf.bool)

        on_diag_E = tf.boolean_mask(energy, on_diag_mask)
        on_diag_E *= 0.5

        off_diag_E = tf.boolean_mask(energy, off_diag_mask)

        if self.cutoffDistance is not None:
            off_diag_pq_ij = tf.boolean_mask(pq_ij, off_diag_mask)
            off_diag_E -= off_diag_pq_ij/self.cutoffDistance

        tot_e = tf.reduce_sum(on_diag_E) + tf.reduce_sum(off_diag_E)
        return tot_e
