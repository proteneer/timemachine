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
        gammaObc=4.85):

        self.params = params
        self.param_idxs = param_idxs 

        self.dielectricOffset = dielectricOffset
        self.cutoffDistance = cutoffDistance
        self.alphaObc = alphaObc
        self.betaObc = betaObc
        self.gammaObc = gammaObc

    def compute_born_radii(self, conf):

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

        oRI = atomicRadii - dielectricOffset
        oRJ = oRI
        sRJ = oRJ * scaledRadiusFactor

        # print(d_ij.shape, sRJ.shape)

        rSRJ = d_ij + sRJ

        diff = oRI - rSRJ

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


        # bornRadii = np.zeros(shape=(num_atoms,))