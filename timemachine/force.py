import tensorflow as tf

class ConservativeForce():
    """
    A conservative force has an integral equal to a scalar-valued energy function.
    """


    def get_params(self):
        return self.params
        # raise NotImplementedError("Abstract base class")

    def energy(self, conf):
        """
        Computes a scalar energy given a geometry.
        """
        raise NotImplementedError("Abstract base class")

    def gradients(self, conf):
        energy = self.energy(conf)
        gradients = tf.gradients(energy, conf)[0]
        return gradients

    def hessians(self, conf):
        energy = self.energy(conf)
        return tf.hessians(energy, conf)[0]


    def mixed_partials(self, conf):
        # the order here matters since we're computing gradients and not
        # jacobians, otherwise they get reduced.
        mps = []
        energy = self.energy(conf)
        for p in self.get_params():
            mps.extend(tf.gradients(tf.gradients(energy, p), conf))
        return mps

class HarmonicAngleForce(ConservativeForce):

    def __init__(self,
        params,
        angle_idxs,
        param_idxs,
        precision=tf.float64):
        """
        This implements a cosine angle potential: V(t) = K(cos(t - t0))^2. 


        Parameters:
        -----------
        params: list of tf.Variables
            an opaque array of parameters used by param_idxs for indexing into

        angle_idxs: [num_angles, 3] np.array
            each element (a, b, c) is a unique bond in the conformation. The angle is defined
            as between the two vectors a-b and c-b

        param_idxs: [num_angles, 2] np.array
            each element (k_idx, t_idx) maps into params for angle constants and ideal lengths

        """
        self.params = params
        self.angle_idxs = angle_idxs
        self.param_idxs = param_idxs

    def energy(self, conf):
        """
        Compute the harmonic bond energy given a collection of molecules.
        """
        cj = tf.gather(conf, self.angle_idxs[:, 0])
        ci = tf.gather(conf, self.angle_idxs[:, 1])
        ck = tf.gather(conf, self.angle_idxs[:, 2])

        kas = tf.gather(self.params, self.param_idxs[:, 0])
        a0s = tf.gather(self.params, self.param_idxs[:, 1])

        vij = cj - ci
        vik = ck - ci

        top = tf.reduce_sum(tf.multiply(vij, vik), -1)
        bot = tf.norm(vij, axis=-1)*tf.norm(vik, axis=-1)

         # 0.975 is to prevent numerical issues for molecules like HC#N
         # we should never have zero angles.
        cos_angles = 0.98*(top/bot)
        angle = tf.acos(cos_angles)

        energies = kas/2*tf.pow(angle - a0s, 2)
        return tf.reduce_sum(energies, -1)  # reduce over all angles


class HarmonicBondForce(ConservativeForce):


    def __init__(self,
        params,
        bond_idxs,
        param_idxs,
        precision=tf.float64):
        """
        Implements a harmonic bond force of the form k(|a-b|-x)^2

        Parameters:
        -----------
        params: list of tf.Variables
            an opaque array of parameters used by param_idxs for indexing into

        bond_idxs: [num_bonds, 2] np.array
            each element (src, dst) is a unique bond in the conformation

        param_idxs: [num_bonds, 2] np.array
            each element (k_idx, r_idx) maps into params for bond constants and ideal lengths

        """
        self.params = params
        self.bond_idxs = bond_idxs
        self.param_idxs = param_idxs

    # def get_params(self):
        # return self.params

    def energy(self, conf):    
        ci = tf.gather(conf, self.bond_idxs[:, 0])
        cj = tf.gather(conf, self.bond_idxs[:, 1])

        dij = tf.norm(ci - cj, axis=-1)

        kbs = tf.gather(self.params, self.param_idxs[:, 0])
        r0s = tf.gather(self.params, self.param_idxs[:, 1])

        energy = tf.reduce_sum(kbs*tf.pow(dij - r0s, 2.0))
        return energy
