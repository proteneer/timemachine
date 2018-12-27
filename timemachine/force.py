import tensorflow as tf

class ConservativeForce():
    """
    A conservative force has an integral equal to a scalar-valued energy function.
    """


    def get_params(self):
        raise NotImplementedError("Abstract base class")

    def energy(self, conf):
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
        params: list of tf.Variable
            an opaque array of parameters used by kb_idxs and r0_idxs for indexing

        bonds: [num_bonds, 2] np.array
            each element (src, dst) is a unique bond in the conformation

        param_idxs: [num_bonds, 2] np.array
            each element (k_idx, r_idx) maps into params for bond constants and ideal lengths

        """
        self.params = params
        self.bond_idxs = bond_idxs
        self.param_idxs = param_idxs




        # todo: cache

        # bond_coords = tf.gather_nd(geometries, gidxs) # batch_size, num_angles, 3, 3

        # ci = bond_coords[:, :, 1, :]
        # cj = bond_coords[:, :, 0, :]
        # cij = cj - ci
        # dij = tf.norm(cij, axis=-1)

        # energies = (self.kb/2)*tf.pow(dij - self.r0, 2)

        # return tf.reduce_sum(energies, -1) # reduce over atoms

        # self.kbs = []
        # self.r0s = []

        # bond_type = [(6.0, 5.0)]

        # self.kb = tf.get_variable("kb", shape=tuple(), dtype=precision, initializer=tf.constant_initializer(10.0))
        # self.r0 = tf.get_variable("r0", shape=tuple(), dtype=precision, initializer=tf.constant_initializer(1.2))
        # self.b0 = tf.get_variable("b0", shape=tuple(), dtype=precision, initializer=tf.constant_initializer(0.0))

    def get_params(self):
        return self.params

    def energy(self, conf):

        ci = tf.gather(conf, self.bond_idxs[:, 0])
        cj = tf.gather(conf, self.bond_idxs[:, 1])

        dij = tf.norm(ci - cj, axis=-1)

        kbs = tf.gather(self.params, self.param_idxs[:, 0])
        r0s = tf.gather(self.params, self.param_idxs[:, 1])

        energy = tf.reduce_sum(kbs*tf.pow(dij - r0s, 2.0))
        return energy
