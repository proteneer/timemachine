import tensorflow as tf
from timemachine.functionals import Energy

class PeriodicTorsion(Energy):

    def __init__(self,
        params,
        torsion_idxs,
        param_idxs,
        precision=tf.float64):
        """
        This implements a periodic torsional potential expanded out into three terms:

        V(a) = k0*(1+cos(1 * a - t0)) + k1*(1+cos(2 * a - t1)) + k2*(1+cos(3 * a - t2))

        Parameters:
        -----------
        params: list of tf.Variables
            an opaque array of parameters used by param_idxs for indexing into

        torsion: [num_torsions, 4] np.array
            each element (a, b, c, d) is a torsion in the conformation. The torsion is defined
            as the angle of the plane defined by the three bond vectors a-b, b-c, c-d. 

        param_idxs: [num_torsions, 6] np.array
            each element (k0_idx, k1_idx, k2_idx, t0_idx, t1_idx, t2_idx) maps into params for angle constants and ideal angles

        """
        self.params = params
        self.torsion_idxs = torsion_idxs
        self.param_idxs = param_idxs

    @staticmethod
    def get_signed_angle(ci, cj, ck, cl):
        """
        The torsion angle between two planes should be periodic but not
        necessarily symmetric. We use an identical but numerically stable arctan2
        implementation as opposed to the OpenMM energy function to avoid a
        singularity when the angle is zero.
        """

        # Taken from the wikipedia arctan2 implementation:
        # https://en.wikipedia.org/wiki/Dihedral_angle

        rij = ci - cj
        rkj = ck - cj
        rkl = ck - cl

        n1 = tf.cross(rij, rkj)
        n2 = tf.cross(rkj, rkl)

        lhs = tf.norm(n1, axis=-1)
        rhs = tf.norm(n2, axis=-1)
        bot = lhs * rhs

        y = tf.reduce_sum(tf.multiply(tf.cross(n1, n2), rkj/tf.norm(rkj, axis=-1)), axis=-1)
        x = tf.reduce_sum(tf.multiply(n1, n2), -1)

        return tf.atan2(y, x)

    def energy(self, conf):

        ci = tf.gather(conf, self.torsion_idxs[:, 0])
        cj = tf.gather(conf, self.torsion_idxs[:, 1])
        ck = tf.gather(conf, self.torsion_idxs[:, 2])
        cl = tf.gather(conf, self.torsion_idxs[:, 3])

        k0s = tf.gather(self.params, self.param_idxs[:, 0])
        k1s = tf.gather(self.params, self.param_idxs[:, 1])
        k2s = tf.gather(self.params, self.param_idxs[:, 2])
        t0s = tf.gather(self.params, self.param_idxs[:, 3])
        t1s = tf.gather(self.params, self.param_idxs[:, 4])
        t2s = tf.gather(self.params, self.param_idxs[:, 5])

        angle = self.get_signed_angle(ci, cj, ck, cl)

        e0 = k0s*(1+tf.cos(1 * angle - t0s))
        e1 = k1s*(1+tf.cos(2 * angle - t1s))
        e2 = k2s*(1+tf.cos(3 * angle - t2s))
        return tf.reduce_sum(e0+e1+e2, axis=-1)

class HarmonicAngle(Energy):

    def __init__(self,
        params,
        angle_idxs,
        param_idxs,
        precision=tf.float64):
        """
        This implements a harmonic angle potential: V(t) = k*(t - t0)^2. 

        Parameters:
        -----------
        params: list of tf.Variables
            an opaque array of parameters used by param_idxs for indexing into

        angle_idxs: [num_angles, 3] np.array
            each element (a, b, c) is a unique angle in the conformation. The angle is defined
            as between the two bond vectors a-b and c-b

        param_idxs: [num_angles, 2] np.array
            each element (k_idx, t_idx) maps into params for angle constants and ideal angles

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

        # (ytz): 1.0 nans, 0.975 nans but 0.98 is okay? (wtf?)
        # we really need another functional form for this
        cos_angles = 0.98*(top/bot)
        angle = tf.acos(cos_angles)

        # (ytz): we used the squared version so that we make this energy being strictly positive
        energies = kas/2*tf.pow(angle - a0s, 2)
        return tf.reduce_sum(energies, -1)  # reduce over all angles


class HarmonicBond(Energy):

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

    def energy(self, conf):    
        ci = tf.gather(conf, self.bond_idxs[:, 0])
        cj = tf.gather(conf, self.bond_idxs[:, 1])
        dij = tf.norm(ci - cj, axis=-1) # don't ever use norm, just always do 2x and less r0*r0 instead
        kbs = tf.gather(self.params, self.param_idxs[:, 0])
        r0s = tf.gather(self.params, self.param_idxs[:, 1])

        # (ytz): we used the squared version so that we make this energy being strictly positive
        energy = tf.reduce_sum(kbs/2*tf.pow(dij - r0s, 2.0))
        return energy
