import jax.numpy as np

from timemachine.jax_functionals import Energy
from timemachine.jax_functionals.jax_utils import distance, delta_r

class HarmonicBond(Energy):

    def __init__(self,
        bond_idxs,
        param_idxs):
        """
        Implements a harmonic bond force of the form k(|a-b|-x)^2

        Parameters:
        -----------
        bond_idxs: [num_bonds, 2] np.array
            each element (src, dst) is a unique bond in the conformation

        param_idxs: [num_bonds, 2] np.array
            each element (k_idx, r_idx) maps into params for bond constants and ideal lengths

        """
        self.bond_idxs = bond_idxs
        self.param_idxs = param_idxs
        super().__init__()

    def energy(self, conf, params, box=None):   
        ci = conf[self.bond_idxs[:, 0]]
        cj = conf[self.bond_idxs[:, 1]]
        dij = distance(ci, cj, box)
        kbs = params[self.param_idxs[:, 0]]
        r0s = params[self.param_idxs[:, 1]]
        energy = np.sum(kbs/2 * np.power(dij - r0s, 2.0))

        return energy


class HarmonicAngle(Energy):

    def __init__(self,
        angle_idxs,
        param_idxs,
        cos_angles=True):
        """
        This implements a harmonic angle potential: V(t) = k*(t - t0)^2 or V(t) = k*(cos(t)-cos(t0))^2

        Parameters:
        -----------
        angle_idxs: [num_angles, 3] np.array
            each element (a, b, c) is a unique angle in the conformation. atom b is defined
            to be the middle atom.

        param_idxs: [num_angles, 2] np.array
            each element (k_idx, t_idx) maps into params for angle constants and ideal angles

        cos_angles: True (default)
            if True, then this instead implements V(t) = k*(cos(t)-cos(t0))^2. This is far more
            numerically stable when the angle is pi.

        """
        self.angle_idxs = angle_idxs
        self.param_idxs = param_idxs
        self.cos_angles = cos_angles
        super().__init__()

    def energy(self, conf, params, box=None):
        """
        Compute the harmonic bond energy given a collection of molecules.
        """

        ci = conf[self.angle_idxs[:, 0]]
        cj = conf[self.angle_idxs[:, 1]]
        ck = conf[self.angle_idxs[:, 2]]

        kas = params[self.param_idxs[:, 0]]
        a0s = params[self.param_idxs[:, 1]]

        vij = delta_r(ci, cj, box)
        vjk = delta_r(ck, cj, box)

        top = np.sum(np.multiply(vij, vjk), -1)
        bot = np.linalg.norm(vij, axis=-1)*np.linalg.norm(vjk, axis=-1)

        cos_angles = top/bot

        # (ytz): we used the squared version so that we make this energy being strictly positive
        if self.cos_angles:
            energies = kas/2*np.power(cos_angles - np.cos(a0s), 2)
        else:
            angle = np.arccos(cos_angles)
            energies = kas/2*np.power(angle - a0s, 2)
        return np.sum(energies, -1)  # reduce over all angles


class PeriodicTorsion(Energy):

    def __init__(self,
        torsion_idxs,
        param_idxs):
        """
        This implements a periodic torsional potential expanded out into three terms:

        V(a) = k0*(1+cos(1 * a - t0)) + k1*(1+cos(2 * a - t1)) + k2*(1+cos(3 * a - t2))

        Parameters:
        -----------
        torsion_idxs: [num_torsions, 4] np.array
            each element (a, b, c, d) is a torsion of four atoms, defined as
            as the angle of the plane defined by the three bond vectors a-b, b-c, c-d. 

        param_idxs: [num_torsions, 6] np.array
            each element (k, phase, periodicity) maps into params for angle constants and ideal angles

        """
        self.torsion_idxs = torsion_idxs
        self.param_idxs = param_idxs
        super().__init__()

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

        rij = delta_r(cj, ci)
        rkj = delta_r(cj, ck)
        rkl = delta_r(cl, ck)

        n1 = np.cross(rij, rkj)
        n2 = np.cross(rkj, rkl)

        lhs = np.linalg.norm(n1, axis=-1)
        rhs = np.linalg.norm(n2, axis=-1)
        bot = lhs * rhs

        y = np.sum(np.multiply(np.cross(n1, n2), rkj/np.linalg.norm(rkj, axis=-1, keepdims=True)), axis=-1)
        x = np.sum(np.multiply(n1, n2), -1)

        return np.arctan2(y, x)

    def angles(self, conf):
        ci = conf[self.torsion_idxs[:, 0]]
        cj = conf[self.torsion_idxs[:, 1]]
        ck = conf[self.torsion_idxs[:, 2]]
        cl = conf[self.torsion_idxs[:, 3]]

        angle = self.get_signed_angle(ci, cj, ck, cl)
        return angle

    def energy(self, conf, params, box=None):
        """
        Compute the torsional energy.
        """
        ci = conf[self.torsion_idxs[:, 0]]
        cj = conf[self.torsion_idxs[:, 1]]
        ck = conf[self.torsion_idxs[:, 2]]
        cl = conf[self.torsion_idxs[:, 3]]

        ks = params[self.param_idxs[:, 0]]
        phase = params[self.param_idxs[:, 1]]
        period = params[self.param_idxs[:, 2]]
        angle = self.get_signed_angle(ci, cj, ck, cl)
        nrg = ks*(1+np.cos(period * angle - phase))
        return np.sum(nrg, axis=-1)