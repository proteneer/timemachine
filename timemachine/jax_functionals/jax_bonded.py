import jax
import jax.numpy as np

from timemachine.jax_functionals import Energy


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

    def energy(self, conf, params):    
        ci = conf[self.bond_idxs[:, 0]]
        cj = conf[self.bond_idxs[:, 1]]
        dij = np.linalg.norm(ci - cj, axis=-1) # don't ever use norm, just always do 2x and less r0*r0 instead
        kbs = params[self.param_idxs[:, 0]]
        r0s = params[self.param_idxs[:, 1]]

        # (ytz): we used the squared version so that we make this energy being strictly positive
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
            numerically stable if your system can become linear.

        """
        self.angle_idxs = angle_idxs
        self.param_idxs = param_idxs
        self.cos_angles = cos_angles
        super().__init__()

    def energy(self, conf, params):
        """
        Compute the harmonic bond energy given a collection of molecules.
        """
        ci = conf[self.angle_idxs[:, 0]]
        cj = conf[self.angle_idxs[:, 1]]
        ck = conf[self.angle_idxs[:, 2]]

        kas = params[self.param_idxs[:, 0]]
        a0s = params[self.param_idxs[:, 1]]

        vij = cj - ci
        vjk = cj - ck

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
