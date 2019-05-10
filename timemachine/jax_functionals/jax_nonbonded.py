import numpy as onp
import jax.numpy as np
from jax.scipy.special import erf, erfc

from timemachine.constants import ONE_4PI_EPS0
from timemachine.jax_functionals import Energy
from timemachine.jax_functionals.jax_utils import delta_r, distance

class LennardJones(Energy):

    def __init__(self, param_idxs, scale_matrix, cutoff=None):
        """
        Implements a non-periodic LJ612 potential using the Lorentz−Berthelot combining
        rules, where sig_ij = (sig_i + sig_j)/2 and eps_ij = sqrt(eps_i * eps_j).

        Parameters
        ----------
        param_idxs: (N,2)
            each tuple (sig, eps) is used as part of the combining rules

        scale_matrix: (N, N)
            scale mask denoting how we should scale interaction e[i,j].
            The elements should be between [0, 1]. If e[i,j] is 1 then the interaction
            is fully included, 0 implies it is discarded.

        cutoff: float
            Whether or not we apply cutoffs to the system. Any interactions
            greater than cutoff is fully discarded.

        """
        self.param_idxs = param_idxs
        self.scale_matrix = scale_matrix
        self.cutoff = cutoff # this probably shouldn't be used
        super().__init__()

    def energy(self, conf, params, box=None):
        """
        
        """
        sig = params[self.param_idxs[:, 0]]
        eps = params[self.param_idxs[:, 1]]

        sig_i = np.expand_dims(sig, 0)
        sig_j = np.expand_dims(sig, 1)
        sig_ij = (sig_i + sig_j)/2
        sig_ij_raw = sig_ij

        eps_i = np.expand_dims(eps, 0)
        eps_j = np.expand_dims(eps, 1)
        eps_ij = self.scale_matrix * np.sqrt(eps_i * eps_j)

        eps_ij_raw = eps_ij

        ri = np.expand_dims(conf, 0)
        rj = np.expand_dims(conf, 1)

        dij = distance(ri, rj, box)

        if self.cutoff is not None:
            eps_ij = np.where(dij < self.cutoff, eps_ij, np.zeros_like(eps_ij))

        keep_mask = self.scale_matrix > 0

        # (ytz): this avoids a nan in the gradient in both jax and tensorflow
        sig_ij = np.where(keep_mask, sig_ij, np.zeros_like(sig_ij))
        eps_ij = np.where(keep_mask, eps_ij, np.zeros_like(eps_ij))

        sig2 = sig_ij/dij
        sig2 *= sig2
        sig6 = sig2*sig2*sig2

        energy = 4*eps_ij*(sig6-1.0)*sig6
        energy = np.where(keep_mask, energy, np.zeros_like(energy))

        # divide by two to deal with symmetry
        return np.sum(energy, axis=-1)/2


class Electrostatics(Energy):

    def __init__(self, param_idxs, scale_matrix):
        """
        Implements electrostatic potential based on coloumb's law. For an in-depth theory guide,
        please refer to:

        http://docs.openmm.org/latest/userguide/theory.html#coulomb-interaction-with-ewald-summation

        Parameters
        ----------
        param_idxs: (N,) tf.Tensor
            indices into params for each atom corresponding to the charge

        scale_matrix: (N, N) tf.Tensor
            how much we scale each interaction by. Note that we follow OpenMM's convention,
            if the scale_matrix[i,j] is exactly 1.0 and the cutoff is not None, then we apply
            the crf correction. The scale matrices should be set to zero for 1-2 and 1-3 ixns.

        """
        self.param_idxs = param_idxs # length N
        self.num_atoms = len(self.param_idxs)
        self.scale_matrix = scale_matrix
        super().__init__()

    def energy(self, conf, params, box=None, cutoff=None, alpha=None, kmax=None):
        """
        Parameters
        ----------

        confs: np.array
            an Nx3 set of conformations

        params: np.array
            parameters used by param_idxs to index into the charges

        box: np.array
            3x3 set of vectors, where box is [[a_x, 0, 0], [b_x, b_y, 0], [c_x, c_y, c_z]]
        
        cutoff: float
            must be less than half the periodic boundary condition for each dim

        alpha: float
            alpha term controlling the erf adjustment

        kmax: int
            number of images by which we tile out reciprocal space.

        """
        charges = params[self.param_idxs]
        charges = np.reshape(charges, (1, -1))

        # if we use periodic boundary conditions, then the following three parameters
        # must be set in order for Ewald to make sense.
        if box is not None:
            # note that periodic boundary conditions are subject to the following
            # convention and constraints:
            # http://docs.openmm.org/latest/userguide/theory.html#periodic-boundary-conditions

            box_lengths = np.linalg.norm(box, axis=-1)
            assert cutoff is not None and cutoff >= 0.00
            assert alpha is not None
            assert kmax is not None

            # this is an implicit assumption in the Ewald calculation. If it were any larger
            # then there may be more than N^2 number of interactions.
            if np.any(box_lengths < 2*cutoff):
                raise ValueError("Box lengths cannot be smaller than twice the cutoff.")

            return self.ewald_energy(conf, box, charges, cutoff, alpha, kmax)

        else:
            raise Exception("Box is not None")

    def self_energy(self, conf, charges, alpha):
        return np.sum(ONE_4PI_EPS0 * np.power(charges, 2) * alpha/np.sqrt(np.pi))

    def ewald_energy(self, conf, box, charges, cutoff, alpha, kmax):
        qi = np.expand_dims(charges, 0) # (1, N)
        qj = np.expand_dims(charges, 1) # (N, 1)
        qij = np.multiply(qi, qj)
        ri = np.expand_dims(conf, 0)
        rj = np.expand_dims(conf, 1)
        dij = distance(ri, rj, box)

        # (ytz): trick used to avoid nans in the diagonal due to the 1/dij term.
        keep_mask = 1 - np.eye(conf.shape[0])
        qij = np.where(keep_mask, qij, np.zeros_like(qij))
        dij = np.where(keep_mask, dij, np.zeros_like(dij))
        eij = np.where(keep_mask, qij/dij, np.zeros_like(dij)) # zero out diagonals

        assert cutoff is not None

        # 1. Assume scale matrix is not used at all (no exceptions, no exclusions)
        # 1a. Direct Space
        eij_direct = np.where(dij > cutoff, np.zeros_like(eij), eij)
        eij_direct *= erfc(alpha*eij_direct)
        eij_direct = ONE_4PI_EPS0*np.sum(eij_direct)/2

        # 1b. Reciprocal Space
        eij_recip = self.reciprocal_energy(conf, box, charges, alpha, kmax)

        # 2. Remove over estimated scale matrix contribution
        # 2a. Remove the diagonal elements again
        eij_offset = (1-self.scale_matrix) * eij
        eij_offset *= erf(alpha*eij_offset)
        eij_offset = ONE_4PI_EPS0*np.sum(eij_offset)/2

        return eij_direct + eij_recip - eij_offset - self.self_energy(conf, charges, alpha)

    def reciprocal_energy(self, conf, box, charges, alpha, kmax):

        assert kmax > 0
        assert box is not None
        assert alpha > 0

        recipBoxSize = (2*np.pi)/np.diag(box)

        mg = []
        lowry = 0
        lowrz = 1

        numRx, numRy, numRz = kmax, kmax, kmax

        for rx in range(numRx):
            for ry in range(lowry, numRy):
                for rz in range(lowrz, numRz):
                    mg.append([rx, ry, rz])
                    lowrz = 1 - numRz
                lowry = 1 - numRy

        mg = np.array(onp.array(mg))

        # lattice vectors
        ki = np.expand_dims(recipBoxSize, axis=0) * mg # [nk, 3]
        ri = np.expand_dims(conf, axis=0) # [1, N, 3]
        rik = np.sum(np.multiply(ri, np.expand_dims(ki, axis=1)), axis=-1) # [nk, N]
        real = np.cos(rik)
        imag = np.sin(rik)
        eikr = real + 1j*imag # [nk, N]
        qi = charges +0j
        Sk = np.sum(qi*eikr, axis=-1)  # [nk]
        n2Sk = np.power(np.abs(Sk), 2)
        k2 = np.sum(np.multiply(ki, ki), axis=-1) # [nk]
        factorEwald = -1/(4*alpha*alpha)
        ak = np.exp(k2*factorEwald)/k2 # [nk]
        nrg = np.sum(ak * n2Sk)
        # the following volume calculation assumes the reduced PBC convention consistent
        # with that of OpenMM
        recipCoeff = (ONE_4PI_EPS0*4*np.pi)/(box[0][0]*box[1][1]*box[2][2]) 

        return recipCoeff * nrg
