import numpy as onp
import jax.numpy as np
from jax.scipy.special import erf, erfc

from timemachine.constants import ONE_4PI_EPS0
from timemachine.potentials.jax_utils import delta_r, distance


def lennard_jones(conf, params, box, param_idxs, scale_matrix, cutoff=None,
    lamb=1.0, alpha=0.5, n=1, m=1):
    """
    Implements a non-periodic LJ612 potential using the Lorentz−Berthelot combining
    rules, where sig_ij = (sig_i + sig_j)/2 and eps_ij = sqrt(eps_i * eps_j). The
    softcore potential is defined by the equations in:

    http://www.alchemistry.org/wiki/Constructing_a_Pathway_of_Intermediate_States

    Parameters
    ----------
    conf: shape [num_atoms, 3] np.array
        atomic coordinates

    params: shape [num_params,] np.array
        unique parameters

    box: shape [3, 3] np.array
        periodic boundary vectors, if not None

    param_idxs: shape [num_atoms, 2] np.array
        each tuple (sig, eps) is used as part of the combining rules

    scale_matrix: shape [num_atoms, num_atoms] np.array
        scale mask denoting how we should scale interaction e[i,j].
        The elements should be between [0, 1]. If e[i,j] is 1 then the interaction
        is fully included, 0 implies it is discarded.

    cutoff: float
        Whether or not we apply cutoffs to the system. Any interactions
        greater than cutoff is fully discarded.
    
    lamb: soft-core float
        lambda value

    alpha: soft-core float
        soft-core scale factor

    n: soft-core integer (or float)
        prefactor lambda exponential

    m: soft-core integer (or float)
        complement (1-lambda) exponential

    """
    sig = params[param_idxs[:, 0]]
    eps = params[param_idxs[:, 1]]

    sig_i = np.expand_dims(sig, 0)
    sig_j = np.expand_dims(sig, 1)
    sig_ij = (sig_i + sig_j)/2
    sig_ij_raw = sig_ij

    eps_i = np.expand_dims(eps, 0)
    eps_j = np.expand_dims(eps, 1)
    eps_ij = scale_matrix * np.sqrt(eps_i * eps_j)

    eps_ij_raw = eps_ij

    ri = np.expand_dims(conf, 0)
    rj = np.expand_dims(conf, 1)

    d_ij = distance(ri, rj, box)

    if cutoff is not None:
        eps_ij = np.where(d_ij < cutoff, eps_ij, np.zeros_like(eps_ij))

    keep_mask = scale_matrix > 0

    # (ytz): this avoids a nan in the gradient in both jax and tensorflow
    d_ij = np.where(keep_mask, d_ij, np.zeros_like(d_ij))
    eps_ij = np.where(keep_mask, eps_ij, np.zeros_like(eps_ij))

    ds_ij = d_ij/sig_ij
    inner = alpha*np.power(1-lamb, m) + np.power(ds_ij, 6)
    energy = 4*eps_ij*np.power(lamb, n) * (np.power(inner, -2) - np.power(inner, -1))
    energy = np.where(keep_mask, energy, np.zeros_like(energy))

    # divide by two to deal with symmetry
    return np.sum(energy)/2


def pairwise_energy(conf, box, charges, cutoff):
    """
    Numerically stable implementation of the pairwise term:
    
    eij = qi*qj/dij

    """
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

    if cutoff is not None:
        eij = np.where(dij > cutoff, np.zeros_like(eij), eij)

    return eij

def electrostatics(conf, params, box, param_idxs, scale_matrix, cutoff=None, alpha=None, kmax=None):
    """
    Compute the electrostatic potential: sum_ij qi*qj/dij

    Parameters
    ----------
    conf: shape [num_atoms, 3] np.array
        atomic coordinates

    params: shape [num_params,] np.array
        unique parameters

    box: shape [3, 3] np.array
        periodic boundary vectors, if not None then Ewald summation is used.

    param_idxs: shape [num_atoms, 2] np.array
        each tuple (sig, eps) is used as part of the combining rules

    scale_matrix: shape [num_atoms, num_atoms] np.array
        scale mask denoting how we should scale interaction e[i,j].
        The elements should be between [0, 1]. If e[i,j] is 1 then the interaction
        is fully included, 0 implies it is discarded.

    cutoff: float
        must be less than half the periodic boundary condition for each dim

    alpha: float
        alpha term controlling the erf adjustment

    kmax: int
        number of images by which we tile out reciprocal space.

    """
    charges = params[param_idxs]

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

        return ewald_energy(conf, box, charges, scale_matrix, cutoff, alpha, kmax)

    else:    
        # non periodic electrostatics is straightforward.
        # note that we do not support reaction field approximations.
        eij = scale_matrix*pairwise_energy(conf, box, charges, cutoff)

        return ONE_4PI_EPS0*np.sum(eij)/2


def self_energy(conf, charges, alpha):
    return np.sum(ONE_4PI_EPS0 * np.power(charges, 2) * alpha/np.sqrt(np.pi))


def ewald_energy(conf, box, charges, scale_matrix, cutoff, alpha, kmax):
    eij = pairwise_energy(conf, box, charges, cutoff)

    assert cutoff is not None

    # 1. Assume scale matrix is not used at all (no exceptions, no exclusions)
    # 1a. Direct Space
    eij_direct = eij * erfc(alpha*eij)
    eij_direct = ONE_4PI_EPS0*np.sum(eij_direct)/2

    # 1b. Reciprocal Space
    eij_recip = reciprocal_energy(conf, box, charges, alpha, kmax)

    # 2. Remove over estimated scale matrix contribution scaled by erf
    eij_offset = (1-scale_matrix) * eij
    eij_offset *= erf(alpha*eij_offset)
    eij_offset = ONE_4PI_EPS0*np.sum(eij_offset)/2

    return eij_direct + eij_recip - eij_offset - self_energy(conf, charges, alpha)

def reciprocal_energy(conf, box, charges, alpha, kmax):

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
