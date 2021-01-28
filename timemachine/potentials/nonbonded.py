import numpy as onp
import jax.numpy as np
from jax.scipy.special import erf, erfc
from jax.ops import index_update, index

from timemachine.constants import ONE_4PI_EPS0
from timemachine.potentials.jax_utils import delta_r, distance, convert_to_4d




def switch_fn(dij, cutoff):
    return np.power(np.cos((np.pi*np.power(dij, 8))/(2*cutoff)), 2)


def nonbonded(
    conf,
    lamb,
    charge_params,
    lj_params,
    exclusion_idxs,
    charge_scales,
    lj_scales,
    cutoff,
    lambda_plane_idxs,
    lambda_offset_idxs):

    # assert box is None

    conf_4d = convert_to_4d(conf, lamb, lambda_plane_idxs, lambda_offset_idxs, cutoff)

    lj = lennard_jones(conf_4d, lj_params, cutoff)
    lj_exc = lennard_jones_exclusion(conf_4d, lj_params, exclusion_idxs, lj_scales, cutoff)
    es = simple_energy(conf_4d, charge_params, exclusion_idxs, charge_scales, cutoff)

    return lj - lj_exc + es


def electrostatics_v2(
    conf,
    charge_params,
    box,
    lamb,
    exclusion_idxs,
    charge_scales,
    beta,
    cutoff,
    lambda_offset_idxs):

    # assert box is None

    conf_4d = convert_to_4d(conf, lamb, lambda_offset_idxs)

    # print(conf_4d)
    if box is not None:
        box_4d = np.eye(4)*1000
        box_4d = index_update(box_4d, index[:3, :3], box)
    else:
        box_4d = None

    return simple_energy(conf_4d, box_4d, charge_params, exclusion_idxs, charge_scales, beta, cutoff)


def nonbonded_v2(
    conf,
    params,
    box,
    lamb,
    exclusion_idxs,
    scales,
    beta,
    cutoff,
    lambda_offset_idxs):

    # assert box is None

    conf_4d = convert_to_4d(conf, lamb, lambda_offset_idxs)

    # print(conf_4d)
    if box is not None:
        box_4d = np.eye(4)*1000
        box_4d = index_update(box_4d, index[:3, :3], box)
    else:
        box_4d = None

    charge_params = params[:, 0]
    lj_params = params[:, 1:]

    charge_scales = scales[:, 0]
    lj_scales = scales[:, 1]

    lj = lennard_jones(conf_4d, lj_params, box_4d, cutoff)
    lj_exc = lennard_jones_exclusion(conf_4d, lj_params, box_4d, exclusion_idxs, lj_scales, cutoff)
    es = simple_energy(conf_4d, box_4d, charge_params, exclusion_idxs, charge_scales, beta, cutoff)

    return lj - lj_exc + es

def lennard_jones_v2(
    conf,
    lj_params,
    box,
    lamb,
    exclusion_idxs,
    lj_scales,
    cutoff,
    lambda_plane_idxs,
    lambda_offset_idxs):

    conf_4d = convert_to_4d(conf, lamb, lambda_plane_idxs, lambda_offset_idxs, cutoff)
    box_4d = np.eye(4)*1000
    box_4d = index_update(box_4d, index[:3, :3], box)

    lj = lennard_jones(conf_4d, lj_params, box_4d, cutoff)
    lj_exc = lennard_jones_exclusion(conf_4d, lj_params, box_4d, exclusion_idxs, lj_scales, cutoff)

    return lj - lj_exc

def nonbonded_v3(
    conf,
    params,
    box,
    lamb,
    charge_rescale_mask,
    lj_rescale_mask,
    scales,
    beta,
    cutoff,
    lambda_plane_idxs,
    lambda_offset_idxs):
    
    N = conf.shape[0]

    conf = convert_to_4d(conf, lamb, lambda_plane_idxs, lambda_offset_idxs, cutoff)

    # make 4th dimension of box large enough so its roughly aperiodic
    if box is not None:
        box_4d = np.eye(4)*1000
        box_4d = index_update(box_4d, index[:3, :3], box)
    else:
        box_4d = None

    box = box_4d

    charges = params[:, 0]
    sig = params[:, 1]
    eps = params[:, 2]

    sig_i = np.expand_dims(sig, 0)
    sig_j = np.expand_dims(sig, 1)
    sig_ij = sig_i + sig_j
    sig_ij_raw = sig_ij

    eps_i = np.expand_dims(eps, 0)
    eps_j = np.expand_dims(eps, 1)

    eps_ij = eps_i * eps_j

    ri = np.expand_dims(conf, 0)
    rj = np.expand_dims(conf, 1)
    dij = distance(ri, rj, box)

    N = conf.shape[0]
    keep_mask = np.ones((N,N)) - np.eye(N)
    keep_mask = np.where(eps_ij != 0, keep_mask, 0)

    if cutoff is not None:
        eps_ij = np.where(dij < cutoff, eps_ij, np.zeros_like(eps_ij))

    # (ytz): this avoids a nan in the gradient in both jax and tensorflow
    sig_ij = np.where(keep_mask, sig_ij, np.zeros_like(sig_ij))
    eps_ij = np.where(keep_mask, eps_ij, np.zeros_like(eps_ij))

    sig2 = sig_ij/dij
    sig2 *= sig2
    sig6 = sig2*sig2*sig2

    eij_lj = 4*eps_ij*(sig6-1.0)*sig6
    eij_lj = np.where(keep_mask, eij_lj, np.zeros_like(eij_lj))

    qi = np.expand_dims(charges, 0) # (1, N)
    qj = np.expand_dims(charges, 1) # (N, 1)
    qij = np.multiply(qi, qj)

    # (ytz): trick used to avoid nans in the diagonal due to the 1/dij term.
    keep_mask = 1 - np.eye(conf.shape[0])
    qij = np.where(keep_mask, qij, np.zeros_like(qij))
    dij = np.where(keep_mask, dij, np.zeros_like(dij))

    # funny enough lim_{x->0} erfc(x)/x = 0
    eij_charge = np.where(keep_mask, qij*erfc(beta*dij)/dij, np.zeros_like(dij)) # zero out diagonals
    if cutoff is not None:
        eij_charge = np.where(dij > cutoff, np.zeros_like(eij_charge), eij_charge)

    eij_total = (eij_lj*lj_rescale_mask + eij_charge*charge_rescale_mask)

    return np.sum(eij_total/2)

def lennard_jones(conf, lj_params, box, cutoff):
    """
    Implements a non-periodic LJ612 potential using the Lorentz−Berthelot combining
    rules, where sig_ij = (sig_i + sig_j)/2 and eps_ij = sqrt(eps_i * eps_j).

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
    
    """
    # box = None
    # assert box is None

    sig = lj_params[:, 0]
    eps = lj_params[:, 1]

    sig_i = np.expand_dims(sig, 0)
    sig_j = np.expand_dims(sig, 1)
    sig_ij = (sig_i + sig_j)/2
    sig_ij_raw = sig_ij

    eps_i = np.expand_dims(eps, 0)
    eps_j = np.expand_dims(eps, 1)

    eps_ij = np.sqrt(eps_i * eps_j)

    ri = np.expand_dims(conf, 0)
    rj = np.expand_dims(conf, 1)
    # gi = np.expand_dims(groups, axis=0)
    # gj = np.expand_dims(groups, axis=1)
    # gij = np.bitwise_and(gi, gj) > 0

    # print(gij)

    # print("BOX", box)
    dij = distance(ri, rj, box)
    # print("DIJ", dij)

    N = conf.shape[0]
    keep_mask = np.ones((N,N)) - np.eye(N)
    keep_mask = np.where(eps_ij != 0, keep_mask, 0)

    if cutoff is not None:
        eps_ij = np.where(dij < cutoff, eps_ij, np.zeros_like(eps_ij))

    # (ytz): this avoids a nan in the gradient in both jax and tensorflow
    sig_ij = np.where(keep_mask, sig_ij, np.zeros_like(sig_ij))
    eps_ij = np.where(keep_mask, eps_ij, np.zeros_like(eps_ij))

    sig2 = sig_ij/dij
    sig2 *= sig2
    sig6 = sig2*sig2*sig2

    eij = 4*eps_ij*(sig6-1.0)*sig6


    # if cutoff is not None:
        # sw = switch_fn(dij, cutoff)
        # eij = eij*sw

    eij = np.where(keep_mask, eij, np.zeros_like(eij))

    # print("eps_ij", eps_ij)
    # print("sig_ij", sig_ij)


    return np.sum(eij/2)


# now we compute the exclusions
def lennard_jones_exclusion(conf, lj_params, box, exclusion_idxs, lj_scales, cutoff, groups=None):

    # box = None
    # assert box is None

    assert exclusion_idxs.shape[1] == 2
    # assert exclusion_idxs.shape[0] == conf.shape[0]
    assert exclusion_idxs.shape[0] == lj_scales.shape[0]

    src_idxs = exclusion_idxs[:, 0]
    dst_idxs = exclusion_idxs[:, 1]
    ri = conf[src_idxs]
    rj = conf[dst_idxs]

    dij = distance(ri, rj, box)

    sig_params = lj_params[:, 0] 
    sig_i = sig_params[src_idxs]
    sig_j = sig_params[dst_idxs]
    sig_ij = (sig_i + sig_j)/2

    eps_params = lj_params[:, 1] 
    eps_i = eps_params[src_idxs]
    eps_j = eps_params[dst_idxs]
    eps_ij = np.sqrt(eps_i * eps_j)
    eps_ij = np.where(eps_ij != 0, eps_ij, 0) # (ytz): avoids nans


    if cutoff is not None:
        eps_ij = np.where(dij < cutoff, eps_ij, np.zeros_like(eps_ij))

    sig2 = sig_ij/dij
    sig2 *= sig2
    sig6 = sig2*sig2*sig2

    scale_ij = lj_scales
    eij_exc = scale_ij*4*eps_ij*(sig6-1.0)*sig6

    if cutoff is not None:
        # sw = switch_fn(dij, cutoff)
        # eij_exc = eij_exc*sw
        eij_exc = np.where(dij > cutoff, np.zeros_like(eij_exc), eij_exc)
        eij_exc = np.where(src_idxs == dst_idxs, np.zeros_like(eij_exc), eij_exc)

    # the exclusion energy is not divided by two.
    return np.sum(eij_exc)


def simple_energy(conf, box, charge_params, exclusion_idxs, charge_scales, beta, cutoff):
    """
    Numerically stable implementation of the pairwise term:
    
    eij = qi*qj/dij

    """
    charges = charge_params
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

    # funny enough lim_{x->0} erfc(x)/x = 0
    eij = np.where(keep_mask, qij*erfc(beta*dij)/dij, np.zeros_like(dij)) # zero out diagonals

    # print(dij)

    if cutoff is not None:
        # sw = switch_fn(dij, cutoff)
        # eij = eij*sw
        eij = np.where(dij > cutoff, np.zeros_like(eij), eij)

    src_idxs = exclusion_idxs[:, 0]
    dst_idxs = exclusion_idxs[:, 1]
    ri = conf[src_idxs]
    rj = conf[dst_idxs]
    dij = distance(ri, rj, box)

    qi = charges[src_idxs]
    qj = charges[dst_idxs]
    qij = np.multiply(qi, qj)

    scale_ij = charge_scales
    eij_exc = scale_ij*qij*erfc(beta*dij)/dij

    if cutoff is not None:
        # sw = switch_fn(dij, cutoff)
        # eij_exc = eij_exc*sw
        eij_exc = np.where(dij > cutoff, np.zeros_like(eij_exc), eij_exc)
        eij_exc = np.where(src_idxs == dst_idxs, np.zeros_like(eij_exc), eij_exc)

    return np.sum(eij/2) - np.sum(eij_exc)


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

    # neutralize the charge
    # todo: neutralize host charges separately from guest charges
    # print("WARNING: NUMBER OF ATOMS USED TO NEUTRALIZED SET EXPLICITLY FOR HOST aCD")
    # num_host_atoms = 126
    # host_charges = charges[:num_host_atoms]
    # guest_charges = charges[num_host_atoms:]
    # host_charges = host_charges - np.sum(host_charges)/host_charges.shape[0]
    # guest_charges = guest_charges - np.sum(guest_charges)/guest_charges.shape[0]
    # charges = np.concatenate([host_charges, guest_charges])

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
        nrg = ONE_4PI_EPS0*eij
        # onp.save("energy.npy", onp.asarray(nrg))
        nrg = np.sum(nrg/2)

        return nrg


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
