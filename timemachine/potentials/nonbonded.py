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

    if conf.shape[-1] == 3:
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

    dij = distance(conf, box)

    N = conf.shape[0]
    keep_mask = np.ones((N,N)) - np.eye(N)
    keep_mask = np.where(eps_ij != 0, keep_mask, 0)

    if cutoff is not None:
        eps_ij = np.where(dij < cutoff, eps_ij, 0)

    # (ytz): this avoids a nan in the gradient in both jax and tensorflow
    sig_ij = np.where(keep_mask, sig_ij, 0)
    eps_ij = np.where(keep_mask, eps_ij, 0)

    inv_dij = 1/dij
    inv_dij = np.where(np.eye(N), 0, inv_dij)

    sig2 = sig_ij*inv_dij
    sig2 *= sig2
    sig6 = sig2*sig2*sig2

    eij_lj = 4*eps_ij*(sig6-1.0)*sig6
    eij_lj = np.where(keep_mask, eij_lj, 0)

    qi = np.expand_dims(charges, 0) # (1, N)
    qj = np.expand_dims(charges, 1) # (N, 1)
    qij = np.multiply(qi, qj)

    # (ytz): trick used to avoid nans in the diagonal due to the 1/dij term.
    keep_mask = 1 - np.eye(conf.shape[0])
    qij = np.where(keep_mask, qij, 0)
    dij = np.where(keep_mask, dij, 0)

    # funny enough lim_{x->0} erfc(x)/x = 0
    eij_charge = np.where(keep_mask, qij*erfc(beta*dij)*inv_dij, 0) # zero out diagonals
    if cutoff is not None:
        eij_charge = np.where(dij > cutoff, 0, eij_charge)

    eij_total = (eij_lj*lj_rescale_mask + eij_charge*charge_rescale_mask)

    return np.sum(eij_total/2)
