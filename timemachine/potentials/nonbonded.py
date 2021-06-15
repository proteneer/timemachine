import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as np
from jax.scipy.special import erfc
from jax.ops import index_update, index
from timemachine.potentials.jax_utils import distance_on_pairs, distance, convert_to_4d, get_all_pairs_indices


def switch_fn(dij, cutoff):
    return np.power(np.cos((np.pi * np.power(dij, 8)) / (2 * cutoff)), 2)


"""
TODO:
* refactor Jax nonbonded to accept specific interacting particle pairs,
    given by arrays particles_i, particles_j where len(particles_i) == len(particles_j)
    rather than a single collection of particles.

    Intent:
        * be able to work with pre-defined neighbor lists
        * be able to evaluate batches of interactions during parallel monte carlo moves
            (e.g. where particles_i is a large collection of trial positions for a displaced particle
            and particles_j are all the other "frozen" coordinates)
        * be able to evaluate just the alchemical interactions
    Will require:
        * refactoring distance(x, box) function
* refactor nonbonded_v3 to output "per atom" energies, so that a collection of trial moves can be scored in a batch
    for the MC use case above
* refactor nonbonded_v3 to accept 4d offset w directly, rather than hard-coding that w must be computed using
    lamb, lambda_plane_idxs, lambda_offset_idxs
        Intent:
            * protocol optimization
* restore nonbonded_v3's JIT-ability -- the branch in validate_coulomb_cutoff makes it JIT-unfriendly!
* add utility function using np.triu_indices to get all-pairs interactions,
    without while avoiding need for dij keep_mask
* point of uncertainty:
    will making this change have any performance impact?
* possibly reduce the number of arguments?
    (currently 10)
* possibly make the signatures more "type-stable"?
    (for example, if conf.shape[1] == 4, we use a code path where 3 required arguments are ignored)
"""

from typing import NamedTuple, Optional

Array = np.array


class Params(NamedTuple):
    sig: Array
    eps: Array
    charges: Array


class Particles(NamedTuple):
    coords: Array
    params: Params


def lennard_jones(dij, sig_ij, eps_ij):
    sig6 = (sig_ij / dij) ** 6
    sig12 = sig6 ** 2

    return 4 * eps_ij * (sig12 - sig6)


def compute_nonbonded_terms(particles_i: Particles, particles_j: Particles,
                            box: Array, beta: float, cutoff: Optional[float] = None):

    # distances and cutoff
    dij = distance_on_pairs(particles_i.coords, particles_j.coords, box)
    if cutoff is None:
        cutoff = np.inf
    keep_mask = dij <= cutoff

    # Lennard-Jones
    sig_ij = particles_i.params.sig + particles_j.params.sig
    eps_ij = particles_i.params.eps * particles_j.params.eps
    eps_ij = np.where(keep_mask, eps_ij, 0)
    lj = lennard_jones(dij, sig_ij, eps_ij)

    # Coulomb
    qij = particles_i.params.charges * particles_j.params.charges
    # funny enough lim_{x->0} erfc(x)/x = 0
    coulomb = qij * erfc(beta * dij) / dij
    coulomb = np.where(keep_mask, coulomb, 0)

    return lj, coulomb


def nonbonded(
        conf,
        params,
        box,
        lamb,
        charge_rescale_mask,
        lj_rescale_mask,
        beta,
        cutoff,
        lambda_plane_idxs,
        lambda_offset_idxs,
        runtime_validate=False,
):
    """See docstring of nonbonded_v3 for more details"""

    N = conf.shape[0]

    if conf.shape[-1] == 3:
        conf = convert_to_4d(conf, lamb, lambda_plane_idxs, lambda_offset_idxs, cutoff)

    # make 4th dimension of box large enough so its roughly aperiodic
    if box is not None:
        box_4d = np.eye(4) * 1000
        box_4d = index_update(box_4d, index[:3, :3], box)
    box = box_4d

    charges = params[:, 0]
    sig = params[:, 1]
    eps = params[:, 2]

    inds_i, inds_j = get_all_pairs_indices(N)
    # n_interactions = len(inds_i)

    particles_i = Particles(
        coords=conf[inds_i],
        params=Params(sig=sig[inds_i], eps=eps[inds_i], charges=charges[inds_i])
    )
    particles_j = Particles(
        coords=conf[inds_j],
        params=Params(sig=sig[inds_j], eps=eps[inds_j], charges=charges[inds_j])
    )

    lj, coulomb = compute_nonbonded_terms(particles_i, particles_j, box, beta, cutoff)

    if (cutoff is not None) and runtime_validate:
        validate_coulomb_cutoff(cutoff, beta, threshold=1e-2)

    eij_total = lj * lj_rescale_mask[inds_i, inds_j] + coulomb * charge_rescale_mask[inds_i, inds_j]

    return np.sum(eij_total)

def nonbonded_v3(
    conf,
    params,
    box,
    lamb,
    charge_rescale_mask,
    lj_rescale_mask,
    beta,
    cutoff,
    lambda_plane_idxs,
    lambda_offset_idxs):
    """Lennard-Jones + Coulomb, with a few important twists:
    * distances are computed in 4D, controlled by lambda, lambda_plane_idxs, lambda_offset_idxs
    * each pairwise LJ and Coulomb term can be multiplied by an adjustable rescale_mask parameter
    * Coulomb terms are multiplied by erfc(beta * distance)
    Parameters
    ----------
    conf : (N, 3) or (N, 4) np.array
        3D or 4D coordinates
        if 3D, will be converted to 4D using (x,y,z) -> (x,y,z,w)
            where w = cutoff * (lambda_plane_idxs + lambda_offset_idxs * lamb)
    params : (N, 3) np.array
        columns [charges, sigmas, epsilons], one row per particle
    box : Optional 3x3 np.array
    lamb : float
    charge_rescale_mask : (N, N) np.array
        the Coulomb contribution of pair (i,j) will be multiplied by charge_rescale_mask[i,j]
    lj_rescale_mask : (N, N) np.array
        the Lennard-Jones contribution of pair (i,j) will be multiplied by lj_rescale_mask[i,j]
    beta : float
        the charge product q_ij will be multiplied by erfc(beta*d_ij)
    cutoff : Optional float
        a pair of particles (i,j) will be considered non-interacting if the distance d_ij
        between their 4D coordinates exceeds cutoff
    lambda_plane_idxs : Optional (N,) np.array
    lambda_offset_idxs : Optional (N,) np.array
    Returns
    -------
    energy : float
    References
    ----------
    * Rodinger, Howell, Pomès, 2005, J. Chem. Phys. "Absolute free energy calculations by thermodynamic integration in four spatial
        dimensions" https://aip.scitation.org/doi/abs/10.1063/1.1946750
    * Darden, York, Pedersen, 1993, J. Chem. Phys. "Particle mesh Ewald: An N log(N) method for Ewald sums in large
    systems" https://aip.scitation.org/doi/abs/10.1063/1.470117
        * Coulomb interactions are treated using the direct-space contribution from eq 2
    """

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

    eps_i = np.expand_dims(eps, 0)
    eps_j = np.expand_dims(eps, 1)

    eps_ij = eps_i * eps_j

    dij = distance(conf, box)

    keep_mask = np.ones((N,N)) - np.eye(N)
    keep_mask = np.where(eps_ij != 0, keep_mask, 0)

    if cutoff is not None:
        validate_coulomb_cutoff(cutoff, beta, threshold=1e-2)
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
    keep_mask = 1 - np.eye(N)
    qij = np.where(keep_mask, qij, 0)
    dij = np.where(keep_mask, dij, 0)

    # funny enough lim_{x->0} erfc(x)/x = 0
    eij_charge = np.where(keep_mask, qij*erfc(beta*dij)*inv_dij, 0) # zero out diagonals
    if cutoff is not None:
        eij_charge = np.where(dij > cutoff, 0, eij_charge)

    eij_total = (eij_lj*lj_rescale_mask + eij_charge*charge_rescale_mask)

    return np.sum(eij_total/2)

def validate_coulomb_cutoff(cutoff=1.0, beta=2.0, threshold=1e-2):
    """check whether f(r) = erfc(beta * r) <= threshold at r = cutoff
    following https://github.com/proteneer/timemachine/pull/424#discussion_r629678467"""
    if erfc(beta * cutoff) > threshold:
        print(UserWarning(f"erfc(beta * cutoff) = {erfc(beta * cutoff)} > threshold = {threshold}"))
