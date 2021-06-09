import jax.numpy as np
from jax.scipy.special import erfc
from jax.ops import index_update, index
from timemachine.potentials.jax_utils import distance, convert_to_4d, get_all_pairs_indices


def switch_fn(dij, cutoff):
    return np.power(np.cos((np.pi*np.power(dij, 8))/(2*cutoff)), 2)

"""
TODO:
* refactor Jax nonbonded to accept specific interacting particle pairs,
    given by arrays particles_i, particles_j where len(particles_i) == len(particles_j)
    rather than a single collection of particles.
    
    on top of this, we could define two convenience functions for getting the list of interacting particles, emulating
    cdist and pdist from scipy.spatial.distance
    >>> inds_i, inds_j = all_pairs(mol)
    >>> particles_i, particles_j = mol[inds_i], mol[inds_j]
    or
    >>> (inds_i, inds_j) = all_pairs(mol_a, mol_b)
    >>> particles_i, particles_j = mol_a[particles_i], mol_b[particles_j]
    
    
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

    inds_i, inds_j = get_all_pairs_indices(N)
    sig_ij = sig[inds_i] + sig[inds_j]

    eps_ij = eps[inds_i] * eps[inds_j]

    dij = distance(conf, box)

    if cutoff is not None:
        validate_coulomb_cutoff(cutoff, beta, threshold=1e-2)
        eps_ij = np.where(dij < cutoff, eps_ij, 0)


    inv_dij = 1/dij

    sig2 = sig_ij*inv_dij
    sig2 *= sig2
    sig6 = sig2*sig2*sig2

    eij_lj = 4*eps_ij*(sig6-1.0)*sig6
    qij = charges[inds_i] * charges[inds_j]

    # funny enough lim_{x->0} erfc(x)/x = 0
    eij_charge = qij*erfc(beta*dij)*inv_dij
    if cutoff is not None:
        eij_charge = np.where(dij > cutoff, 0, eij_charge)

    eij_total = (eij_lj*lj_rescale_mask + eij_charge*charge_rescale_mask)

    return np.sum(eij_total)


def validate_coulomb_cutoff(cutoff=1.0, beta=2.0, threshold=1e-2):
    """check whether f(r) = erfc(beta * r) <= threshold at r = cutoff
    following https://github.com/proteneer/timemachine/pull/424#discussion_r629678467"""
    if erfc(beta * cutoff) > threshold:
        print(UserWarning(f"erfc(beta * cutoff) = {erfc(beta * cutoff)} > threshold = {threshold}"))
