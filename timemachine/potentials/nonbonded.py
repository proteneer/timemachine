import jax.numpy as np
from jax.ops import index, index_update
from jax.scipy.special import erfc

from timemachine.potentials.jax_utils import convert_to_4d, delta_r, distance, distance_on_pairs


def switch_fn(dij, cutoff):
    return np.power(np.cos((np.pi * np.power(dij, 8)) / (2 * cutoff)), 2)


from typing import Optional

Array = np.array


def lennard_jones(dij, sig_ij, eps_ij):
    """https://en.wikipedia.org/wiki/Lennard-Jones_potential"""
    sig6 = (sig_ij / dij) ** 6
    sig12 = sig6 ** 2

    return 4 * eps_ij * (sig12 - sig6)


def direct_space_pme(dij, qij, beta):
    """Direct-space contribution from eq 2 of:
    Darden, York, Pedersen, 1993, J. Chem. Phys.
    "Particle mesh Ewald: An N log(N) method for Ewald sums in large systems"
    https://aip.scitation.org/doi/abs/10.1063/1.470117
    """
    return qij * erfc(beta * dij) / dij


def nonbonded_block(xi, xj, box, params_i, params_j, beta, cutoff):
    """
    This is a modified version of nonbonded_v3 that computes a block of
    interactions between two sets of particles x_i and x_j. It is assumed that
    there are no exclusions between the two particle sets. Typical use cases
    include computing the interaction energy between the environment and a ligand.

    This is mainly used for testing, as it does not support 4D decoupling or
    alchemical semantics yet.

    Parameters
    ----------
    xi : (N,3) np.ndarray
        Coordinates
    xj : (N,3) np.ndarray
        Coordinates
    box : Optional 3x3 np.array
        Periodic boundary conditions
    beta : float
        the charge product q_ij will be multiplied by erfc(beta*d_ij)
    cutoff : Optional float
        a pair of particles (i,j) will be considered non-interacting if the distance d_ij
        between their 3D coordinates exceeds cutoff

    Returns
    -------
    scalar
        Interaction energy

    """
    ri = np.expand_dims(xi, axis=1)
    rj = np.expand_dims(xj, axis=0)
    d2ij = np.sum(np.power(delta_r(ri, rj, box), 2), axis=-1)
    dij = np.sqrt(d2ij)
    sig_i = np.expand_dims(params_i[:, 1], axis=1)
    sig_j = np.expand_dims(params_j[:, 1], axis=0)
    eps_i = np.expand_dims(params_i[:, 2], axis=1)
    eps_j = np.expand_dims(params_j[:, 2], axis=0)

    sig_ij = sig_i + sig_j
    eps_ij = eps_i * eps_j

    qi = np.expand_dims(params_i[:, 0], axis=1)
    qj = np.expand_dims(params_j[:, 0], axis=0)

    qij = np.multiply(qi, qj)

    es = direct_space_pme(dij, qij, beta)
    lj = lennard_jones(dij, sig_ij, eps_ij)

    nrg = np.where(dij > cutoff, 0, es + lj)
    return np.sum(nrg)


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
    lambda_offset_idxs,
    runtime_validate=True,
):
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
    runtime_validate: bool
        check whether beta is compatible with cutoff
        (if True, this function will currently not play nice with Jax JIT)
        TODO: is there a way to conditionally print a runtime warning inside
            of a Jax JIT-compiled function, without triggering a Jax ConcretizationTypeError?

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
    if runtime_validate:
        assert (charge_rescale_mask == charge_rescale_mask.T).all()
        assert (lj_rescale_mask == lj_rescale_mask.T).all()

    N = conf.shape[0]

    if conf.shape[-1] == 3:
        conf = convert_to_4d(conf, lamb, lambda_plane_idxs, lambda_offset_idxs, cutoff)

    # make 4th dimension of box large enough so its roughly aperiodic
    if box is not None:
        if box.shape[-1] == 3:
            box_4d = np.eye(4) * 1000
            box_4d = index_update(box_4d, index[:3, :3], box)
        else:
            box_4d = box
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

    keep_mask = np.ones((N, N)) - np.eye(N)
    keep_mask = np.where(eps_ij != 0, keep_mask, 0)

    if cutoff is not None:
        if runtime_validate:
            validate_coulomb_cutoff(cutoff, beta, threshold=1e-2)
        eps_ij = np.where(dij < cutoff, eps_ij, 0)

    # (ytz): this avoids a nan in the gradient in both jax and tensorflow
    sig_ij = np.where(keep_mask, sig_ij, 0)
    eps_ij = np.where(keep_mask, eps_ij, 0)

    inv_dij = 1 / dij
    inv_dij = np.where(np.eye(N), 0, inv_dij)

    sig2 = sig_ij * inv_dij
    sig2 *= sig2
    sig6 = sig2 * sig2 * sig2

    eij_lj = 4 * eps_ij * (sig6 - 1.0) * sig6
    eij_lj = np.where(keep_mask, eij_lj, 0)

    qi = np.expand_dims(charges, 0)  # (1, N)
    qj = np.expand_dims(charges, 1)  # (N, 1)
    qij = np.multiply(qi, qj)

    # (ytz): trick used to avoid nans in the diagonal due to the 1/dij term.
    keep_mask = 1 - np.eye(N)
    qij = np.where(keep_mask, qij, 0)
    dij = np.where(keep_mask, dij, 0)

    # funny enough lim_{x->0} erfc(x)/x = 0
    eij_charge = np.where(keep_mask, qij * erfc(beta * dij) * inv_dij, 0)  # zero out diagonals
    if cutoff is not None:
        eij_charge = np.where(dij > cutoff, 0, eij_charge)

    eij_total = eij_lj * lj_rescale_mask + eij_charge * charge_rescale_mask

    return np.sum(eij_total / 2)


def nonbonded_v3_on_specific_pairs(conf, params, box, inds_l, inds_r, beta: float, cutoff: Optional[float] = None):
    """See nonbonded_v3 docstring for more details

    Notes
    -----
    * Responsibility of caller to ensure pair indices (inds_l, inds_r) are complete.
        In case of parameter interpolation, more pairs need to be added.
    """

    # distances and cutoff
    dij = distance_on_pairs(conf[inds_l], conf[inds_r], box)
    if cutoff is None:
        cutoff = np.inf
    keep_mask = dij <= cutoff

    def apply_cutoff(x):
        return np.where(keep_mask, x, 0)

    charges, sig, eps = params.T

    # vdW by Lennard-Jones
    sig_ij = apply_cutoff(sig[inds_l] + sig[inds_r])
    eps_ij = apply_cutoff(eps[inds_l] * eps[inds_r])
    vdW = lennard_jones(dij, sig_ij, eps_ij)

    # Electrostatics by direct-space part of PME
    qij = apply_cutoff(charges[inds_l] * charges[inds_r])
    electrostatics = direct_space_pme(dij, qij, beta)

    return vdW, electrostatics


def validate_coulomb_cutoff(cutoff=1.0, beta=2.0, threshold=1e-2):
    """check whether f(r) = erfc(beta * r) <= threshold at r = cutoff
    following https://github.com/proteneer/timemachine/pull/424#discussion_r629678467"""
    if erfc(beta * cutoff) > threshold:
        print(UserWarning(f"erfc(beta * cutoff) = {erfc(beta * cutoff)} > threshold = {threshold}"))
