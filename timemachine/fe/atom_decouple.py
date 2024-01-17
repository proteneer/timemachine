# Utilities for improving the nonbonded decoupling algorithm by incorporating effective
# volume as a cheap surrogate

import numpy as np

from timemachine.constants import DEFAULT_KT
from timemachine.fe.single_topology import SingleTopology
from timemachine.fe.utils import get_romol_conf
from timemachine.ff import Forcefield


def estimate_volume(coords, radii, n_samples):
    # Estimate the volume of a set of possibly intersecting spheres using Monte Carlo
    # The method random samples in side a box bounded by the min and max of the coordinates,
    # padded by the maximum radius. Random points are then placed inside the box and flagged
    # to be within the molecule or outside of the molecule.

    # Generate the padded bounding box
    max_radius = np.amax(radii)
    fbl_corner = np.amin(coords, axis=0) - max_radius
    btr_corner = np.amax(coords, axis=0) + max_radius
    n_atoms = coords.shape[0]

    # Bulk generate points in the box
    random_point_in_box = np.random.uniform(fbl_corner, btr_corner, size=(n_samples, 1, 3))

    # Detect if these points are within any atom's radius in the molecule
    coords = coords.reshape(1, n_atoms, 3)
    dists = np.linalg.norm(coords - random_point_in_box, axis=-1)
    radii = radii.reshape(1, n_atoms)
    predicates = dists < radii
    n_true = np.sum(np.any(predicates, axis=-1))

    # Compute volume of cube
    box_size = btr_corner - fbl_corner
    box_volume = np.prod(box_size)

    # Volume of molecule = volume of cube * fractional occupancy
    return box_volume * (n_true / n_samples)


def estimate_radii(qljws):
    # Estimate the effective radius of each atom in the molecule given
    # the charge (unused), sigma, epsilon, and w coordinate.

    # Rough outline of the method is
    # 1) Compute the r_min(sigma, w), as determined by the first, real, stationary point of U
    # 2) Compute the difference between U(0) - U(r_min) to determine a barrier
    # 3) If the barrier is lower than some threshold, we set radius = 0, otherwise, radius = r_min
    sigmas = qljws[:, 1] * 2
    epsilons = qljws[:, 2] ** 2
    ws = qljws[:, 3]
    # generalization of rmin = 2**1/6 with a 4D decoupling parameter
    # (other solutions are imaginary)
    r_min = np.sqrt(2 ** (1 / 3) * sigmas**2 - ws**2)
    r_min = np.nan_to_num(r_min)
    beta = 1 / DEFAULT_KT

    def u_fn(x):
        sr = sigmas / np.sqrt(x**2 + ws**2)
        return beta * 4 * epsilons * (sr**12 - sr**6)

    barrier_threshold = 5
    r_min = np.where((u_fn(0) - u_fn(r_min)) < barrier_threshold, 0, r_min)

    return r_min


def estimate_volumes_along_schedule(mol_a, mol_b, core, ff, lamb_schedule):
    """
    Estimate volumes as we alchemically morph mol_a into mol_b along the lamb_schedule
    Parameters
    ----------
    mol_a: Chem.Mol
    mol_b: Chem.Mol
    core: np.ndarray
    ff: Forcefield
    lamb_schedule: np.array of size N
    Returns
    -------
    list of floats of size N
        Returns a list of volumes each value of lambda
    """
    ff = Forcefield.load_default()
    st = SingleTopology(mol_a, mol_b, core, ff)
    vols = []

    conf_a = get_romol_conf(mol_a)
    conf_b = get_romol_conf(mol_b)

    for lamb in lamb_schedule:
        # note, "meet in the middle"
        conf_c = st.combine_confs(conf_a, conf_b, lamb)
        cutoff = 1.2  # doesn't really too matter much except for scaling of w
        qljws = st._get_guest_params(st.ff.q_handle_solv, st.ff.lj_handle_solv, lamb, cutoff=cutoff)
        qljws = np.array(qljws)
        radii = estimate_radii(qljws)
        vol = estimate_volume(conf_c, radii, n_samples=1000000)
        vols.append(vol)

    return vols


def estimate_avg_volumes_along_schedule(qljws_by_state, trajs_by_state):
    """
    Estimate volumes as we alchemically morph mol_a into mol_b along the lamb_schedule
    Parameters
    ----------
    trajs_by_state: num_windows x num
    Returns
    -------
    list of floats of size N
        Returns a list of volumes each value of lambda
    """
    assert len(qljws_by_state) == len(trajs_by_state)
    avgs = []

    for qljws, frames in zip(qljws_by_state, trajs_by_state):
        radii = estimate_radii(qljws)
        vols = []
        for conf in frames:
            vols.append(estimate_volume(conf, radii, n_samples=1000000))

        avgs.append(np.mean(vols))

    return avgs
