import numpy as np
import tensorflow as tf

def sorted_squared_distances(confs):
    """
    Compute an observed a sorted squared distance matrix.

    Warning: do *NOT* take a sqrt of the result as this results in
    NaNs when doing derivative calculations. This has to do with how sqrt
    works through autodiff in that you end up with a zero in the
    denominator.

    Parameters
    ----------
    confs: [B, N, 3]
        A batch of geometries.

    Returns
    -------
    tf.Tensor [N, N, B]
        An op representing sorted square distances

    """
    ri = tf.expand_dims(confs, 1) # [B, 1, N, 3]
    rj = tf.expand_dims(confs, 2) # [B, N, 1, 3]
    d2ij = tf.reduce_sum(tf.pow(ri-rj, 2), axis=-1)
    d2ij_t = tf.transpose(d2ij)
    sorted_d2ij = tf.contrib.framework.sort(d2ij_t, axis=-1)
    return sorted_d2ij

def vibrational_eigenvalues(conf, masses, energies):
    """
    Compute harmonic frequencies.

    Parameters
    ----------
    conf: tf.Tensor (N,3)
        Geometry

    masses: np.ndarray (N,)
        masses of each atom

    energies: timemachine.ConservativeForce
        Energies used to compute the hessian

    Returns
    -------
    list of tf.complex128
        Returns real and imaginary frequencies computed from
        the eigenvalues

    """
    hessians = []
    for e in energies:
        hessians.append(e.hessians(conf))
    net_hessians = tf.reduce_sum(tf.stack(hessians, axis=0), axis=0)
    # masses =  tf.tile(masses, [3])
    masses = np.repeat(masses, 3)
    reduced_mij = tf.sqrt(tf.expand_dims(masses, 0) * tf.expand_dims(masses, 1))
    net_hessians = tf.reshape(net_hessians, (conf.shape[0]*3, conf.shape[0]*3))
    net_hessians = net_hessians/reduced_mij
    eigenvalues = tf.linalg.eigvalsh(net_hessians)
    return eigenvalues
    # eigenvalues = tf.cast(eigenvalues, dtype=tf.complex128)

    print("DEBUG", tf.gradients(eigenvalues, net_hessians))
    return VIBRATIONAL_CONSTANT*tf.sqrt(eigenvalues)


def radius_of_gyration(confs, num_atoms):
    com = tf.reduce_mean(confs, -2, keep_dims=True)
    adj_xs = confs - com
    squared_norms = tf.reduce_sum(tf.multiply(adj_xs, adj_xs), axis=-1)
    ssn = tf.reduce_sum(squared_norms, -1)
    rg = ssn/(2*num_atoms)
    return rg

class Rg():
    """
    Radius of gyration
    """

    def __init__(self, n_bonds):
        self.n_bonds = n_bonds

    def obs(self, confs):
        com = tf.reduce_mean(confs, -2, keep_dims=True)
        adj_xs = confs - com
        squared_norms = tf.reduce_sum(tf.multiply(adj_xs, adj_xs), axis=-1)
        ssn = tf.reduce_sum(squared_norms, -1)
        rg = ssn/(2*self.n_bonds)
        return rg

class J3Coupling():
    """ 3-bond couplings """

    def __init__(self, mol_graph, A=7, B=-1, C=5):

        # https://www.ucl.ac.uk/nmr/NMR_lecture_notes/L3_3_97_web.pdf
        # default values for aliphatic hydrocarbons
        self.mol_graph = mol_graph
        self.n_bonds = len(self.mol_graph.get_bond_idxs())
        self.torsion_idxs = self.mol_graph.get_torsion_idxs()
        self.A = A
        self.B = B
        self.C = C

    def obs(self, confs):
        angles = energy_mod.dihedral_angle(
            confs,
            torsion_idxs_i=self.torsion_idxs[:, 0].tolist(),
            torsion_idxs_j=self.torsion_idxs[:, 1].tolist(),
            torsion_idxs_k=self.torsion_idxs[:, 2].tolist(),
            torsion_idxs_l=self.torsion_idxs[:, 3].tolist()
        )

        return tf.reduce_sum(self.A + self.B * tf.cos(angles) + self.C * tf.cos(2*angles), -1)/(2*self.n_bonds - 3) 
        