import numpy as np
import tensorflow as tf
import utils

class SortedDistances():

    """ 
    Given K conformations of Nx3 atoms, we can form a set of
    [N,N,K] distance matrices. The last dimension is then sorted
    independently from smallest to longest, forming a marginalized
    probability distribution. This roughly corresponds to what's
    actually observed in short-range NOEs.
    """

    def __init__(self, mask):
        self.ignore_mask = mask



    def obs(self, confs):

        dij = utils.generate_distance_matrix(confs, self.ignore_mask) # [batch_size, (num_atoms-1)*(num_atoms)/2 upper right off diagonal]
        batch_size = dij.shape[0]
        dij_t = tf.transpose(dij) # shape [(num_atoms-1)*(num_atoms)/2, batch_size]

        sort_idxs = np.argsort(dij_t.numpy(), axis=-1) # (ytz) why do we need x.numpy() here?
        
        gather_idxs = []
        for idx, a in enumerate(sort_idxs):
            row = []
            for bidx in range(batch_size):
                row.append((idx, a[bidx]))
            gather_idxs.append(row)
        gather_idxs = np.array(gather_idxs, dtype=np.int32)
        sorted_dij = tf.gather_nd(dij_t, gather_idxs)
        return sorted_dij

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
        