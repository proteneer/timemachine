import jax
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from attic.modules.potentials import shape
from timemachine.fe.system import minimize_scipy
from timemachine.fe.topology import DualTopology, get_romol_conf
from timemachine.ff import Forcefield


def get_shape_params(sig_eps):
    # Estimate the effective radius of each atom in the molecule given
    # the charge (unused), sigma, epsilon, and w coordinate.

    # Rough outline of the method is
    # 1) Compute the r_min(sigma, w), as determined by the first, real, stationary point of U
    # 2) Compute the difference between U(0) - U(r_min) to determine a barrier
    # 3) If the barrier is lower than some threshold, we set radius = 0, otherwise, radius = r_min
    sigmas = sig_eps[:, 0] * 2
    epsilons = sig_eps[:, 1] ** 2
    epsilons = np.ones_like(epsilons) * 2.7  # bpkelley hack
    # generalization of rmin = 2**1/6 with a 4D decoupling parameter
    # (other solutions are imaginary)
    r_min = np.sqrt(2 ** (1 / 3) * sigmas**2)
    r_min_half = np.nan_to_num(r_min) / 12
    return np.stack([r_min_half, epsilons], axis=1)


from rdkit.Chem import SDWriter
from rdkit.Geometry import Point3D


def voodoo_aligner(mol_a, mol_b):
    """
    Given two molecules, re-generate conformations so that they're optimally aligned.
    """
    ff = Forcefield.load_default()
    dt = DualTopology(mol_a, mol_b, ff)
    vacuum_system = dt.setup_chiral_end_state()
    U_fn = vacuum_system.get_U_fn()
    sig_eps_a = get_shape_params(ff.lj_handle.parameterize(mol_a))

    sig_eps_a[7:, 1] = 0
    # print(sig_eps_a)
    # assert 0
    sig_eps_b = get_shape_params(ff.lj_handle.parameterize(mol_b))
    sig_eps_b[8:, 1] = 0

    print(sig_eps_a)
    print(sig_eps_b)

    x_a_initial = get_romol_conf(mol_a)
    num_atoms_a = len(x_a_initial)
    x_b_initial = get_romol_conf(mol_b)
    x_c = np.concatenate([x_a_initial, x_b_initial])

    def U_shape(x_combined):
        x_a = x_combined[:num_atoms_a]
        x_b = x_combined[num_atoms_a:]
        n_overlap = shape.normalized_overlap(x_a, sig_eps_a, x_b, sig_eps_b)
        print("N_overlap", n_overlap)
        # low overlap bad
        # high overlap good

        return -1000 * n_overlap

    @jax.jit
    def augmented_U(x_combined):
        return U_fn(x_combined) + U_shape(x_combined)

    x_c_final_conf = minimize_scipy(augmented_U, x_c)

    x_a_final_conf = x_c_final_conf[:num_atoms_a]
    x_b_final_conf = x_c_final_conf[num_atoms_a:]

    print(f"mol_A: U_internal {U_fn(x_c):.2f} -> {U_fn(x_c_final_conf):.2f}")
    print(f"mol_AB: U_shape {U_shape(x_c):.2f} -> {U_shape(x_c_final_conf):.2f}")

    mol_a.RemoveAllConformers()
    mol_b.RemoveAllConformers()

    final_conf = Chem.Conformer()
    for atom_idx, (x, y, z) in enumerate(x_a_initial):
        final_conf.SetAtomPosition(atom_idx, Point3D(x * 10, y * 10, z * 10))
    mol_a.AddConformer(final_conf, 0)
    final_conf = Chem.Conformer()
    for atom_idx, (x, y, z) in enumerate(x_a_final_conf):
        final_conf.SetAtomPosition(atom_idx, Point3D(x * 10, y * 10, z * 10))
    mol_a.AddConformer(final_conf, 1)
    mol_a.GetConformer(1)
    mol_a_writer = SDWriter("mol_a.sdf")
    mol_a_writer.write(mol_a, 0)
    mol_a_writer.write(mol_a, 1)
    mol_a_writer.close()

    final_conf = Chem.Conformer()
    for atom_idx, (x, y, z) in enumerate(x_b_initial):
        final_conf.SetAtomPosition(atom_idx, Point3D(x * 10, y * 10, z * 10))
    mol_b.AddConformer(final_conf, 0)
    final_conf = Chem.Conformer()
    for atom_idx, (x, y, z) in enumerate(x_b_final_conf):
        final_conf.SetAtomPosition(atom_idx, Point3D(x * 10, y * 10, z * 10))
    mol_b.AddConformer(final_conf, 1)
    mol_b.GetConformer(1)
    mol_b_writer = SDWriter("mol_b.sdf")
    mol_b_writer.write(mol_b, 0)
    mol_b_writer.write(mol_b, 1)
    mol_b_writer.close()

    assert 0


if __name__ == "__main__":
    mol_a = Chem.AddHs(Chem.MolFromSmiles("C1CCCCC1"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("C1CCCCC1C"))

    AllChem.EmbedMolecule(mol_a)
    AllChem.EmbedMolecule(mol_b)

    voodoo_aligner(mol_a, mol_b)
