from jax.config import config; config.update("jax_enable_x64", True)
import jax
import functools

from timemachine.potentials import rigid_shape
from scipy.spatial.transform import Rotation
import rmsd
import numpy as np
import time

from rdkit import Chem
from rdkit.Chem import AllChem

def recenter(conf):
    """Return copy of conf with mean coordinates subtracted"""
    return conf - np.mean(conf, axis=0)

def get_conf(romol, idx):
    """Get the idx'th conformer of romol, in nanometers"""
    conformer = romol.GetConformer(idx)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)/10
    return recenter(guest_conf)

def test_rotation():
    x = np.random.rand(10, 3)
    q = Rotation.random(random_state=2020).as_quat()
    x_r = rigid_shape.rotate(x, q)
    assert rmsd.kabsch_rmsd(x_r, x) < 1e-8


def make_conformer(mol, conf_a, conf_b):
    mol.RemoveAllConformers()
    mol = Chem.CombineMols(mol, mol)
    cc = Chem.Conformer(mol.GetNumAtoms())
    conf = np.concatenate([conf_a, conf_b])
    conf *= 10
    for idx, pos in enumerate(np.asarray(conf)):
        cc.SetAtomPosition(idx, (float(pos[0]), float(pos[1]), float(pos[2])))
    mol.AddConformer(cc)

    return mol


def get_heavy_atom_idxs(mol):

    idxs = []
    for a_idx, a in enumerate(mol.GetAtoms()):
        if a.GetAtomicNum() > 1:
            idxs.append(a_idx)
    return np.array(idxs, dtype=np.int32)


def test_vjp():

    prefactor = 2.7  # unitless
    lamb = (4 * np.pi) / (3 * prefactor)  # unitless
    kappa = np.pi / (np.power(lamb, 2 / 3))  # unitless
    sigma = 0.15 # nanometers
    alpha = kappa / (sigma * sigma)
    # TODO: extract the preceding setup block and share between test_custom_op and test_volume_range?

    suppl = Chem.SDMolSupplier("tests/data/ligands_40.sdf", removeHs=False)
    mols = []
    for mol in suppl:
        mols.append(mol)

    ligand_a = mols[0]
    ligand_b = mols[2]

    ligand_a = Chem.AddHs(Chem.MolFromSmiles("C12C3C4C1C5C2C3C45"))
    ligand_b = Chem.AddHs(Chem.MolFromSmiles("C12C3C4C1C5C2C3C45"))
    AllChem.EmbedMolecule(ligand_a, randomSeed=2020)
    AllChem.EmbedMolecule(ligand_b, randomSeed=2021)

    coords_a = get_conf(ligand_a, idx=0)
    params_a = np.stack([
        np.zeros(ligand_a.GetNumAtoms()) + alpha,
        np.zeros(ligand_a.GetNumAtoms()) + prefactor,
    ], axis=1)

    coords_b = get_conf(ligand_b, idx=0)
    params_b = np.stack([
        np.zeros(ligand_b.GetNumAtoms()) + alpha,
        np.zeros(ligand_b.GetNumAtoms()) + prefactor,
    ], axis=1)

    # coords_a = coords_a[get_heavy_atom_idxs(ligand_a)]
    # params_a = params_a[get_heavy_atom_idxs(ligand_a)]
    # coords_b = coords_b[get_heavy_atom_idxs(ligand_b)]
    # params_b = params_b[get_heavy_atom_idxs(ligand_b)]

    coords_a = coords_a - np.mean(coords_a, axis=0)
    coords_b = coords_b - np.mean(coords_b, axis=0)

    # TEST HESSIAN
    # qi = Rotation.from_euler('z',25,degrees=True).as_quat() # needs to be transposed
    # qi = Rotation.from_euler('z',0,degrees=True).as_quat() # needs to be transposed

    writer = Chem.SDWriter('quat.sdf')

    trip_counts = []

    for seed in range(100):
        # seed = 361
        # print(seed)
        qi = Rotation.random(random_state=seed).as_quat()
        # qi = Rotation.from_euler('zyx', [(np.random.rand()-0.5)*90, (np.random.rand()-0.5)*90, (np.random.rand()-0.5)*90], degrees=True).as_quat()

        # switch to our convention for position of real component
        qi = np.array([qi[3], qi[0], qi[1], qi[2]])
        # print(qi)
        coords_r = rigid_shape.rotate(coords_b, qi)
        c_mol = Chem.CombineMols(ligand_a, ligand_b)
        # conf = make_conformer(c_mol, coords_a, coords_r)
        # writer.write(conf)

        # po = rigid_shape.q_opt(coords_a, coords_r, params_a, params_b)
        # q_final = rigid_shape.q_from_p(po)
        # conf = make_conformer(c_mol, coords_a, rigid_shape.rotate(coords_r, q_final))

        rot, count = rigid_shape.bfgs_minimize(coords_a, coords_r, params_a, params_b)
        trip_counts.append(count)
        print("done", rot, "count", count)
        # assert 0
        # print(rot, converged)
        # q_final = rigid_shape.q_from_p(po)
        conf = make_conformer(c_mol, coords_a, rigid_shape.rotate_euler(coords_r, rot))


        writer.write(conf)

    print("mean", np.mean(trip_counts), "std", np.std(trip_counts), "max", np.amax(trip_counts), "min", np.amin(trip_counts))

    writer.close()

    # grad_fn = jax.jacrev(rigid_shape.q_opt, argnums=(0,1))
    # dq_dxa, dq_dxb = grad_fn(coords_a, coords_b, params_a, params_b)
    # print(dq_dxa, dq_dxb)

    # print(po)


# def test_minimize():

#     prefactor = 2.7  # unitless
#     lamb = (4 * np.pi) / (3 * prefactor)  # unitless
#     kappa = np.pi / (np.power(lamb, 2 / 3))  # unitless
#     sigma = 1.6  # angstroms or nm # TODO: check unit?
#     alpha = kappa / (sigma * sigma)
#     # TODO: extract the preceding setup block and share between test_custom_op and test_volume_range?

#     suppl = Chem.SDMolSupplier("tests/data/ligands_40.sdf", removeHs=False)
#     mols = []
#     for mol in suppl:
#         mols.append(mol)

#     ligand_a = mols[0]
#     ligand_b = mols[3]

#     coords_a = get_conf(ligand_a, idx=0)
#     params_a = np.stack([
#         np.zeros(ligand_a.GetNumAtoms()) + alpha,
#         np.zeros(ligand_a.GetNumAtoms()) + prefactor,
#     ], axis=1)

#     coords_b = get_conf(ligand_b, idx=0)
#     params_b = np.stack([
#         np.zeros(ligand_b.GetNumAtoms()) + alpha,
#         np.zeros(ligand_b.GetNumAtoms()) + prefactor,
#     ], axis=1)

#     coords_a = coords_a - np.mean(coords_a, axis=0)
#     coords_b = coords_b - np.mean(coords_b, axis=0)

#     # TEST HESSIAN
#     qi = Rotation.from_euler('z',25,degrees=True).as_quat() # needs to be transposed
#     qi = np.array([qi[3], qi[0], qi[1], qi[2]])
#     # qi = Rotation.random(random_state=2021).as_quat()
#     coords_b = rigid_shape.rotate(coords_b, qi)

#     po = rigid_shape.q_opt(coords_a, coords_b, params_a, params_b)
#     # print(po)

#     # assert 0
#     dg_dq_fn = jax.jacrev(rigid_shape.rotated_normalized_overlap_3)
#     grad = dg_dq_fn(po, coords_a, params_a, coords_b, params_b)
#     print("grad", grad)

#     q_hess_fn = jax.hessian(rigid_shape.rotated_normalized_overlap_3, argnums=(0,))
#     q_mp_fn = jax.jacfwd(jax.grad(rigid_shape.rotated_normalized_overlap_3, argnums=(1,)), argnums=(0,))

#     # positive semi definite matrix!
#     M = q_mp_fn(po, coords_a, params_a, coords_b, params_b)[0][0]
#     H = q_hess_fn(po, coords_a, params_a, coords_b, params_b)[0][0]
#     H_inv = np.linalg.inv(H)
#     print("H:", H)
#     print("H_inv:", H_inv)
#     print("M:", M)
#     print("-M.H_inv", np.transpose(-np.matmul(M, H_inv), axes=(2,0,1)))

#     q_opt_jac_fn = jax.jit(jax.jacrev(rigid_shape.q_opt, argnums=(0,)))
#     q_J = q_opt_jac_fn(coords_a, coords_b, params_a, params_b)

#     print("q_J", q_J[0])

#     # holy shit q_J[0] == -M.H_inv! cuz math (which means we don't have to back prop through derivatives etc.)
#     assert 0


#     # print("dummy angle", np.pi/25)

#     # qi = Rotation.from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).as_quat()


#     # grad_fn = jax.jit(jax.grad(rigid_shape.q_loss, argnums=(0, 1)))
#     loss_fn = jax.jit(rigid_shape.q_loss)


#     c_mol = Chem.CombineMols(ligand_a, ligand_b)
#     writer = Chem.SDWriter('quat.sdf')

#     for angle in np.linspace(0, 180, 100):


#         # qi = Rotation.from_quat([np.cos(np.pi/2), 0, 0, np.sin(np.pi/2)]).as_quat()


#         # q_opt = rigid_shape.q_opt(coords_a, coords_b, params_a, params_b)
#         loss, q_final = loss_fn(coords_a, coords_b, params_a, params_b)
#         print("loss", loss, "angle", angle, "q_final", q_final)


#         # conf = make_conformer(c_mol, coords_a, rigid_shape.rotate(coords_b, q_final))
#         # writer.write(conf)
#         # assert 0


#         # dqopt_dx = grad_fn(coords_a, coords_b, params_a, params_b)

#         # print("force", dqopt_dx)
#     writer.close()

#     assert 0


#     # start = time.time()
#     # dqopt_dx = opt_fn(coords_a, coords_b, params_a, params_b)
#     # print(time.time() - start)

#     # print(q_opt)

#     # a_idxs = get_heavy_atom_idxs(ligand_a)
#     # b_idxs = get_heavy_atom_idxs(ligand_b)

#     a_idxs = np.arange(ligand_a.GetNumAtoms())
#     b_idxs = np.arange(ligand_b.GetNumAtoms())

#     u_fn = functools.partial(
#         rigid_shape.partial_normalized_overlap,
#         x_a=coords_a,
#         params_a=params_a,
#         x_b=coords_b,
#         params_b=params_b,
#         a_idxs=a_idxs,
#         b_idxs=b_idxs
#     )

#     grad_fn = jax.grad(u_fn, argnums=(0,))
#     u_fn = jax.jit(u_fn)
#     grad_fn = jax.jit(grad_fn)

#     lr = 3e-5
#     c_mol = Chem.CombineMols(ligand_a, ligand_b)
#     writer = Chem.SDWriter('quat.sdf')
    
#     # different random rotation
#     # qi = Rotation.random(random_state=2021).as_quat()
#     qi = np.array([1.0, 0.0, 0.0, 0.0])

#     for step in range(150):
#         u = u_fn(qi)
#         print(step, "nrg", u, "Q", qi)
#         du_dq = grad_fn(qi)[0]
#         qi = qi - lr*du_dq
#         qi = qi/np.linalg.norm(qi)
#         # print(qi)
#         conf = make_conformer(c_mol, coords_a, rigid_shape.rotate(coords_b, qi))
#         writer.write(conf)

#     writer.close()
