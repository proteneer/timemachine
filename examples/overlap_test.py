
import os

from jax.config import config; config.update("jax_enable_x64", True)
from rdkit import Chem
from rdkit.Chem import AllChem

import multiprocessing


import jax
import jax.numpy.linalg as linalg
import jax.numpy as np
import functools
import jax.numpy as np
import numpy as onp

from scipy.stats import special_ortho_group
from ff.handlers.deserialize import deserialize_handlers

from rdkit import Chem
from rdkit.Chem import AllChem

from ff import handlers
from timemachine.potentials import bonded, shape
from timemachine.integrator import langevin_coefficients


def inertia_tensor(conf, masses):
    com = np.average(conf, axis=0, weights=masses)
    conf = conf - com
    conf_T = conf.transpose()
    
    xs = conf[:, 0]
    ys = conf[:, 1]
    zs = conf[:, 2]
    xx = np.average(ys*ys + zs*zs, weights=masses)
    yy = np.average(xs*xs + zs*zs, weights=masses)
    zz = np.average(xs*xs + ys*ys, weights=masses)
    xy = np.average(-xs*ys, weights=masses)
    xz = np.average(-xs*zs, weights=masses)
    yz = np.average(-ys*zs, weights=masses)
    tensor = np.array([
        [xx, xy, xz],
        [xy, yy, yz],
        [xz, yz, zz]
    ])

    return com, tensor


def pmi_restraints(conf, params, box, lamb, a_idxs, b_idxs, masses, angle_force, com_force):

    a_com, a_tensor = inertia_tensor(conf[a_idxs], masses[a_idxs])
    b_com, b_tensor = inertia_tensor(conf[b_idxs], masses[b_idxs])

    # don't want to deal with backprop of eigenvalues if we don't have to
    a_eval, a_evec = np.linalg.eigh(a_tensor) # already sorted
    b_eval, b_evec = np.linalg.eigh(b_tensor) # already sorted

    loss = []
    for d in range(3):
        x = a_evec[d]
        y = b_evec[d]

        # arccos is always defined between 1 and -1
        # returns a value between 0 and pi
        # a_pos = np.arccos(np.sum(x*y)/np.linalg.norm(x)*np.linalg.norm(y))
        # a_neg = np.arccos(np.sum(-x*y)/np.linalg.norm(-x)*np.linalg.norm(y))
        a_pos = np.arccos(np.sum(x*y)) # norm is always 1
        a_neg = np.arccos(np.sum(-x*y)) # norm is always 1
        a = np.amin([a_pos, a_neg])
        loss.append(angle_force*a*a)

        # want to greatest force at theta=90 (or dot product of 0)

        # a_pos = np.arccos(np.sum(x*y)/np.linalg.norm(x)*np.linalg.norm(y))
        # loss.append(angle_force*a_pos*a_pos)

        # a = np.sum(x*y)/np.linalg.norm(x)*np.linalg.norm(y)
        # loss = np.sqrt(1-a*a)*100

    loss = np.array(loss)

    return np.sum(loss) + np.linalg.norm(b_com - a_com)*com_force

def recenter(conf):
    return conf - np.mean(conf, axis=0)

def get_conf(romol, idx):
    conformer = romol.GetConformer(idx)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    guest_conf /= 10
    return recenter(guest_conf)

def make_conformer(mol, conf_a, conf_b):
    mol.RemoveAllConformers()
    mol = Chem.CombineMols(mol, mol)
    cc = Chem.Conformer(mol.GetNumAtoms())
    conf = np.concatenate([conf_a, conf_b])
    conf *= 10
    for idx, pos in enumerate(onp.asarray(conf)):
        cc.SetAtomPosition(idx, (float(pos[0]), float(pos[1]), float(pos[2])))
    mol.AddConformer(cc)

    return mol


def get_heavy_atom_idxs(mol):

    idxs = []
    for a_idx, a in enumerate(mol.GetAtoms()):
        if a.GetAtomicNum() > 1:
            idxs.append(a_idx)
    return np.array(idxs, dtype=np.int32)


def convergence(args):
    epoch, lamb = args

    suppl = Chem.SDMolSupplier("tests/data/ligands_40.sdf", removeHs=False)

    ligands = []
    for mol in suppl:
        ligands.append(mol)

    ligand_a = ligands[0]
    ligand_b = ligands[1]

    # print(ligand_a.GetNumAtoms())
    # print(ligand_b.GetNumAtoms())

    # ligand_a = Chem.AddHs(Chem.MolFromSmiles("CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C"))
    # ligand_b = Chem.AddHs(Chem.MolFromSmiles("CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C"))
    # ligand_a = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1CC"))
    # ligand_b = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1CC"))
    # AllChem.EmbedMolecule(ligand_a, randomSeed=2020)
    # AllChem.EmbedMolecule(ligand_b, randomSeed=2020)

    coords_a = get_conf(ligand_a, idx=0)
    coords_b = get_conf(ligand_b, idx=0)
    # coords_b = np.matmul(coords_b, special_ortho_group.rvs(3))

    coords_a = recenter(coords_a)
    coords_b = recenter(coords_b)

    coords = np.concatenate([coords_a, coords_b])

    # a_idxs = np.arange(0, ligand_a.GetNumAtoms()) # change to heavy atoms
    # b_idxs = np.arange(0, ligand_b.GetNumAtoms()) # change to heavy atoms
    a_idxs = get_heavy_atom_idxs(ligand_a)
    b_idxs = get_heavy_atom_idxs(ligand_b)
    # print(a_idxs)
    # print(b_idxs)
    b_idxs += ligand_a.GetNumAtoms()


    nrg_fns = []

    forcefield = 'ff/params/smirnoff_1_1_0_ccc.py'
    ff_raw = open(forcefield, "r").read()
    ff_handlers = deserialize_handlers(ff_raw)

    combined_mol = Chem.CombineMols(ligand_a, ligand_b)

    for handler in ff_handlers:
        if isinstance(handler, handlers.HarmonicBondHandler):
            bond_idxs, (bond_params, _) = handler.parameterize(combined_mol)
            nrg_fns.append(
                functools.partial(bonded.harmonic_bond,
                    params=bond_params,
                    box=None,
                    # lamb=None,
                    bond_idxs=bond_idxs
                )
            )
        elif isinstance(handler, handlers.HarmonicAngleHandler):
            angle_idxs, (angle_params, _) = handler.parameterize(combined_mol)
            nrg_fns.append(
                functools.partial(bonded.harmonic_angle,
                    params=angle_params,
                    box=None,
                    # lamb=None,
                    angle_idxs=angle_idxs
                )
            )
        # elif isinstance(handler, handlers.ImproperTorsionHandler):
        #     torsion_idxs, (torsion_params, _) = handler.parameterize(combined_mol)
        #     print(torsion_idxs)
        #     assert 0
        #     nrg_fns.append(
        #         functools.partial(bonded.periodic_torsion,
        #             params=torsion_params,
        #             box=None,
        #             lamb=None,
        #             torsion_idxs=torsion_idxs
        #         )
        #     )
        # elif isinstance(handler, handlers.ProperTorsionHandler):
        #     torsion_idxs, (torsion_params, _) = handler.parameterize(combined_mol)
        #     # print(torsion_idxs)
        #     nrg_fns.append(
        #         functools.partial(bonded.periodic_torsion,
        #             params=torsion_params,
        #             box=None,
        #             lamb=None,
        #             torsion_idxs=torsion_idxs
        #         )
        #     )

    masses_a = onp.array([a.GetMass() for a in ligand_a.GetAtoms()])
    masses_b = onp.array([a.GetMass() for a in ligand_b.GetAtoms()])

    # super_masses = np.ones_like(np.concatenate([masses_a, masses_b]))
    # super_masses = np.concatenate([masses, masses])
    combined_masses = np.concatenate([masses_a, masses_b])

    com_restraint_fn = functools.partial(bonded.centroid_restraint,
        params=None,
        box=None,
        lamb=None,
        # masses=combined_masses, # try making this ones-like
        masses=np.ones_like(combined_masses),
        group_a_idxs=a_idxs,
        group_b_idxs=b_idxs,
        kb=50.0,
        b0=0.0)


    prefactor = 2.7 # unitless
    shape_lamb = (4*np.pi)/(3*prefactor) # unitless
    kappa = np.pi/(np.power(shape_lamb, 2/3)) # unitless
    sigma = 0.15
    alpha = kappa/(sigma*sigma)

    alphas = np.zeros(combined_mol.GetNumAtoms())+alpha
    weights = np.zeros(combined_mol.GetNumAtoms())+prefactor

    shape_restraint_fn = functools.partial(shape.harmonic_overlap,
        box=None,
        lamb=None,
        params=None,
        a_idxs=a_idxs,
        b_idxs=b_idxs,
        alphas=alphas,
        weights=weights,
        k=200.0
    )

    # pmi_restraint_fn = functools.partial(pmi_restraints,
    #     params=None,
    #     box=None,
    #     lamb=None,
    #     a_idxs=a_idxs,
    #     b_idxs=b_idxs,
    #     masses=combined_masses,
    #     angle_force=50,
    #     com_force=50
    # )

    def restraint_fn(conf, lamb):
        return (1-lamb)*com_restraint_fn(conf) + lamb*shape_restraint_fn(conf)

    nrg_fns.append(restraint_fn)

    def nrg_fn(conf, lamb):
        s = []
        for u in nrg_fns:
            s.append(u(conf, lamb=lamb))
        return np.sum(s)


    grad_fn = jax.grad(nrg_fn, argnums=(0,1))
    grad_fn = jax.jit(grad_fn)

    du_dx_fn = jax.grad(nrg_fn, argnums=(0))
    du_dx_fn = jax.jit(du_dx_fn)

    x_t = coords
    v_t = np.zeros_like(x_t)

    w = Chem.SDWriter('frames_heavy_'+str(epoch)+'.sdf')

    dt = 1.5e-3
    # print(combined_masses.shape)
    ca, cb, cc = langevin_coefficients(300.0, dt, 1.0, combined_masses)
    cb = -1*onp.expand_dims(cb, axis=-1)
    cc = onp.expand_dims(cc, axis=-1)

    du_dls = []

    for step in range(100000):

        # if step % 200 == 0:
        #     u = nrg_fn(x_t, lamb)
        #     print("step", step, "nrg", onp.asarray(u), "avg_du_dl",  onp.mean(du_dls), "overlap", overlap_fn(x_t))
        #     mol = make_conformer(combined_mol, x_t[:ligand_a.GetNumAtoms()], x_t[ligand_a.GetNumAtoms():])
        #     w.write(mol)
        #     w.flush()

        if step % 5 == 0:
            du_dx, du_dl = grad_fn(x_t, lamb)
            du_dls.append(du_dl)
        else:
            du_dx = du_dx_fn(x_t, lamb)[0]


        v_t = ca*v_t + cb*du_dx + cc*onp.random.normal(size=x_t.shape)
        x_t = x_t + v_t*dt

    return np.mean(onp.mean(du_dls))


if __name__ == "__main__":

    pool = multiprocessing.Pool() # defaults to # of cpus

    # lambda_schedule = np.linspace(0, 1.0, os.cpu_count())
    lambda_schedule = np.linspace(0, 1.0, 24)
    # lambda_schedule = np.linspace(0.2, 0.6, 24)

    print("cpu count:", os.cpu_count())

    for epoch in range(100):
        args = []
        for lamb in lambda_schedule:
            args.append((epoch, lamb))
        avg_du_dls = pool.map(convergence, args)
        avg_du_dls = np.asarray(avg_du_dls)

        for lamb, ddl in zip(lambda_schedule, avg_du_dls):
            print("lambda", lamb, "du_dl",  ddl)
        print(epoch, "epoch", "deltaG", onp.trapz(avg_du_dls,lambda_schedule))
