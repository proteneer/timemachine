from jax.config import config

config.update("jax_enable_x64", True)
import jax

import copy
import functools

from rdkit import Chem
from fe import endpoint_correction
from timemachine.integrator import langevin_coefficients
from timemachine.potentials import bonded
from timemachine import constants
import numpy as np

from pathlib import Path
from ff.handlers.deserialize import deserialize_handlers
from ff import Forcefield
from fe.utils import get_romol_conf


def setup_system():

    root = Path(__file__).parent.parent
    path_to_ligand = str(root.joinpath("tests/data/ligands_40.sdf"))
    path_to_ff = str(root.joinpath("ff/params/smirnoff_1_1_0_ccc.py"))

    with open(path_to_ff) as f:
        ff_handlers = deserialize_handlers(f.read())

    forcefield = Forcefield(ff_handlers)
    suppl = Chem.SDMolSupplier(path_to_ligand, removeHs=False)
    all_mols = [x for x in suppl]
    mol_a = copy.deepcopy(all_mols[1])

    # test identity transformation
    # mol_a, _, _, forcefield = relative.hif2a_ligand_pair
    mol_b = Chem.Mol(mol_a)

    # parameterize with bonds and angles
    bond_params_a, bond_idxs_a = forcefield.hb_handle.parameterize(mol_a)
    angle_params_a, angle_idxs_a = forcefield.ha_handle.parameterize(mol_a)

    bond_params_b, bond_idxs_b = forcefield.hb_handle.parameterize(mol_b)
    angle_params_b, angle_idxs_b = forcefield.ha_handle.parameterize(mol_b)

    box = np.eye(3) * 100.0

    bond_fn = functools.partial(
        bonded.harmonic_bond,
        bond_idxs=np.concatenate([bond_idxs_a, bond_idxs_b + mol_a.GetNumAtoms()]),
        params=np.concatenate([bond_params_a, bond_params_b]),
        box=box,
        lamb=None,
    )

    angle_fn = functools.partial(
        bonded.harmonic_angle,
        angle_idxs=np.concatenate([angle_idxs_a, angle_idxs_b + mol_a.GetNumAtoms()]),
        params=np.concatenate([angle_params_a, angle_params_b]),
        box=box,
        lamb=None,
    )

    core_idxs = []
    core_params = []

    for a in mol_a.GetAtoms():
        if a.IsInRing():
            core_idxs.append((a.GetIdx(), a.GetIdx() + mol_a.GetNumAtoms()))
            # if the force constant is too high, no k/t combination can generate
            # good overlap
            core_params.append((50.0, 0.0))

    core_idxs = np.array(core_idxs, dtype=np.int32)
    core_params = np.array(core_params, dtype=np.float64)

    core_restr = functools.partial(bonded.harmonic_bond, bond_idxs=core_idxs, params=core_params, box=box, lamb=None)

    # left hand state has intractable restraints turned on.
    def u_lhs_fn(x_t):
        return bond_fn(x_t) + angle_fn(x_t) + core_restr(x_t)

    # right hand state is post-processed from independent gas phase simulations
    def u_rhs_fn(x_t):
        return bond_fn(x_t) + angle_fn(x_t)

    return u_lhs_fn, u_rhs_fn, core_idxs, core_params, mol_a, mol_b


# (ytz): do not remove, useful for visualization in pymol
def make_conformer(mol_a, mol_b, conf_c):
    mol_a = Chem.Mol(mol_a)
    mol_b = Chem.Mol(mol_b)

    """Remove all of mol's conformers, make a new mol containing two copies of mol,
    assign positions to each copy using conf_a and conf_b, respectively, assumed in nanometers"""
    assert conf_c.shape[0] == mol_a.GetNumAtoms() + mol_b.GetNumAtoms()
    mol_a.RemoveAllConformers()
    mol_b.RemoveAllConformers()
    mol = Chem.CombineMols(mol_a, mol_b)
    cc = Chem.Conformer(mol.GetNumAtoms())
    conf = np.copy(conf_c)
    conf *= 10  # TODO: label this unit conversion?
    for idx, pos in enumerate(np.asarray(conf)):
        cc.SetAtomPosition(idx, (float(pos[0]), float(pos[1]), float(pos[2])))
    mol.AddConformer(cc)

    return mol


def test_endpoint_correction():
    seed = 2021
    np.random.seed(seed)

    # this PR tests that endpoint correction for two molecules generates a correct, overlapping distribution.
    u_lhs_fn, u_rhs_fn, core_idxs, core_params, mol_a, mol_b = setup_system()

    combined_mass = np.concatenate([[a.GetMass() for a in mol_a.GetAtoms()], [b.GetMass() for b in mol_b.GetAtoms()]])

    combined_conf = np.concatenate([get_romol_conf(mol_a), get_romol_conf(mol_b)])

    dt = 1.5e-3
    friction = 1.0

    temperature = 300.0

    beta = 1 / (constants.BOLTZ * temperature)

    ca, cb, cc = langevin_coefficients(temperature, dt, friction, combined_mass)
    cb = -1 * np.expand_dims(cb, axis=-1)
    cc = np.expand_dims(cc, axis=-1)

    x_t = combined_conf
    v_t = np.zeros_like(x_t)

    n_steps = 50000
    equilibrium_steps = 5000
    sampling_frequency = 100

    # n_steps = 5000
    # equilibrium_steps = 500
    # sampling_frequency = 100

    lhs_xs = []
    lhs_du_dx_fn = jax.jit(jax.grad(u_lhs_fn))
    for step in range(n_steps):
        du_dx = lhs_du_dx_fn(x_t)
        v_t = ca * v_t + cb * du_dx + cc * np.random.normal(size=x_t.shape)
        x_t = x_t + v_t * dt
        if step % sampling_frequency == 0 and step > equilibrium_steps:
            lhs_xs.append(x_t)

    # reset x and v
    x_t = combined_conf
    v_t = np.zeros_like(x_t)

    rhs_xs = []
    rhs_du_dx_fn = jax.jit(jax.grad(u_rhs_fn))
    for step in range(n_steps):
        du_dx = rhs_du_dx_fn(x_t)
        v_t = ca * v_t + cb * du_dx + cc * np.random.normal(size=x_t.shape)
        x_t = x_t + v_t * dt
        if step % sampling_frequency == 0 and step > equilibrium_steps:
            rhs_xs.append(x_t)

    lhs_xs = np.array(lhs_xs)
    rhs_xs = np.array(rhs_xs)

    k_translation = 200.0

    results = []
    for i, k_rotation in enumerate([0.0, 50.0, 1000.0]):

        lhs_du, rhs_du, rotations, translations = endpoint_correction.estimate_delta_us(
            k_translation, k_rotation, core_idxs, core_params, beta, lhs_xs, rhs_xs, seed=seed
        )
        # Verify that output is deterministic, only done for first iteration to reduce time of the test
        if i == 0:
            test_lhs_du, test_rhs_du, test_rotations, test_translations = endpoint_correction.estimate_delta_us(
                k_translation, k_rotation, core_idxs, core_params, beta, lhs_xs, rhs_xs, seed=seed
            )

            np.testing.assert_array_equal(lhs_du, test_lhs_du)
            np.testing.assert_array_equal(rhs_du, test_rhs_du)
            np.testing.assert_array_equal(rotations, test_rotations)
            np.testing.assert_array_equal(translations, test_translations)

        overlap = endpoint_correction.overlap_from_cdf(lhs_du, rhs_du)
        results.append(overlap)

        print("k_rotation", k_rotation, "overlap", overlap)

    assert results[0] < 0.15
    assert results[1] > 0.30
    assert results[2] < 0.25

    # assert overlaps[0] < 0.15
    # assert overlaps[3] > 0.45
    # assert overlaps[-1] < 0.2

    # print(k_rotation, overlap)

    # print("trial", trial, "lhs amin amax", np.amin(lhs_du), np.amax(lhs_du))
    # print("trial", trial, "rhs amin amax", np.amin(rhs_du), np.amax(rhs_du))

    # plt.clf()
    # plt.title("BAR")
    # plt.hist(np.array(lhs_du), alpha=0.5, label='forward', density=True)
    # plt.hist(np.array(rhs_du), alpha=0.5, label='backward', density=True)
    # plt.legend()
    # plt.savefig("overlap_data/trial_"+str(trial)+"_over_lap.png")

    # print("rhs time", time.time()-start)
    # print("avg rmsd", np.mean(rmsds))
    # dG = pymbar.BAR(BETA*lhs_du, -BETA*np.array(rhs_du))[0]/BETA
    # print("trial", trial, "step", step, "dG estimate", dG, "msd", np.mean(rmsds))

    # return
