# Test enhanced sampling protocols

import os
import pickle
from jax.config import config; config.update("jax_enable_x64", True)
import jax

from rdkit import Chem
from rdkit.Chem import AllChem

from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers
from ff.handlers import openmm_deserializer
from md import builders
from md import minimizer
from timemachine.lib import custom_ops
from fe import topology
from fe import free_energy
from fe.utils import get_romol_conf

from timemachine.potentials import bonded, nonbonded
from timemachine.integrator import langevin_coefficients
from timemachine.constants import BOLTZ
from timemachine import lib
from timemachine.potentials import rmsd

from md.barostat.utils import get_group_indices, get_bond_list

import numpy as np
import matplotlib.pyplot as plt

from md import enhanced_sampling

from scipy.special import logsumexp


MOL_SDF = """
  Mrv2115 09292117373D          

 15 16  0  0  0  0            999 V2000
   -1.3280    3.9182   -1.1733 F   0  0  0  0  0  0  0  0  0  0  0  0
    0.4924    2.9890   -0.9348 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.6519    3.7878   -0.9538 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.9215    3.2010   -0.8138 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.0376    1.8091   -0.6533 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.8835    1.0062   -0.6230 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.6026    1.5878   -0.7603 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5399    0.7586   -0.7175 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2257    0.5460    0.5040 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6191    1.4266    2.2631 F   0  0  0  0  0  0  0  0  0  0  0  0
   -2.3596   -0.2866    0.5420 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.8171   -0.9134   -0.6298 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.1427   -0.7068   -1.8452 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.0087    0.1257   -1.8951 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0878    0.3825   -3.7175 F  0  0  0  0  0  0  0  0  0  0  0  0
  2  3  4  0  0  0  0
  3  4  4  0  0  0  0
  4  5  4  0  0  0  0
  5  6  4  0  0  0  0
  6  7  4  0  0  0  0
  2  7  4  0  0  0  0
  7  8  1  0  0  0  0
  8  9  4  0  0  0  0
  9 11  4  0  0  0  0
 11 12  4  0  0  0  0
 12 13  4  0  0  0  0
 13 14  4  0  0  0  0
  8 14  4  0  0  0  0
  9 10  1  0  0  0  0
  1  2  1  0  0  0  0
 14 15  1  0  0  0  0
M  END
$$$$"""

# MOL_SDF = """
#   Mrv2115 10162101243D          

#  21 21  0  0  0  0            999 V2000
#     2.1318    1.8713    2.5342 C   0  0  0  0  0  0  0  0  0  0  0  0
#     1.8519    2.8199    3.5366 C   0  0  0  0  0  0  0  0  0  0  0  0
#     0.7171    3.6436    3.4311 C   0  0  0  0  0  0  0  0  0  0  0  0
#    -0.1356    3.5226    2.3195 C   0  0  0  0  0  0  0  0  0  0  0  0
#     0.1424    2.5747    1.3160 C   0  0  0  0  0  0  0  0  0  0  0  0
#     1.2788    1.7354    1.4122 C   0  0  0  0  0  0  0  0  0  0  0  0
#     1.5879    0.7509    0.3390 C   0  0  2  0  0  0  0  0  0  0  0  0
#     2.4840    1.3772   -0.7628 C   0  0  2  0  0  0  0  0  0  0  0  0
#     2.8248    0.3828   -1.8933 C   0  0  0  0  0  0  0  0  0  0  0  0
#     2.9658    1.2839    2.6243 H   0  0  0  0  0  0  0  0  0  0  0  0
#     2.4765    2.9137    4.3404 H   0  0  0  0  0  0  0  0  0  0  0  0
#     0.5148    4.3338    4.1600 H   0  0  0  0  0  0  0  0  0  0  0  0
#    -0.9559    4.1273    2.2385 H   0  0  0  0  0  0  0  0  0  0  0  0
#    -0.4861    2.5043    0.5104 H   0  0  0  0  0  0  0  0  0  0  0  0
#     0.6602    0.3872   -0.1139 H   0  0  0  0  0  0  0  0  0  0  0  0
#     2.0954   -0.1202    0.7650 H   0  0  0  0  0  0  0  0  0  0  0  0
#     3.4189    1.7271   -0.3152 H   0  0  0  0  0  0  0  0  0  0  0  0
#     1.9719    2.2387   -1.2013 H   0  0  0  0  0  0  0  0  0  0  0  0
#     3.3637   -0.4777   -1.4927 H   0  0  0  0  0  0  0  0  0  0  0  0
#     3.4543    0.8699   -2.6402 H   0  0  0  0  0  0  0  0  0  0  0  0
#     1.9124    0.0354   -2.3815 H   0  0  0  0  0  0  0  0  0  0  0  0
#   1  2  2  0  0  0  0
#   2  3  1  0  0  0  0
#   3  4  2  0  0  0  0
#   4  5  1  0  0  0  0
#   5  6  2  0  0  0  0
#   6  1  1  0  0  0  0
#   6  7  1  0  0  0  0
#   7  8  1  0  0  0  0
#   8  9  1  0  0  0  0
#   1 10  1  0  0  0  0
#   2 11  1  0  0  0  0
#   3 12  1  0  0  0  0
#   4 13  1  0  0  0  0
#   5 14  1  0  0  0  0
#   7 15  1  0  0  0  0
#   7 16  1  0  0  0  0
#   8 17  1  0  0  0  0
#   8 18  1  0  0  0  0
#   9 19  1  0  0  0  0
#   9 20  1  0  0  0  0
#   9 21  1  0  0  0  0
# M  END
# $$$$"""

# MOL_SDF = """
#   Mrv2115 10162101383D          

#  36 36  0  0  0  0            999 V2000
#     2.2010    1.3262    0.0659 C   0  0  0  0  0  0  0  0  0  0  0  0
#     1.5613    2.5749    0.1698 C   0  0  0  0  0  0  0  0  0  0  0  0
#     0.1567    2.6513    0.1914 C   0  0  0  0  0  0  0  0  0  0  0  0
#    -0.6087    1.4737    0.1180 C   0  0  0  0  0  0  0  0  0  0  0  0
#     0.0195    0.2104    0.0204 C   0  0  0  0  0  0  0  0  0  0  0  0
#     1.4322    0.1509   -0.0073 C   0  0  0  0  0  0  0  0  0  0  0  0
#    -0.7912   -1.0285   -0.1039 C   0  0  2  0  0  0  0  0  0  0  0  0
#    -1.0819   -1.3356   -1.5972 C   0  0  2  0  0  0  0  0  0  0  0  0
#    -1.9270   -2.6205   -1.7970 C   0  0  2  0  0  0  0  0  0  0  0  0
#    -2.2161   -2.9249   -3.2924 C   0  0  2  0  0  0  0  0  0  0  0  0
#    -3.0632   -4.2144   -3.4755 C   0  0  1  0  0  0  0  0  0  0  0  0
#    -3.3605   -4.5318   -4.9662 C   0  0  1  0  0  0  0  0  0  0  0  0
#    -4.2071   -5.8201   -5.1564 C   0  0  2  0  0  0  0  0  0  0  0  0
#    -4.4989   -6.1293   -6.6434 C   0  0  0  0  0  0  0  0  0  0  0  0
#     3.2220    1.2712    0.0409 H   0  0  0  0  0  0  0  0  0  0  0  0
#     2.1174    3.4270    0.2319 H   0  0  0  0  0  0  0  0  0  0  0  0
#    -0.3087    3.5597    0.2576 H   0  0  0  0  0  0  0  0  0  0  0  0
#    -1.6288    1.5473    0.1292 H   0  0  0  0  0  0  0  0  0  0  0  0
#     1.9128   -0.7484   -0.0882 H   0  0  0  0  0  0  0  0  0  0  0  0
#    -1.7323   -0.9177    0.4418 H   0  0  0  0  0  0  0  0  0  0  0  0
#    -0.2625   -1.8704    0.3516 H   0  0  0  0  0  0  0  0  0  0  0  0
#    -0.1320   -1.4500   -2.1284 H   0  0  0  0  0  0  0  0  0  0  0  0
#    -1.6162   -0.4880   -2.0373 H   0  0  0  0  0  0  0  0  0  0  0  0
#    -2.8772   -2.5053   -1.2672 H   0  0  0  0  0  0  0  0  0  0  0  0
#    -1.3920   -3.4680   -1.3584 H   0  0  0  0  0  0  0  0  0  0  0  0
#    -1.2669   -3.0403   -3.8249 H   0  0  0  0  0  0  0  0  0  0  0  0
#    -2.7512   -2.0782   -3.7338 H   0  0  0  0  0  0  0  0  0  0  0  0
#    -2.5256   -5.0574   -3.0314 H   0  0  0  0  0  0  0  0  0  0  0  0
#    -4.0099   -4.0953   -2.9403 H   0  0  0  0  0  0  0  0  0  0  0  0
#    -3.8975   -3.6878   -5.4081 H   0  0  0  0  0  0  0  0  0  0  0  0
#    -2.4128   -4.6501   -5.4993 H   0  0  0  0  0  0  0  0  0  0  0  0
#    -5.1607   -5.7112   -4.6315 H   0  0  0  0  0  0  0  0  0  0  0  0
#    -3.6770   -6.6729   -4.7225 H   0  0  0  0  0  0  0  0  0  0  0  0
#    -5.0947   -7.0413   -6.7197 H   0  0  0  0  0  0  0  0  0  0  0  0
#    -5.0571   -5.3094   -7.1022 H   0  0  0  0  0  0  0  0  0  0  0  0
#    -3.5661   -6.2759   -7.1938 H   0  0  0  0  0  0  0  0  0  0  0  0
#   1  2  2  0  0  0  0
#   2  3  1  0  0  0  0
#   3  4  2  0  0  0  0
#   4  5  1  0  0  0  0
#   5  6  2  0  0  0  0
#   6  1  1  0  0  0  0
#   5  7  1  0  0  0  0
#   7  8  1  0  0  0  0
#   8  9  1  0  0  0  0
#   9 10  1  0  0  0  0
#  10 11  1  0  0  0  0
#  11 12  1  0  0  0  0
#  12 13  1  0  0  0  0
#  13 14  1  0  0  0  0
#   1 15  1  0  0  0  0
#   2 16  1  0  0  0  0
#   3 17  1  0  0  0  0
#   4 18  1  0  0  0  0
#   6 19  1  0  0  0  0
#   7 20  1  0  0  0  0
#   7 21  1  0  0  0  0
#   8 22  1  0  0  0  0
#   8 23  1  0  0  0  0
#   9 24  1  0  0  0  0
#   9 25  1  0  0  0  0
#  10 26  1  0  0  0  0
#  10 27  1  0  0  0  0
#  11 28  1  0  0  0  0
#  11 29  1  0  0  0  0
#  12 30  1  0  0  0  0
#  12 31  1  0  0  0  0
#  13 32  1  0  0  0  0
#  13 33  1  0  0  0  0
#  14 34  1  0  0  0  0
#  14 35  1  0  0  0  0
#  14 36  1  0  0  0  0
# M  END
# $$$$"""

# Why doesn't benzene work? probably bad-bond lengths->instant blowup
# MOL_SDF = """
#   Mrv2115 10152122132D          

#  12 12  0  0  0  0            999 V2000
#    -0.3669    0.6543    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
#    -1.1224    0.6543    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
#    -1.5002    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
#    -1.1224   -0.6543    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
#    -0.3669   -0.6543    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
#     0.0109    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
#    -0.0166    1.2610    0.0000 F   0  0  0  0  0  0  0  0  0  0  0  0
#     0.7114    0.0000    0.0000 F   0  0  0  0  0  0  0  0  0  0  0  0
#    -0.0166   -1.2610    0.0000 F   0  0  0  0  0  0  0  0  0  0  0  0
#    -1.4727   -1.2610    0.0000 F   0  0  0  0  0  0  0  0  0  0  0  0
#    -2.2007    0.0000    0.0000 F   0  0  0  0  0  0  0  0  0  0  0  0
#    -1.4727    1.2610    0.0000 F   0  0  0  0  0  0  0  0  0  0  0  0
#   1  2  2  0  0  0  0
#   2  3  1  0  0  0  0
#   3  4  2  0  0  0  0
#   4  5  1  0  0  0  0
#   5  6  2  0  0  0  0
#   6  1  1  0  0  0  0
#   1  7  1  0  0  0  0
#   6  8  1  0  0  0  0
#   5  9  1  0  0  0  0
#   4 10  1  0  0  0  0
#   3 11  1  0  0  0  0
#   2 12  1  0  0  0  0
# M  END
# $$$$"""

# (ytz): do not remove, useful for visualization in pymol
def make_conformer(mol, conf):
    mol = Chem.Mol(mol)
    mol.RemoveAllConformers()
    cc = Chem.Conformer(mol.GetNumAtoms())
    conf = conf*10
    for idx, pos in enumerate(np.asarray(conf)):
        cc.SetAtomPosition(idx, (float(pos[0]), float(pos[1]), float(pos[2])))
    mol.AddConformer(cc)
    return mol

def align_sample(x_gas, x_solvent):
    """
    Return a rigidly transformed x_gas that is maximally aligned to x_solvent.
    """
    num_atoms = len(x_gas)

    xa = x_solvent[-num_atoms:]
    xb = x_gas

    assert xa.shape == xb.shape

    xb_new = rmsd.align_x2_unto_x1(xa, xb)
    return xb_new


def generate_gas_phase_samples(
    mol,
    ff,
    temperature,
    U_target,
    steps_per_batch=250,
    num_batches=10000):
    """
    Generate a set of gas-phase samples by running steps_per_batch * num_batches steps

    Parameters
    ----------
    mol: Chem.Mol
    
    ff: forcefield

    temperature: float

    U_target: fn
        Potential energy function we wish to re-weight into

    Returns
    -------
    3-tuple
        Return counts, samples, energies
    """
    masses = np.array([a.GetMass() for a in mol.GetAtoms()])
    x0 = get_romol_conf(mol)

    kT = temperature*BOLTZ
    masses = np.array([a.GetMass() for a in mol.GetAtoms()])
    num_workers = jax.device_count()

    state = enhanced_sampling.EnhancedState(mol, ff)

    xs_easy = enhanced_sampling.generate_samples(
        masses,
        x0,
        state.U_easy,
        temperature,
        steps_per_batch,
        num_batches,
        num_workers
    )

    writer = Chem.SDWriter("results.sdf")
    num_atoms = mol.GetNumAtoms()
    torsions = []

    # discard first few batches for burn-in and reshape into a single flat array
    xs_easy = xs_easy[:, 1000:, :, :]

    batch_U_easy_fn = jax.pmap(jax.vmap(state.U_easy))

    batch_U_target_fn = jax.pmap(jax.vmap(U_target))

    Us_target = batch_U_target_fn(xs_easy)
    Us_easy = batch_U_easy_fn(xs_easy)

    log_numerator = -Us_target.reshape(-1)/kT
    log_denominator = -Us_easy.reshape(-1)/kT

    log_weights = log_numerator - log_denominator
    weights = np.exp(log_weights - logsumexp(log_weights))

    # sample from weights
    sample_size = len(weights)
    idxs = np.random.choice(np.arange(len(weights)), size=sample_size, p=weights)

    unique_target_kv = {}
    for i in idxs:
        if i not in unique_target_kv:
            unique_target_kv[i] = 0
        unique_target_kv[i] += 1

    # keys() and values() will always return in the same order in python3
    unique_target_idxs = np.array(list(unique_target_kv.keys()))
    unique_target_counts = np.array(list(unique_target_kv.values()))

    Us_target_unique = Us_target.reshape(-1)[unique_target_idxs]
    xs_target_unique = xs_easy.reshape(-1, num_atoms, 3)[unique_target_idxs]

    return unique_target_counts, xs_target_unique, Us_target_unique


def generate_solvent_phase_samples(mol, ff, temperature):

    x0 = get_romol_conf(mol)

    masses = np.array([a.GetMass() for a in mol.GetAtoms()])
    num_workers = jax.device_count()
    state = enhanced_sampling.EnhancedState(mol, ff)
    water_system, water_coords, water_box, water_topology = builders.build_water_system(3.0)
    num_water_atoms = len(water_coords)
    afe = free_energy.AbsoluteFreeEnergy(mol, ff, decharge=False)
    ff_params = ff.get_ordered_params()
    ubps, params, masses, coords = afe.prepare_host_edge(ff_params, water_system, water_coords)

    dt = 1.5e-3
    friction = 1.0
    intg = lib.LangevinIntegrator(temperature, dt, friction, masses, 2021)
    
    pressure = 1.0
    interval = 5
    bond_list = get_bond_list(ubps[0])
    group_idxs = get_group_indices(bond_list)

    barostat = lib.MonteCarloBarostat(
        len(masses),
        pressure,
        temperature,
        group_idxs,
        interval,
        2022
    )

    box = water_box
    host_coords = coords[:num_water_atoms]
    new_host_coords = minimizer.minimize_host_4d([mol], water_system, host_coords, ff, water_box)
    coords[:num_water_atoms] = new_host_coords

    bps = []
    for p, bp in zip(params, ubps):
        bps.append(bp.bind(p))

    all_impls = [bp.bound_impl(np.float32) for bp in bps]
    intg_impl = intg.impl()
    barostat_impl = barostat.impl(all_impls)

    intg_equil = lib.LangevinIntegrator(temperature, 1e-4, friction, masses, 2021)
    intg_equil_impl = intg_equil.impl()

    equil_ctxt = custom_ops.Context(
        coords,
        np.zeros_like(coords),
        box,
        intg_equil_impl,
        all_impls,
        None
    )

    lamb = 0.0
    equil_schedule = np.ones(5000)*lamb
    equil_ctxt.multiple_steps(equil_schedule)

    x0 = equil_ctxt.get_x_t()
    v0 = np.zeros_like(x0)

    ctxt = custom_ops.Context(
        x0,
        v0,
        box,
        intg_impl,
        all_impls,
        barostat_impl
    )

    lamb = 0.0
    num_steps = 800000
    # num_steps = 100
    # num_steps = 20000
    lambda_windows = np.array([0.0])
    u_interval = 1000
    x_interval = 1000

    full_us, xs, boxes = ctxt.multiple_steps_U(
        lamb,
        num_steps,
        lambda_windows,
        u_interval,
        x_interval
    )

    return xs[50:], boxes[50:], full_us[50:], bps[-1], params[-1]

def test_condensed_phase():
    
    # generate gas-phase xs, with LJ terms only turned on
    # generate condensed-phase xs, batch RMSD align and re-weight

    #              xx x x <-- torsion indices
    #          01 23456 7 8 9
    mol = Chem.MolFromMolBlock(MOL_SDF, removeHs=False)
    # torsion_idxs = np.array([5,6,7,8])

    masses = np.array([a.GetMass() for a in mol.GetAtoms()])
    num_ligand_atoms = len(masses)

    ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_sc.py').read())
    ff = Forcefield(ff_handlers)

    cache_path = "cache.pkl"

    temperature = 300.0

    state = enhanced_sampling.EnhancedState(mol, ff)

    U_target = state.U_full

    if not os.path.exists(cache_path):
        print("Generating cache")

        # generate samples in some target state
        target_counts, xs_target_unique, Us_target_unique = generate_gas_phase_samples(mol, ff, temperature, U_target)
        xs_solvent, boxes_solvent, Us_full, nb_bp, nb_params = generate_solvent_phase_samples(mol, ff, temperature)
        with open(cache_path, "wb") as fh:
            pickle.dump((target_counts, xs_target_unique, Us_target_unique, xs_solvent, boxes_solvent, Us_full, nb_bp, nb_params), fh)

    with open(cache_path, "rb") as fh:
        target_counts, xs_target_unique, Us_target_unique, xs_solvent, boxes_solvent, Us_full, nb_bp, nb_params = pickle.load(fh)

    # xs_easy = xs_easy.reshape(-1, num_ligand_atoms, 3)

    log_numerator = []
    log_denominator = []

    params_i = nb_params[-num_ligand_atoms:] # ligand params
    params_j = nb_params[:-num_ligand_atoms] # water params

    state = enhanced_sampling.EnhancedState(mol, ff)

    beta = 2.0

    print("params_i", params_i)
    print("params_j", params_j)

    def before_U_k(x_solvent, box_solvent):
        x_water = x_solvent[:-num_ligand_atoms] # water coords
        x_original = x_solvent[-num_ligand_atoms:] # ligand coords
        U_k = nonbonded.nonbonded_off_diagonal(x_original, x_water, box_solvent, params_i, params_j)
        return U_k
    
    @jax.jit
    def U_a(x):
        return state.U_full(x)

    # we know what the energy is actually, no need to compute it
    @jax.jit
    def U_b(x):
        return U_target(x)
        # state.U_decharged(x)

    batch_before_U_k_fn = jax.jit(jax.vmap(before_U_k))
    batch_U_a_fn = jax.jit(jax.vmap(U_a))
    batch_U_b_fn = jax.jit(jax.vmap(U_b))
    before_U_k_nrgs = batch_before_U_k_fn(xs_solvent, boxes_solvent)

    def after_U_k(x_gas, x_solvent, box_solvent):
        x_gas_aligned = align_sample(x_gas, x_solvent) # align gas phase conformer
        x_water = x_solvent[:-num_ligand_atoms] # water coords
        U_k = nonbonded.nonbonded_off_diagonal(x_gas_aligned, x_water, box_solvent, params_i, params_j)
        return U_k
    
    batch_after_U_k_fn = jax.jit(jax.vmap(after_U_k, in_axes=(0, None, None)))

    kT = temperature*BOLTZ

    print(xs_target_unique.shape, "unique samples out of", np.sum(target_counts), "total samples")

    all_delta_us_unique = []

    writer = Chem.SDWriter("test2.sdf")

    for idx, (x_solvent, box_solvent) in enumerate(zip(xs_solvent, boxes_solvent)):

        # writer.write(make_conformer(mol, x_solvent[-num_ligand_atoms:]))
        # for x_g in xs_target_unique[:100]:
        #     x_gas_aligned = align_sample(x_g, x_solvent) # align gas phase conformer
        #     new_mol = make_conformer(mol, np.asarray(x_gas_aligned))
        #     print("rmsd after alignment", np.linalg.norm(x_gas_aligned - x_solvent[-num_ligand_atoms:]))
        #     writer.write(new_mol)

        # writer.flush()
        # writer.close()
        # assert 0

        before_U_k_sample = before_U_k_nrgs[idx]
        # before_U_a_sample = U_a(x_solvent[-num_ligand_atoms:])
        # before_U_b_samples = batch_U_b_fn(xs_target_unique)
        # before_U_samples = before_U_k_sample + before_U_a_sample + before_U_b_samples

        after_U_k_samples = batch_after_U_k_fn(xs_target_unique, x_solvent, box_solvent)
        # after_U_a_samples = batch_U_a_fn(xs_target_unique)
        # after_U_b_sample = U_b(x_solvent[-num_ligand_atoms:])
        # after_U_samples = after_U_k_samples + after_U_a_samples + after_U_b_sample

        # print("after", after_U_samples)

        # print("after-before on U_ks only", after_U_k_samples - before_U_k_sample)
        # assert 0
        # print("after-before on U_a and U_b only", before_U_a_sample + before_U_b_samples - (after_U_a_samples + after_U_b_sample))

        # delta_us = (after_U_samples - before_U_samples)/kT
        delta_us = (after_U_k_samples - before_U_k_sample)/kT

        all_delta_us_unique.append(delta_us)
        bins = 100
        # print("delta_us", "max", np.amax(delta_us), "min", np.amin(delta_us))

        # for this to be accurate, we need to do weighted sum!
        # print("after < before count", np.sum(delta_us < 0), "before > after count", np.sum(delta_us > 0))
        weights = np.exp(-delta_us) # un normalized
        print("iteration", idx, "before_U_k_sample", before_U_k_sample, "unique_weights", "max", np.amax(weights), "min", np.amin(weights))
        # plt.hist(delta_us, bins, density=True, alpha=0.2, label="frame_"+str(idx), weights=target_counts)
        plt.hist(np.exp(-delta_us), bins, density=True, alpha=0.2, label="frame_"+str(idx), weights=target_counts)
        # plt.show()
        # assert 0

    # given K streams and N samples per stream with different counts
    # compute 1/(total_counts_per_stream*number_of_streams) * sum_s^K sum_i^N c_i exp(-delta_u_i)
    target_counts = np.array(target_counts)
    prefactor = np.sum(target_counts)*len(all_delta_us_unique)
    all_delta_us_unique = np.array(all_delta_us_unique)
    ratio = np.mean(np.average(np.exp(-all_delta_us_unique), axis=1, weights=target_counts))

    print("ratio of Z_e/Z_f, computed by np.mean(np.exp(-delta_us)):", ratio)

    plt.xlabel("exp(-delta_u)")
    plt.ylabel("density")

    plt.show()


# def test_gas_phase():
#     """
#     This test attempts re-weighting in the gas-phase, where given a proposal
#     distribution 
#     """
#     return

#     #              xx x x <-- torsion indices
#     #          01 23456 7 8 9
#     mol = Chem.MolFromMolBlock(MOL_SDF, removeHs=False)
#     torsion_idxs = np.array([5,6,7,8])

#     # this is broken
#     #      0  12 3 4 5  6  7 8 | torsions are 2,3,6,7
#     # smi = "C1=CC=C(C=C1)C(=O)O"
#     # mol = Chem.AddHs(Chem.MolFromSmiles(smi))
#     # AllChem.EmbedMolecule(mol)
#     # torsion_idxs = np.array([2,3,6,7])

#     masses = np.array([a.GetMass() for a in mol.GetAtoms()])

#     ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_sc.py').read())
#     ff = Forcefield(ff_handlers)
#     x0 = get_romol_conf(mol)

#     steps_per_batch = 100
#     num_batches = 20000

#     temperature = 300
#     kT = temperature*BOLTZ
#     masses = np.array([a.GetMass() for a in mol.GetAtoms()])
#     num_workers = jax.device_count()

#     state = enhanced_sampling.EnhancedState(mol, ff)

#     xs_easy = enhanced_sampling.generate_samples(
#         masses,
#         x0,
#         state.U_easy,
#         temperature,
#         steps_per_batch,
#         num_batches,
#         num_workers
#     )

#     writer = Chem.SDWriter("results.sdf")
#     num_atoms = mol.GetNumAtoms()
#     torsions = []

#     # discard first few batches for burn-in and reshape into a single flat array
#     xs_easy = xs_easy[:, 1000:, :, :]

#     @jax.jit
#     def get_torsion(x_t):
#         cijkl = x_t[torsion_idxs]
#         return bonded.signed_torsion_angle(*cijkl)

#     batch_torsion_fn = jax.pmap(jax.vmap(get_torsion))
#     batch_U_easy_fn = jax.pmap(jax.vmap(state.U_easy))
#     batch_U_decharged_fn = jax.pmap(jax.vmap(state.U_decharged))
#     batch_U_charged_fn = jax.pmap(jax.vmap(state.U_full))

#     kT = BOLTZ*temperature

#     torsions_easy = batch_torsion_fn(xs_easy).reshape(-1)
#     log_numerator = -batch_U_decharged_fn(xs_easy).reshape(-1)/kT
#     log_denominator = -batch_U_easy_fn(xs_easy).reshape(-1)/kT

#     log_weights = log_numerator - log_denominator
#     weights = np.exp(log_weights - logsumexp(log_weights))

#     # sample from weights
#     sample_size = len(weights)*10
#     idxs = np.random.choice(np.arange(len(weights)), size=sample_size, p=weights)
#     unique_samples = len(set(idxs.tolist()))
#     print("unique samples", unique_samples, "ratio", unique_samples/sample_size)

#     torsions_reweight = torsions_easy[idxs]
#     bins = np.linspace(250, 400, 50) # binned into 5kJ/mol chunks

#     plt.xlabel("energy (kJ/mol)")
#     plt.hist(Us_reweight, density=True, bins=bins, alpha=0.5, label="p_decharged (rw)")
#     plt.hist(Us_decharged, density=True, bins=bins, alpha=0.5, label="p_decharged (md)")
#     plt.legend()
#     plt.savefig("rw_energy_distribution.png")
#     plt.clf()

#     Us_reweight = batch_U_decharged_fn(xs_decharged)

#     torsions_decharged = batch_torsion_fn(xs_decharged).reshape(-1)
#     torsions_reweight_lhs = torsions_reweight[np.nonzero(torsions_reweight < 0)]

#     plt.xlabel("torsion_angle")
#     plt.hist(torsions_easy, density=True, bins=50, label='p_easy', alpha=0.5)
#     plt.hist(torsions_reweight, density=True, bins=50, label='p_decharged (rw)', alpha=0.5)
#     plt.hist(torsions_reweight_lhs, density=True, bins=25, label='p_decharged (rw, lhs only)', alpha=0.5)
#     plt.hist(torsions_decharged, density=True, bins=25, label='p_decharged (md)', alpha=0.5)
#     plt.legend()
#     plt.savefig("rw_torsion_distribution.png")

#     # verify that the histogram of torsions_reweight is
#     # 1) symmetric about theta = 0
#     # 2) agrees with that of a fresh simulation using U_decharged

#     torsions_reweight_lhs, edges = np.histogram(torsions_reweight, bins=50, range=(-np.pi, 0), density=True)
#     torsions_reweight_rhs, edges = np.histogram(torsions_reweight, bins=50, range=( 0, np.pi), density=True)

#     # test symmetry about theta=0
#     assert np.mean((torsions_reweight_lhs - torsions_reweight_rhs[::-1])**2) < 1e-2

#     torsions_decharged_lhs, edges = np.histogram(torsions_decharged, bins=50, range=(-np.pi, 0), density=True)

#     # test against directly simulated results using U_decharged
#     assert np.mean((torsions_reweight_lhs - torsions_decharged_lhs[::-1])**2) < 1e-2



if __name__ == "__main__":
    # test_gas_phase()
    test_condensed_phase()