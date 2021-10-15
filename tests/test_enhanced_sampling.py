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

MOL_SDF = """
  Mrv2115 10142118323D          

 22 23  0  0  0  0            999 V2000
   -2.8467    0.6329   -1.0843 C   0  0  0  0  0  0  0  0  0  0  0  0
   -4.2556    0.6150   -1.0620 C   0  0  0  0  0  0  0  0  0  0  0  0
   -4.9547    1.2088    0.0015 C   0  0  0  0  0  0  0  0  0  0  0  0
   -4.2465    1.8151    1.0520 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.8376    1.8123    1.0521 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.1166    1.2101   -0.0122 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.4313    2.4113   -0.0047 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0237    2.4058   -0.0103 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6937    1.1848   -0.0032 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0356   -0.0342    0.0210 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.4436   -0.0177    0.0044 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.1422    1.2000   -0.0035 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.2441    0.1129   -2.1231 F   0  0  0  0  0  0  0  0  0  0  0  0
   -2.2254    2.3702    2.0653 F   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7806   -1.7096    0.1159 Br  0  0  0  0  0  0  0  0  0  0  0  0
   -4.7793    0.1727   -1.8203 H   0  0  0  0  0  0  0  0  0  0  0  0
   -5.9778    1.2005    0.0133 H   0  0  0  0  0  0  0  0  0  0  0  0
   -4.7600    2.2532    1.8201 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.9367    3.2995   -0.0018 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.4744    3.3023   -0.0223 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.9674   -0.8931    0.0068 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.1650    1.2053   -0.0045 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0  0  0  0
  2  3  1  0  0  0  0
  3  4  2  0  0  0  0
  4  5  1  0  0  0  0
  5  6  2  0  0  0  0
  6  1  1  0  0  0  0
  7  8  2  0  0  0  0
  8  9  1  0  0  0  0
  9 10  2  0  0  0  0
 10 11  1  0  0  0  0
 11 12  2  0  0  0  0
 12  7  1  0  0  0  0
  6  9  1  0  0  0  0
  1 13  1  0  0  0  0
  5 14  1  0  0  0  0
 10 15  1  0  0  0  0
  2 16  1  0  0  0  0
  3 17  1  0  0  0  0
  4 18  1  0  0  0  0
  7 19  1  0  0  0  0
  8 20  1  0  0  0  0
 11 21  1  0  0  0  0
 12 22  1  0  0  0  0
M  END
$$$$"""

# (ytz): do not remove, useful for visualization in pymol
def make_conformer(mol, conf):
    mol = Chem.Mol(mol)
    mol.RemoveAllConformers()
    cc = Chem.Conformer(mol.GetNumAtoms())
    conf = np.copy(conf)
    conf *= 10  # convert from nm to A
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

    xb_new = rmsd.align_x2_unto_x1(xb, xa)
    return xb_new

    # for each xs_solvent
    # draw 100 random samples from xs_gas and reweight

def generate_gas_phase_samples(mol, ff):

    masses = np.array([a.GetMass() for a in mol.GetAtoms()])
    x0 = get_romol_conf(mol)

    steps_per_batch = 100
    # num_batches = 20000
    num_batches = 2000

    temperature = 300
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
    batch_U_decharged_fn = jax.pmap(jax.vmap(state.U_decharged))

    Us_decharged = batch_U_decharged_fn(xs_easy)
    Us_easy = batch_U_easy_fn(xs_easy)

    log_numerator = -Us_decharged.reshape(-1)/kT
    log_denominator = -Us_easy.reshape(-1)/kT

    log_weights = log_numerator - log_denominator
    weights = np.exp(log_weights - logsumexp(log_weights))

    # sample from weights
    sample_size = len(weights)
    idxs = np.random.choice(np.arange(len(weights)), size=sample_size, p=weights)
    unique_samples = len(set(idxs.tolist()))

    Us_decharged = Us_decharged.reshape(-1)[idxs]
    xs_decharged = xs_easy.reshape(-1, num_atoms, 3)[idxs]

    assert len(Us_decharged) == len(xs_decharged)

    # return xs_decharged, Us_decharged
    return xs_easy, Us_easy, idxs

def generate_solvent_phase_samples(mol, ff):

    x0 = get_romol_conf(mol)

    masses = np.array([a.GetMass() for a in mol.GetAtoms()])
    num_workers = jax.device_count()
    state = enhanced_sampling.EnhancedState(mol, ff)
    water_system, water_coords, water_box, water_topology = builders.build_water_system(3.0)
    num_water_atoms = len(water_coords)
    afe = free_energy.AbsoluteFreeEnergy(mol, ff, decharge=True)
    ff_params = ff.get_ordered_params()
    ubps, params, masses, coords = afe.prepare_host_edge(ff_params, water_system, water_coords)

    temperature = 300
    dt = 1.5e-3
    friction = 1.0
    intg = lib.LangevinIntegrator(temperature, dt, friction, masses, 2021)
    
    pressure = 1.0
    interval = 10
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

    x0 = coords
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
    # num_steps = 200000
    num_steps = 5000
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

    # return xs[50:], boxes[50:], full_us[50:], params[-1]

    return xs, boxes, full_us, params[-1]

def test_condensed_phase():
    
    # generate gas-phase xs, with LJ terms only turned on
    # generate condensed-phase xs, batch RMSD align and re-weight


    #              xx x x <-- torsion indices
    #          01 23456 7 8 9
    mol = Chem.MolFromMolBlock(MOL_SDF, removeHs=False)
    # torsion_idxs = np.array([5,6,7,8])

    # this is broken
    #      0  12 3 4 5  6  7 8 | torsions are 2,3,6,7
    # smi = "C1=CC=C(C=C1)C(=O)O"
    # mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    # AllChem.EmbedMolecule(mol)
    # torsion_idxs = np.array([2,3,6,7])

    masses = np.array([a.GetMass() for a in mol.GetAtoms()])
    num_ligand_atoms = len(masses)

    ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_sc.py').read())
    ff = Forcefield(ff_handlers)

    cache_path = "cache.pkl"

    if not os.path.exists(cache_path):
        print("Generating cache")
        xs_easy, Us_easy, decharged_idxs = generate_gas_phase_samples(mol, ff)
        # there are duplicate samples here since xs_gas is re-weighted
        xs_solvent, boxes_solvent, Us_full, nb_params = generate_solvent_phase_samples(mol, ff)
        with open(cache_path, "wb") as fh:
            pickle.dump((xs_easy, Us_easy, decharged_idxs, xs_solvent, boxes_solvent, Us_full, nb_params), fh)
        print("Exiting..., please re-run")
        # assert 0

    with open(cache_path, "rb") as fh:
        xs_easy, Us_easy, decharged_idxs, xs_solvent, boxes_solvent, Us_full, nb_params = pickle.load(fh)

    xs_easy = xs_easy.reshape(-1, num_ligand_atoms, 3)

    log_numerator = []
    log_denominator = []

    params_i = nb_params[-num_ligand_atoms:] # ligand params
    params_j = nb_params[:-num_ligand_atoms] # water params

    def before_nrg(x_solvent, box_solvent):
        x_water = x_solvent[:-num_ligand_atoms] # water coords
        x_original = x_solvent[-num_ligand_atoms:] # ligand coords
        return nonbonded.nonbonded_off_diagonal(x_original, x_water, box_solvent, params_i, params_j)        

    batch_before_nrg_fn = jax.jit(jax.vmap(before_nrg))
    nrgs = batch_before_nrg_fn(xs_solvent, boxes_solvent)
    print(nrgs)

    def align_and_eval_delta_U(x_gas, x_solvent, box_solvent):
        x_gas_aligned = align_sample(x_gas, x_solvent) # replacement ligand coords
        x_water = x_solvent[:-num_ligand_atoms] # water coords
        return nonbonded.nonbonded_off_diagonal(x_gas_aligned, x_water, box_solvent, params_i, params_j)

    batch_after_nrg_fn = jax.jit(jax.vmap(align_and_eval_delta_U, in_axes=(0, None, None)))

    temperature = 300.0
    kT = temperature*BOLTZ

    # YTZ: don't use Us_easy.... probably not the term we want to use when we reweight to a different U_b

    # (ytz): optimization, compute energies only on the unique samples
    unique_decharged_kv = {}
    for i in decharged_idxs:
        if i not in unique_decharged_kv:
            unique_decharged_kv[i] = 0
        unique_decharged_kv[i] += 1

    unique_decharged_idxs = list(unique_decharged_kv.keys())
    unique_decharged_counts = list(unique_decharged_kv.values())
    unique_xs_decharged = xs_easy[unique_decharged_idxs]

    print(unique_xs_decharged.shape, "unique samples out of", len(decharged_idxs), "total samples")

    all_delta_us_unique = []

    for idx, (x_solvent, box_solvent) in enumerate(zip(xs_solvent, boxes_solvent)):
        print("before_nrgs", nrgs[idx], "frame", idx)
        delta_us = (batch_after_nrg_fn(unique_xs_decharged, x_solvent, box_solvent)- nrgs[idx])/kT
        all_delta_us_unique.append(delta_us)
        bins = 100
        plt.hist(np.exp(-delta_us), bins, density=True, alpha=0.2, label="frame_"+str(idx), weights=unique_decharged_counts)


    prefactor = np.sum(unique_decharged_counts)*len(all_delta_us_unique)
    unique_decharged_counts = np.array(unique_decharged_counts)
    all_delta_us_unique = np.array(all_delta_us_unique)
    # print("!!!", np.sum(all_delta_us_unique*unique_decharged_counts)
    # ratio = np.sum(np.exp(-all_delta_us_unique)*unique_decharged_counts)/prefactor
    # print("Ratio:", ratio)
    ratio = np.mean(np.average(np.exp(-all_delta_us_unique), axis=1, weights=unique_decharged_counts))

    print("Ratio:", ratio)

    assert 0
    print(all_delta_us_unique.shape)

    print((np.exp(-np.sum(all_delta_us_unique, axis=0))).shape)

    print("sum of nrgs", -np.sum(all_delta_us_unique, axis=0))
    assert 0
    print("before pow", np.exp(-np.sum(all_delta_us_unique, axis=0)))
    print("after pow", np.power(np.exp(-np.sum(all_delta_us_unique, axis=0)), unique_decharged_counts))
    # print("prod", np.prod(np.power(np.exp(-np.sum(all_delta_us_unique, axis=0)), unique_decharged_counts)))


    ratio = (1/prefactor)*np.prod(np.power(np.exp(-np.sum(all_delta_us_unique, axis=0)), unique_decharged_counts))  # sum along x0, y0, etc


    # let a, b be weights of two samples, and (x0, y0), (x1, y1), (x2, y2) be two samples
    # each drawn from three independent streams, weighted according to (a, b):
    # we wish to compute the mean: 1/(a+b+a+b+a+b) * a*exp(-x0)+b*exp(-y0)+a*x1+b*y1+a*x2+b*y2))
    # = 1/prefactor * exp(-(a*(x0+x1+x2)+b*(y0+y1+y2)))
    # = 1/prefactor * exp(-a*(x0+x1+x2))*exp(-b*(y0+y1+y2))
    # = 1/prefactor * exp(-(x0+x1+x2))^a * exp(-(y0+y1+y2)^b

    # let a, b be weights of two samples, and (x0, y0), (x1, y1), (x2, y2) be two samples
    # each drawn from three independent streams, weighted according to (a, b):
    # we wish to compute the mean: 1/(a+b+a+b+a+b) * exp(-(a*x0+b*y0+a*x1+b*y1+a*x2+b*y2))
    # = 1/prefactor * exp(-(a*(x0+x1+x2)+b*(y0+y1+y2)))
    # = 1/prefactor * exp(-a*(x0+x1+x2))*exp(-b*(y0+y1+y2))
    # = 1/prefactor * exp(-(x0+x1+x2))^a * exp(-(y0+y1+y2)^b

    # avg_delta_us = np.average(all_delta_us_unique, weights=unique_decharged_counts)

    # all_delta_us_unique = np.concatenate(all_delta_us_unique)

    print("ratio of Z_e/Z_f, computed by np.mean(np.exp(-delta_us)):", ratio)
    print(all_delta_us.shape)

    plt.xlabel("exp(-delta_u)")
    plt.ylabel("density")

    plt.show()
        # print("after_nrgs", batch_after_nrg(xs_gas, x_solvent, box_solvent).shape)

        # assert 0

        # for x_gas in xs_gas:



        #     x_gas_aligned = align_sample(x_gas, x_solvent) # replacement ligand coords
        #     nrg_after = nonbonded.nonbonded_off_diagonal(x_gas_aligned, x_water, box, params_i, params_j)

        #     print("Delta_U", nrg_after-nrg_before)
            # assert 0
            # print(nrg)
            # evaluate energy difference

    print(xs_gas.shape)
    print(xs_solvent.shape)



    # generate a rand

    assert 0

    x0 = get_romol_conf(mol)

    steps_per_batch = 100
    # num_batches = 20000
    num_batches = 2000

    temperature = 300
    kT = temperature*BOLTZ
    masses = np.array([a.GetMass() for a in mol.GetAtoms()])
    num_workers = jax.device_count()

    state = enhanced_sampling.EnhancedState(mol, ff)

    # xs_easy = enhanced_sampling.generate_samples(
    #     masses,
    #     x0,
    #     state.U_easy,
    #     temperature,
    #     steps_per_batch,
    #     num_batches,
    #     num_workers
    # )

    # writer = Chem.SDWriter("results.sdf")
    # num_atoms = mol.GetNumAtoms()
    # torsions = []

    # # discard first few batches for burn-in and reshape into a single flat array
    # xs_easy = xs_easy[:, 1000:, :, :]

    # batch_U_easy_fn = jax.pmap(jax.vmap(state.U_easy))
    # batch_U_decharged_fn = jax.pmap(jax.vmap(state.U_decharged))

    # Us_decharged = batch_U_decharged_fn(xs_easy)
    # Us_easy = batch_U_easy_fn(xs_easy)

    # log_numerator = -Us_decharged.reshape(-1)/kT
    # log_denominator = -Us_easy.reshape(-1)/kT

    # log_weights = log_numerator - log_denominator
    # weights = np.exp(log_weights - logsumexp(log_weights))

    # # sample from weights
    # sample_size = len(weights)
    # idxs = np.random.choice(np.arange(len(weights)), size=sample_size, p=weights)
    # unique_samples = len(set(idxs.tolist()))

    # Us_decharged = Us_decharged.reshape(-1)[idxs]
    # xs_decharged = xs_easy.reshape(-1, num_atoms, 3)[idxs]
    
    # assert len(Us_decharged) == len(xs_decharged)

    xs, boxes, us = generate_condensed_phase_samples(mol, ff)

    # generate samples in the solvent phase.
    # xs_solvent = 

def test_gas_phase():
    """
    This test attempts re-weighting in the gas-phase, where given a proposal
    distribution 
    """
    return

    #              xx x x <-- torsion indices
    #          01 23456 7 8 9
    mol = Chem.MolFromMolBlock(MOL_SDF, removeHs=False)
    torsion_idxs = np.array([5,6,7,8])

    # this is broken
    #      0  12 3 4 5  6  7 8 | torsions are 2,3,6,7
    # smi = "C1=CC=C(C=C1)C(=O)O"
    # mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    # AllChem.EmbedMolecule(mol)
    # torsion_idxs = np.array([2,3,6,7])

    masses = np.array([a.GetMass() for a in mol.GetAtoms()])

    ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_sc.py').read())
    ff = Forcefield(ff_handlers)
    x0 = get_romol_conf(mol)

    steps_per_batch = 100
    num_batches = 20000

    temperature = 300
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

    @jax.jit
    def get_torsion(x_t):
        cijkl = x_t[torsion_idxs]
        return bonded.signed_torsion_angle(*cijkl)

    batch_torsion_fn = jax.pmap(jax.vmap(get_torsion))
    batch_U_easy_fn = jax.pmap(jax.vmap(state.U_easy))
    batch_U_decharged_fn = jax.pmap(jax.vmap(state.U_decharged))

    kT = BOLTZ*temperature

    torsions_easy = batch_torsion_fn(xs_easy).reshape(-1)
    log_numerator = -batch_U_decharged_fn(xs_easy).reshape(-1)/kT
    log_denominator = -batch_U_easy_fn(xs_easy).reshape(-1)/kT

    log_weights = log_numerator - log_denominator
    weights = np.exp(log_weights - logsumexp(log_weights))

    # sample from weights
    sample_size = len(weights)*10
    idxs = np.random.choice(np.arange(len(weights)), size=sample_size, p=weights)
    unique_samples = len(set(idxs.tolist()))
    print("unique samples", unique_samples, "ratio", unique_samples/sample_size)

    torsions_reweight = torsions_easy[idxs]

    # assert that torsions sampled from U_decharged on one half are also consistent
    xs_decharged = enhanced_sampling.generate_samples(
        masses,
        x0,
        state.U_decharged,
        temperature,
        steps_per_batch,
        num_batches,
        num_workers
    )

    Us_reweight = batch_U_decharged_fn(xs_easy).reshape(-1)[idxs]
    Us_decharged = batch_U_decharged_fn(xs_decharged).reshape(-1)

    bins = np.linspace(250, 400, 50) # binned into 5kJ/mol chunks

    plt.xlabel("energy (kJ/mol)")
    plt.hist(Us_reweight, density=True, bins=bins, alpha=0.5, label="p_decharged (rw)")
    plt.hist(Us_decharged, density=True, bins=bins, alpha=0.5, label="p_decharged (md)")
    plt.legend()
    plt.savefig("rw_energy_distribution.png")
    plt.clf()

    Us_reweight = batch_U_decharged_fn(xs_decharged)

    torsions_decharged = batch_torsion_fn(xs_decharged).reshape(-1)
    torsions_reweight_lhs = torsions_reweight[np.nonzero(torsions_reweight < 0)]

    plt.xlabel("torsion_angle")
    plt.hist(torsions_easy, density=True, bins=50, label='p_easy', alpha=0.5)
    plt.hist(torsions_reweight, density=True, bins=50, label='p_decharged (rw)', alpha=0.5)
    plt.hist(torsions_reweight_lhs, density=True, bins=25, label='p_decharged (rw, lhs only)', alpha=0.5)
    plt.hist(torsions_decharged, density=True, bins=25, label='p_decharged (md)', alpha=0.5)
    plt.legend()
    plt.savefig("rw_torsion_distribution.png")

    # verify that the histogram of torsions_reweight is
    # 1) symmetric about theta = 0
    # 2) agrees with that of a fresh simulation using U_decharged

    torsions_reweight_lhs, edges = np.histogram(torsions_reweight, bins=50, range=(-np.pi, 0), density=True)
    torsions_reweight_rhs, edges = np.histogram(torsions_reweight, bins=50, range=( 0, np.pi), density=True)

    # test symmetry about theta=0
    assert np.mean((torsions_reweight_lhs - torsions_reweight_rhs[::-1])**2) < 1e-2

    torsions_decharged_lhs, edges = np.histogram(torsions_decharged, bins=50, range=(-np.pi, 0), density=True)

    # test against directly simulated results using U_decharged
    assert np.mean((torsions_reweight_lhs - torsions_decharged_lhs[::-1])**2) < 1e-2



if __name__ == "__main__":
    test_gas_phase()
    test_condensed_phase()