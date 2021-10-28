# Test enhanced sampling protocols

from math import pi
import os
import pickle

from jax.config import config

config.update("jax_enable_x64", True)
import jax

from rdkit import Chem

from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers

from timemachine.potentials import bonded, nonbonded
from timemachine.constants import BOLTZ
from timemachine.potentials import rmsd

import numpy as np
import matplotlib.pyplot as plt

from md import enhanced_sampling

from scipy.special import logsumexp

# from fe.pdb_writer import PDBWriter
from fe import model_utils

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

# MOL_SDF = """
#   Mrv2115 10152122132D

#  12 12  0  0  0  0            999 V2000
#    -0.3669    0.6543    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
#    -1.1224    0.6543    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
#    -1.5002    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
#    -1.1224   -0.6543    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
#    -0.3669   -0.6543    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
#     0.0109    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
#    -0.0166    1.2610    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
#     0.7114    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
#    -0.0166   -1.2610    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
#    -1.4727   -1.2610    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
#    -2.2007    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
#    -1.4727    1.2610    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
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
    conf = conf * 10
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


def test_condensed_phase():

    # generate gas-phase xs, with LJ terms only turned on
    # generate condensed-phase xs, batch RMSD align and re-weight

    #              xx x x <-- torsion indices
    #          01 23456 7 8 9
    mol = Chem.MolFromMolBlock(MOL_SDF, removeHs=False)
    # torsion_idxs = np.array([5,6,7,8])

    masses = np.array([a.GetMass() for a in mol.GetAtoms()])
    num_ligand_atoms = len(masses)

    ff_handlers = deserialize_handlers(open("ff/params/smirnoff_1_1_0_sc.py").read())
    ff = Forcefield(ff_handlers)

    cache_path = "cache.pkl"

    temperature = 300.0

    state = enhanced_sampling.EnhancedState(mol, ff)

    U_gas = state.U_full

    if not os.path.exists(cache_path):
        print("Generating cache")

        # generate samples in some target state
        (
            gas_counts,
            xs_gas_unique,
            Us_gas_unique,
        ) = enhanced_sampling.generate_gas_phase_samples(mol, ff, temperature, U_gas)

        # (
        #     xs_solvent,
        #     boxes_solvent,
        #     Us_full,
        #     nb_params,
        #     topology_objs,
        # ) = enhanced_sampling.generate_solvent_phase_samples(mol, ff, temperature)
        # gas_counts, xs_gas_unique, Us_gas_unique = generate_gas_phase_samples(mol, ff, temperature, U_gas)
        with open(cache_path, "wb") as fh:
            pickle.dump(
                (
                    gas_counts,
                    xs_gas_unique,
                    Us_gas_unique,
                    # xs_solvent,
                    # boxes_solvent,
                    # Us_full,
                    # nb_params,
                    # topology_objs,
                ),
                fh,
            )

    with open(cache_path, "rb") as fh:
        (
            gas_counts,
            xs_gas_unique,
            Us_gas_unique,
            # xs_solvent,
            # boxes_solvent,
            # Us_full,
            # nb_params,
            # topology_objs,
        ) = pickle.load(fh)

    ubps, params, masses, coords, box = enhanced_sampling.get_solvent_phase_system(
        mol, ff
    )

    xs_solvent = []
    boxes_solvent = []

    for _, x, b in enhanced_sampling.generate_solvent_phase_samples(
        ubps, params, masses, coords, box, temperature
    ):
        xs_solvent.append(x)
        boxes_solvent.append(b)

    xs_solvent = np.array(xs_solvent)
    boxes_solvent = np.array(boxes_solvent)

    print(xs_solvent.shape)
    print(boxes_solvent.shape)

    nb_params = params[-1]

    gas_counts = np.array(gas_counts)

    # model_utils.generate_imaged_topology(
    # topology_objs, xs_solvent[0], boxes_solvent[0], f"solvent_debug.pdb"
    # )

    params_i = nb_params[-num_ligand_atoms:]  # ligand params
    params_j = nb_params[:-num_ligand_atoms]  # water params

    state = enhanced_sampling.EnhancedState(mol, ff)

    print("params_i", params_i)
    print("params_j", params_j)

    def before_U_k(x_solvent, box_solvent):
        x_water = x_solvent[:-num_ligand_atoms]  # water coords
        x_original = x_solvent[-num_ligand_atoms:]  # ligand coords
        U_k = nonbonded.nonbonded_off_diagonal(
            x_original, x_water, box_solvent, params_i, params_j
        )
        return U_k

    @jax.jit
    def U_a(x):
        return state.U_full(x)

    # (note): we know what the energy is actually, no need to compute it
    @jax.jit
    def U_b(x):
        return U_gas(x)

    batch_before_U_k_fn = jax.jit(jax.vmap(before_U_k))
    batch_U_a_fn = jax.jit(jax.vmap(U_a))
    batch_U_b_fn = jax.jit(jax.vmap(U_b))
    before_U_k_nrgs = batch_before_U_k_fn(xs_solvent, boxes_solvent)

    def after_U_k(x_gas, x_solvent, box_solvent):
        x_gas_aligned = align_sample(x_gas, x_solvent)  # align gas phase conformer
        x_water = x_solvent[:-num_ligand_atoms]  # water coords
        U_k = nonbonded.nonbonded_off_diagonal(
            x_gas_aligned, x_water, box_solvent, params_i, params_j
        )
        return U_k

    batch_after_U_k_fn = jax.jit(jax.vmap(after_U_k, in_axes=(0, None, None)))
    kT = temperature * BOLTZ

    print(
        f"{xs_gas_unique.shape} unique samples out of {np.sum(gas_counts)} total samples"
    )

    all_delta_us_unique = []

    writer = Chem.SDWriter("test2.sdf")

    torsion_idxs = np.array([5, 6, 7, 8])

    @jax.jit
    def get_torsion(x_t):
        cijkl = x_t[torsion_idxs]
        return bonded.signed_torsion_angle(*cijkl)

    batch_get_torsion = jax.jit(jax.vmap(get_torsion))
    gas_torsions = batch_get_torsion(xs_gas_unique)

    # verify that these are both bimodal
    # plt.hist(gas_torsions, density=True, weights=gas_counts, bins=50)
    # plt.show()

    print(
        "average gas-phase torsion",
        np.average(gas_torsions, weights=gas_counts),
    )

    all_ratios = []
    all_torsions = []
    solvent_torsions = []

    print(f"{len(xs_solvent)} solvent frames")

    all_weights = []
    all_exp_dus = []

    for idx, (x_solvent, box_solvent) in enumerate(zip(xs_solvent, boxes_solvent)):

        # writer.write(make_conformer(mol, x_solvent[-num_ligand_atoms:]))
        # for x_g in xs_gas_unique[:100]:
        #     x_gas_aligned = align_sample(x_g, x_solvent) # align gas phase conformer
        #     new_mol = make_conformer(mol, np.asarray(x_gas_aligned))
        #     print("rmsd after alignment", np.linalg.norm(x_gas_aligned - x_solvent[-num_ligand_atoms:]))
        #     writer.write(new_mol)

        # writer.flush()
        # writer.close()

        solvent_torsion = get_torsion(x_solvent[-num_ligand_atoms:])

        solvent_torsions.append(solvent_torsion)
        before_U_k_sample = before_U_k_nrgs[idx]
        after_U_k_samples = batch_after_U_k_fn(xs_gas_unique, x_solvent, box_solvent)

        # before_U_a_sample = U_a(x_solvent[-num_ligand_atoms:])
        # before_U_b_samples = batch_U_b_fn(xs_gas_unique)
        # before_U_samples = before_U_k_sample + before_U_a_sample + before_U_b_samples
        # after_U_a_samples = batch_U_a_fn(xs_gas_unique)
        # after_U_b_sample = U_b(x_solvent[-num_ligand_atoms:])
        # after_U_samples = after_U_k_samples + after_U_a_samples + after_U_b_sample

        delta_us = (after_U_k_samples - before_U_k_sample) / kT

        weights = gas_counts * np.exp(-delta_us)
        all_weights.append(weights)

        max_weight_arg = np.argmax(weights)
        max_weight = np.amax(weights)
        min_weight = np.amin(weights)
        max_torsion = gas_torsions[max_weight_arg]

        print(
            f"solvent_torsion {solvent_torsion} max_torsion {max_torsion} max_weight: {max_weight} min_weight: {min_weight}"
        )

        # plt.hist(
        #     gas_torsions,
        #     bins=np.linspace(-np.pi, np.pi, 100),
        #     density=True,
        #     alpha=0.5,
        #     weights=weights / np.sum(weights),
        # )
        # plt.show()

        all_delta_us_unique.append(delta_us)
        all_torsions.append(gas_torsions)

        # print(
        #     f"iteration {idx} before_U_k_sample {before_U_k_sample} unique_weights, max {np.amax(weights)} min {np.amin(weights)} ratio estimate: {ratio_estimate} dG_estimate {dG_estimate} torsion_estimate {np.mean(all_torsions)}"
        # )

    # given K streams and N samples per stream with different counts
    # compute 1/(total_counts_per_stream*number_of_streams) * sum_j^K sum_i^N c_i exp(-delta_u_ij)
    all_delta_us_unique = np.array(all_delta_us_unique)
    total_counts = len(all_delta_us_unique) * np.sum(gas_counts)
    ratio = (all_delta_us_unique * np.expand_dims(gas_counts, axis=0)) / total_counts

    print("ratio", ratio)
    ratio = np.mean(
        np.average(np.exp(-all_delta_us_unique), axis=1, weights=gas_counts)
    )
    print("ratio", ratio)

    all_weights = np.array(all_weights).reshape(-1)
    sum_of_weights = np.sum(all_weights)
    normalized_weights = all_weights / sum_of_weights

    all_torsions = np.asarray(all_torsions).reshape(-1)

    assert all_torsions.shape == normalized_weights.shape

    print("ratio of Z_e/Z_f, computed by np.mean(np.exp(-delta_us)):", ratio)
    print("avg_torsion", np.average(all_torsions, weights=normalized_weights))

    # plt.xlabel("exp(-delta_u)")
    # plt.ylabel("density")

    plt.hist(
        all_torsions,
        bins=np.linspace(-np.pi, np.pi, 100),
        density=True,
        alpha=0.5,
        weights=normalized_weights,
    )
    plt.savefig("enhanced_torsions.png")

    plt.clf()

    solvent_torsions = np.asarray(solvent_torsions)

    plt.hist(
        solvent_torsions, bins=np.linspace(-np.pi, np.pi, 100), density=True, alpha=0.5
    )

    plt.savefig("solvent_torsions.png")

    assert (np.sum(normalized_weights) - 1) < 1e-6


def test_adaptive_condensed_phase():

    mol = Chem.MolFromMolBlock(MOL_SDF, removeHs=False)
    masses = np.array([a.GetMass() for a in mol.GetAtoms()])
    num_ligand_atoms = len(masses)

    ff_handlers = deserialize_handlers(open("ff/params/smirnoff_1_1_0_sc.py").read())
    ff = Forcefield(ff_handlers)

    cache_path = "cache.pkl"

    temperature = 300.0

    state = enhanced_sampling.EnhancedState(mol, ff)

    U_gas = state.U_full

    if not os.path.exists(cache_path):
        print("Generating cache")

        # generate samples in some target state
        (
            gas_counts,
            xs_gas_unique,
            Us_gas_unique,
        ) = enhanced_sampling.generate_gas_phase_samples(mol, ff, temperature, U_gas)

        with open(cache_path, "wb") as fh:
            pickle.dump(
                (
                    gas_counts,
                    xs_gas_unique,
                    Us_gas_unique,
                ),
                fh,
            )

    with open(cache_path, "rb") as fh:
        (gas_counts, xs_gas_unique, Us_gas_unique) = pickle.load(fh)

    ubps, params, masses, coords, box = enhanced_sampling.get_solvent_phase_system(
        mol, ff
    )

    # print(xs_solvent.shape)
    # print(boxes_solvent.shape)

    nb_params = params[-1]

    gas_counts = np.array(gas_counts)

    params_i = nb_params[-num_ligand_atoms:]  # ligand params
    params_j = nb_params[:-num_ligand_atoms]  # water params

    state = enhanced_sampling.EnhancedState(mol, ff)

    print("params_i", params_i)
    print("params_j", params_j)

    @jax.jit
    def before_U_k(x_solvent, box_solvent):
        x_water = x_solvent[:-num_ligand_atoms]  # water coords
        x_original = x_solvent[-num_ligand_atoms:]  # ligand coords
        U_k = nonbonded.nonbonded_off_diagonal(
            x_original, x_water, box_solvent, params_i, params_j
        )
        return U_k

    @jax.jit
    def U_a(x):
        return state.U_full(x)

    # (note): we know what the energy is actually, no need to compute it
    @jax.jit
    def U_b(x):
        return U_gas(x)

    batch_before_U_k_fn = jax.jit(jax.vmap(before_U_k))
    batch_U_a_fn = jax.jit(jax.vmap(U_a))
    batch_U_b_fn = jax.jit(jax.vmap(U_b))
    # before_U_k_nrgs = batch_before_U_k_fn(xs_solvent, boxes_solvent)

    def after_U_k(x_gas, x_solvent, box_solvent):
        x_gas_aligned = align_sample(x_gas, x_solvent)  # align gas phase conformer
        x_water = x_solvent[:-num_ligand_atoms]  # water coords
        U_k = nonbonded.nonbonded_off_diagonal(
            x_gas_aligned, x_water, box_solvent, params_i, params_j
        )
        return U_k

    batch_after_U_k_fn = jax.jit(jax.vmap(after_U_k, in_axes=(0, None, None)))
    kT = temperature * BOLTZ

    print(
        f"{xs_gas_unique.shape} unique samples out of {np.sum(gas_counts)} total samples"
    )

    all_delta_us_unique = []

    # writer = Chem.SDWriter("test2.sdf")

    torsion_idxs = np.array([5, 6, 7, 8])

    @jax.jit
    def get_torsion(x_t):
        cijkl = x_t[torsion_idxs]
        return bonded.signed_torsion_angle(*cijkl)

    batch_get_torsion = jax.jit(jax.vmap(get_torsion))
    gas_torsions = batch_get_torsion(xs_gas_unique)

    # verify that these are both bimodal
    # plt.hist(gas_torsions, density=True, weights=gas_counts, bins=50)
    # plt.show()

    assert np.abs(np.average(gas_torsions, weights=gas_counts)) < 0.1

    all_ratios = []
    all_torsions = []
    solvent_torsions = []

    # print(f"{len(xs_solvent)} solvent frames")

    all_weights = []

    num_batches = 2000
    sample_generator = enhanced_sampling.generate_solvent_phase_samples(
        ubps, params, masses, coords, box, temperature, num_batches=num_batches
    )

    torsion_weights = []
    torsion_angles = []

    next_x = None

    # (ytz):
    gas_weights = gas_counts / np.sum(gas_counts)

    K = 1000
    for iteration in range(num_batches):
        # kick off
        # print("SENDING", next_x)
        x_solvent, box_solvent = sample_generator.send(next_x)
        before_U_k_sample = before_U_k(x_solvent, box_solvent)
        pi_x = np.exp(-before_U_k_sample / kT)

        # writer.write(make_conformer(mol, x_solvent[-num_ligand_atoms:]))
        # for x_g in xs_gas_unique[:100]:
        #     x_gas_aligned = align_sample(x_g, x_solvent) # align gas phase conformer
        #     new_mol = make_conformer(mol, np.asarray(x_gas_aligned))
        #     print("rmsd after alignment", np.linalg.norm(x_gas_aligned - x_solvent[-num_ligand_atoms:]))
        #     writer.write(new_mol)

        # writer.flush()
        # writer.close()

        solvent_torsion = get_torsion(x_solvent[-num_ligand_atoms:])
        solvent_torsions.append(solvent_torsion)

        gas_samples_yi = xs_gas_unique[
            np.random.choice(np.arange(len(xs_gas_unique)), size=K, p=gas_weights)
        ]

        after_U_y_i_samples = batch_after_U_k_fn(gas_samples_yi, x_solvent, box_solvent)

        pi_yi = np.exp(-after_U_y_i_samples / kT)
        normalized_pi_yi = pi_yi / np.sum(pi_yi)
        new_y = gas_samples_yi[np.random.choice(np.arange(K), p=normalized_pi_yi)]

        # log_pi_yi = -after_U_y_i_samples / kT
        # normalized_pi_yi = np.exp(log_pi_yi - logsusum(pi_yi)
        # new_y = gas_samples_yi[np.random.choice(np.arange(K), p=normalized_pi_yi)]

        gas_samples_x_i_sub_1 = xs_gas_unique[
            np.random.choice(np.arange(len(xs_gas_unique)), size=K - 1, p=gas_weights)
        ]

        # align new_y unto the ligand present in x_solvent and returns an aligned new_y
        new_y_aligned = align_sample(new_y, x_solvent)
        x_solvent_new_y = np.copy(x_solvent)
        x_solvent_new_y[-num_ligand_atoms:] = new_y_aligned

        after_U_x_i_samples = batch_after_U_k_fn(
            gas_samples_x_i_sub_1, x_solvent_new_y, box_solvent
        )

        pi_x_i_sub_1 = np.exp(-after_U_x_i_samples / kT)
        pi_x_combined = np.concatenate([pi_x_i_sub_1, pi_x])

        print(np.sum(pi_yi), np.sum(pi_x_combined))

        ratio = np.sum(pi_yi) / np.sum(pi_x_combined)

        if np.random.rand() < ratio:
            # accept
            print("accept")
            next_x = x_solvent_new_y
        else:
            # reject
            print("reject")
            next_x = x_solvent

        continue

        # weights_yi/np.sum(weights_yi)

        # numerator

        delta_us = (after_U_k_samples - before_U_k_sample) / kT

        weights = gas_counts * np.exp(-delta_us)
        sum_of_weights = np.sum(weights)
        normalized_weights = weights / sum_of_weights

        max_weight_arg = np.argmax(weights)
        max_weight = np.amax(weights)
        min_weight = np.amin(weights)
        max_torsion = gas_torsions[max_weight_arg]

        print(
            f"solvent_torsion {solvent_torsion} max_torsion {max_torsion} max_weight: {max_weight} min_weight: {min_weight}"
        )

        # plt.hist(
        #     gas_torsions,
        #     bins=np.linspace(-np.pi, np.pi, 100),
        #     density=True,
        #     alpha=0.5,
        #     weights=weights / np.sum(weights),
        # )
        # plt.show()

        all_delta_us_unique.append(delta_us)
        all_torsions.append(gas_torsions)

        # draw a new sample random

        choice_idx = np.random.choice(
            np.arange(len(xs_gas_unique)), p=normalized_weights
        )
        aligned_x_gas = align_sample(xs_gas_unique[choice_idx], x_solvent)

        new_x_solvent = np.asarray(x_solvent)
        new_x_solvent[-num_ligand_atoms:] = aligned_x_gas

        torsion_weights.append(normalized_weights)
        torsion_angles.append(gas_torsions)

        # sample_generator.send(new_x_solvent)
        next_x = new_x_solvent

        # print(
        #     f"iteration {idx} before_U_k_sample {before_U_k_sample} unique_weights, max {np.amax(weights)} min {np.amin(weights)} ratio estimate: {ratio_estimate} dG_estimate {dG_estimate} torsion_estimate {np.mean(all_torsions)}"
        # )

        tmp_torsion_weights = np.array(torsion_weights).reshape(-1)
        tmp_torsion_angles = np.array(torsion_angles).reshape(-1)

        plt.clf()
        plt.hist(
            tmp_torsion_angles,
            bins=np.linspace(-np.pi, np.pi, 100),
            density=True,
            weights=tmp_torsion_weights,
        )
        plt.title(f"iteration {iteration}")
        plt.savefig("enhanced_sampling.png")

        # plt.show()

    # given K streams and N samples per stream with different counts
    # compute 1/(total_counts_per_stream*number_of_streams) * sum_j^K sum_i^N c_i exp(-delta_u_ij)
    all_delta_us_unique = np.array(all_delta_us_unique)
    total_counts = len(all_delta_us_unique) * np.sum(gas_counts)
    ratio = (all_delta_us_unique * np.expand_dims(gas_counts, axis=0)) / total_counts

    print("ratio", ratio)
    ratio = np.mean(
        np.average(np.exp(-all_delta_us_unique), axis=1, weights=gas_counts)
    )
    print("ratio", ratio)

    all_weights = np.array(all_weights).reshape(-1)
    sum_of_weights = np.sum(all_weights)
    normalized_weights = all_weights / sum_of_weights

    all_torsions = np.asarray(all_torsions).reshape(-1)

    assert all_torsions.shape == normalized_weights.shape

    print("ratio of Z_e/Z_f, computed by np.mean(np.exp(-delta_us)):", ratio)
    print("avg_torsion", np.average(all_torsions, weights=normalized_weights))

    # plt.xlabel("exp(-delta_u)")
    # plt.ylabel("density")

    plt.hist(
        all_torsions,
        bins=np.linspace(-np.pi, np.pi, 100),
        density=True,
        alpha=0.5,
        weights=normalized_weights,
    )
    plt.savefig("enhanced_torsions.png")

    plt.clf()

    solvent_torsions = np.asarray(solvent_torsions)

    plt.hist(
        solvent_torsions, bins=np.linspace(-np.pi, np.pi, 100), density=True, alpha=0.5
    )

    plt.savefig("solvent_torsions.png")

    assert (np.sum(normalized_weights) - 1) < 1e-6


if __name__ == "__main__":
    # test_condensed_phase()
    test_adaptive_condensed_phase()
