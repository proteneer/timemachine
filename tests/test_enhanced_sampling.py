# Test enhanced sampling protocols
from jax.config import config; config.update("jax_enable_x64", True)
import jax

from rdkit import Chem
from rdkit.Chem import AllChem

from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers
from fe import topology
from fe.utils import get_romol_conf

from timemachine.potentials import bonded, nonbonded
from timemachine.integrator import langevin_coefficients
from timemachine.constants import BOLTZ

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


def test_gas_phase():
    """
    This test attempts re-weighting in the gas-phase, where given a proposal
    distribution 
    """

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
    print("unique samples", len(unique_samples), "ratio", len(sample_size)/len(unique_samples))

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