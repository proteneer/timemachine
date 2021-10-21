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
from timemachine.lib import potentials
from timemachine.potentials import rmsd

from md.barostat.utils import get_group_indices, get_bond_list

import numpy as np
import matplotlib.pyplot as plt

from md import enhanced_sampling

from scipy.special import logsumexp
from fe.pdb_writer import PDBWriter
from fe import model_utils

import mdtraj

# MOL_SDF = """
#   Mrv2115 09292117373D          

#  15 16  0  0  0  0            999 V2000
#    -1.3280    3.9182   -1.1733 F   0  0  0  0  0  0  0  0  0  0  0  0
#     0.4924    2.9890   -0.9348 C   0  0  0  0  0  0  0  0  0  0  0  0
#     1.6519    3.7878   -0.9538 C   0  0  0  0  0  0  0  0  0  0  0  0
#     2.9215    3.2010   -0.8138 C   0  0  0  0  0  0  0  0  0  0  0  0
#     3.0376    1.8091   -0.6533 C   0  0  0  0  0  0  0  0  0  0  0  0
#     1.8835    1.0062   -0.6230 C   0  0  0  0  0  0  0  0  0  0  0  0
#     0.6026    1.5878   -0.7603 C   0  0  0  0  0  0  0  0  0  0  0  0
#    -0.5399    0.7586   -0.7175 C   0  0  0  0  0  0  0  0  0  0  0  0
#    -1.2257    0.5460    0.5040 C   0  0  0  0  0  0  0  0  0  0  0  0
#    -0.6191    1.4266    2.2631 F   0  0  0  0  0  0  0  0  0  0  0  0
#    -2.3596   -0.2866    0.5420 C   0  0  0  0  0  0  0  0  0  0  0  0
#    -2.8171   -0.9134   -0.6298 C   0  0  0  0  0  0  0  0  0  0  0  0
#    -2.1427   -0.7068   -1.8452 C   0  0  0  0  0  0  0  0  0  0  0  0
#    -1.0087    0.1257   -1.8951 C   0  0  0  0  0  0  0  0  0  0  0  0
#    -0.0878    0.3825   -3.7175 F  0  0  0  0  0  0  0  0  0  0  0  0
#   2  3  4  0  0  0  0
#   3  4  4  0  0  0  0
#   4  5  4  0  0  0  0
#   5  6  4  0  0  0  0
#   6  7  4  0  0  0  0
#   2  7  4  0  0  0  0
#   7  8  1  0  0  0  0
#   8  9  4  0  0  0  0
#   9 11  4  0  0  0  0
#  11 12  4  0  0  0  0
#  12 13  4  0  0  0  0
#  13 14  4  0  0  0  0
#   8 14  4  0  0  0  0
#   9 10  1  0  0  0  0
#   1  2  1  0  0  0  0
#  14 15  1  0  0  0  0
# M  END
# $$$$"""

MOL_SDF = """
  Mrv2115 10152122132D          

 12 12  0  0  0  0            999 V2000
   -0.3669    0.6543    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.1224    0.6543    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.5002    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.1224   -0.6543    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.3669   -0.6543    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0109    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0166    1.2610    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.7114    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0166   -1.2610    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.4727   -1.2610    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.2007    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.4727    1.2610    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0  0  0  0
  2  3  1  0  0  0  0
  3  4  2  0  0  0  0
  4  5  1  0  0  0  0
  5  6  2  0  0  0  0
  6  1  1  0  0  0  0
  1  7  1  0  0  0  0
  6  8  1  0  0  0  0
  5  9  1  0  0  0  0
  4 10  1  0  0  0  0
  3 11  1  0  0  0  0
  2 12  1  0  0  0  0
M  END
$$$$"""


def generate_solvent_phase_samples(mol, ff, temperature, k_core):

    x0 = get_romol_conf(mol)

    masses = np.array([a.GetMass() for a in mol.GetAtoms()])
    num_workers = jax.device_count()
    state = enhanced_sampling.EnhancedState(mol, ff)
    water_system, water_coords, water_box, water_topology = builders.build_water_system(3.0)
    water_box = water_box + np.eye(3)*0.5 # add a small margin around the box for stability
    num_water_atoms = len(water_coords)
    afe = free_energy.AbsoluteReplacementFreeEnergy(mol, ff)
    ff_params = ff.get_ordered_params()
    ubps, params, masses, coords = afe.prepare_host_edge(ff_params, water_system, water_coords)

    dt = 1.5e-3
    friction = 1.0
    
    pressure = 1.0
    interval = 5

    box = water_box
    host_coords = coords[:num_water_atoms]
    new_host_coords = minimizer.minimize_host_4d([mol], water_system, host_coords, ff, water_box)
    coords[:num_water_atoms] = new_host_coords

    # tbd, add restraints

    bps = []
    for p, bp in zip(params, ubps):
        bps.append(bp.bind(p))

    num_ligand_atoms = mol.GetNumAtoms()
    combined_core_idxs = np.stack([
        num_water_atoms + np.arange(num_ligand_atoms),
        num_water_atoms + num_ligand_atoms + np.arange(num_ligand_atoms)
    ], axis=1).astype(np.int32)
    
    core_params = np.zeros_like(combined_core_idxs).astype(np.float64)
    core_params[:, 0] = k_core

    B = len(combined_core_idxs)

    restraint_potential = potentials.HarmonicBond(
        combined_core_idxs,
    )

    restraint_potential.bind(core_params)
    bps.append(restraint_potential)

    all_impls = [bp.bound_impl(np.float32) for bp in bps]
    
    intg_equil = lib.LangevinIntegrator(temperature, 1e-4, friction, masses, 2021)
    intg_equil_impl = intg_equil.impl()

    # equilibration/minimization doesn't need a barostat
    equil_ctxt = custom_ops.Context(
        coords,
        np.zeros_like(coords),
        box,
        intg_equil_impl,
        all_impls,
        None
    )

    print("start equilibration")

    lamb = 0.0
    equil_schedule = np.ones(50000)*lamb
    equil_ctxt.multiple_steps(equil_schedule)


    x0 = equil_ctxt.get_x_t()
    v0 = np.zeros_like(x0)
    # production

    intg = lib.LangevinIntegrator(temperature, dt, friction, masses, 2021)
    intg_impl = intg.impl()

    # reset impls
    all_impls = [bp.bound_impl(np.float32) for bp in bps]

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
    barostat_impl = barostat.impl(all_impls)
    # barostat_impl = None

    ctxt = custom_ops.Context(
        x0,
        v0,
        box,
        intg_impl,
        all_impls,
        barostat_impl
    )

    lamb = 0.0
    num_steps = 1000000
    lambda_windows = np.array([0.0, 1.0])

    u_interval = 500
    x_interval = 500

    full_Us, xs, boxes = ctxt.multiple_steps_U(
        lamb,
        num_steps,
        lambda_windows,
        u_interval,
        x_interval
    )

    # take off 10% for burn in
    burn_in = len(full_Us)//10
    full_Us = full_Us[burn_in:]

    delta_Us = full_Us[:, 1] - full_Us[:, 0]
    delta_Us = np.where(delta_Us < -5000, np.inf, delta_Us) # anything less than 5000 kJ/mol is likely an overflow

    kT = temperature*BOLTZ

    reduced_delta_us = delta_Us/kT

    np.savez("delta_us.npz", dus=reduced_delta_us)

    ratio_estimate = np.mean(np.exp(-reduced_delta_us))
    dG_estimate = -kT*(logsumexp(-reduced_delta_us) - np.log(len(reduced_delta_us)))

    print("k_core", k_core, "dG_estimate", dG_estimate)

def test_condensed_phase():
    
    mol = Chem.MolFromMolBlock(MOL_SDF, removeHs=False)

    ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_sc.py').read())
    ff = Forcefield(ff_handlers)

    temperature = 300.0

    # for k_core in [1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]:
    for k_core in [100.0]:
        generate_solvent_phase_samples(mol, ff, temperature, k_core)

if __name__ == "__main__":
    # test_gas_phase()
    test_condensed_phase()