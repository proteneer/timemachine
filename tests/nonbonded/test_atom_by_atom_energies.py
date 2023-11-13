import numpy as np
import pytest
from common import assert_energy_arrays_match

from timemachine.constants import DEFAULT_TEMP
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.lib import custom_ops
from timemachine.md import builders
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.md.exchange.exchange_mover import BDExchangeMove
from timemachine.potentials import HarmonicBond, Nonbonded


@pytest.mark.memcheck
@pytest.mark.parametrize(
    "precision,atol,rtol,threshold", [(np.float64, 1e-8, 1e-8, 1e16), (np.float32, 1e-4, 2e-3, 1e8)]
)
def test_nonbonded_atom_by_atom_energies_match(precision, atol, rtol, threshold):
    """Verify that if looking at not computing the subsets of energies matches the references"""
    rng = np.random.default_rng(2023)
    ff = Forcefield.load_default()
    system, conf, box, _ = builders.build_water_system(4.0, ff.water_ff)
    bps, _ = openmm_deserializer.deserialize_system(system, cutoff=1.2)
    nb = next(bp for bp in bps if isinstance(bp.potential, Nonbonded))
    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    all_group_idxs = get_group_indices(get_bond_list(bond_pot), conf.shape[0])

    # Get the indices of atoms to compute per atom energies for
    target_atoms = rng.choice(all_group_idxs)
    N = conf.shape[0]

    params = nb.params

    beta = nb.potential.beta
    cutoff = nb.potential.cutoff
    func = custom_ops.atom_by_atom_energies_f32
    if precision == np.float64:
        func = custom_ops.atom_by_atom_energies_f64

    target_atoms = np.array(target_atoms).astype(np.int32)
    # Shuffling all atoms will break the test as the atom_by_atom_energies function assumes
    # the ordering is sequential
    all_atoms = np.arange(N).astype(np.int32)

    mover = BDExchangeMove(beta, cutoff, params, all_group_idxs, DEFAULT_TEMP)
    ref_energies = mover.U_fn_unsummed(conf, box, target_atoms, all_atoms)
    comp_energies = func(target_atoms, conf, params, box, beta, cutoff)
    assert ref_energies.shape == comp_energies.shape
    assert_energy_arrays_match(ref_energies, comp_energies, atol=atol, rtol=rtol, threshold=threshold)
