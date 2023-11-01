import numpy as np
import pytest

from timemachine.constants import DEFAULT_TEMP
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.lib import custom_ops
from timemachine.md import builders
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.potentials import HarmonicBond, Nonbonded


@pytest.mark.memcheck
@pytest.mark.parametrize("moves", [1, 2, 10])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("seed", [2023])
def test_two_clashy_water_moves(moves, precision, seed):
    """Given two waters directly on top of each other in a box, the exchange mover should accept almost any move"""
    ff = Forcefield.load_default()
    system, conf, box, topo = builders.build_water_system(1.0, ff.water_ff)
    bps, _ = openmm_deserializer.deserialize_system(system, cutoff=1.2)

    nb = next(bp for bp in bps if isinstance(bp.potential, Nonbonded))
    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    all_group_idxs = get_group_indices(get_bond_list(bond_pot), conf.shape[0])

    # Get first to mols
    group_idxs = all_group_idxs

    conf_idxs = np.array(group_idxs).reshape(-1)

    conf = conf[conf_idxs]
    # Set the two waters ontop of each other
    conf[group_idxs[0], :] = conf[group_idxs[1], :]

    # box = np.eye(3) * 100.0

    N = conf.shape[0]

    params = nb.params[conf_idxs]

    cutoff = nb.potential.cutoff
    klass = custom_ops.BDExchangeMove_f32
    if precision == np.float64:
        klass = custom_ops.BDExchangeMove_f64

    bdem = klass(N, group_idxs, params, DEFAULT_TEMP, nb.potential.beta, cutoff, seed)

    num_steps = 1
    last_conf = conf
    for _ in range(moves):
        x_move, x_box = bdem.move(last_conf, box, num_steps)
        # The box will never change
        np.testing.assert_array_equal(box, x_box)
        num_moved = 0
        for mol_idxs in group_idxs:
            if not np.all(x_move[mol_idxs] == last_conf[mol_idxs]):
                num_moved += 1
        assert num_moved <= 1, "More than one mol moved, something is wrong"
        last_conf = x_move
