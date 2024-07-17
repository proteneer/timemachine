from importlib import resources

import numpy as np
import pytest
from common import fixed_overflowed

from timemachine.constants import DEFAULT_PRESSURE, DEFAULT_TEMP
from timemachine.fe.free_energy import HostConfig
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat, custom_ops
from timemachine.lib.fixed_point import fixed_to_float
from timemachine.md import builders, minimizer
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.potentials import HarmonicBond, Nonbonded, SummedPotential
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology

pytestmark = [pytest.mark.memcheck]


@pytest.mark.parametrize(
    "precision,rtol,atol",
    [pytest.param(np.float64, 1e-8, 1e-8, marks=pytest.mark.nightly(reason="slow")), (np.float32, 1e-4, 1e-6)],
)
def test_deterministic_energies(precision, rtol, atol):
    """Verify that recomputing the energies of frames that have already had energies computed
    before, will produce the same bitwise identical energy.
    """
    seed = 1234
    dt = 1.5e-3
    temperature = DEFAULT_TEMP
    pressure = DEFAULT_PRESSURE
    barostat_interval = 25
    proposals_per_move = 1000
    targeted_water_sampling_interval = 100
    mol_a, mol_b, _ = get_hif2a_ligand_pair_single_topology()

    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    # build the protein system.
    with resources.path("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as path_to_pdb:
        complex_system, complex_coords, complex_box, _, num_water_atoms = builders.build_protein_system(
            str(path_to_pdb), ff.protein_ff, ff.water_ff
        )
    host_fns, host_masses = openmm_deserializer.deserialize_system(complex_system, cutoff=1.2)

    # resolve host clashes
    host_config = HostConfig(complex_system, complex_coords, complex_box, num_water_atoms)
    min_coords = minimizer.minimize_host_4d([mol_a, mol_b], host_config, ff)

    x0 = min_coords
    v0 = np.zeros_like(x0)

    harmonic_bond_potential = next(bp for bp in host_fns if isinstance(bp.potential, HarmonicBond))
    bond_list = get_bond_list(harmonic_bond_potential.potential)
    group_idxs = get_group_indices(bond_list, len(host_masses))
    water_idxs = [group for group in group_idxs if len(group) == 3]

    baro = MonteCarloBarostat(
        x0.shape[0],
        pressure,
        temperature,
        group_idxs,
        barostat_interval,
        seed,
    )

    nb = next(bp for bp in host_fns if isinstance(bp.potential, Nonbonded))

    # Select the protein as the target for targeted insertion
    radius = 1.0
    target_idxs = next(group for group in group_idxs if len(group) > 3)
    tibdem = custom_ops.TIBDExchangeMove_f32(
        x0.shape[0],
        target_idxs,
        water_idxs,
        nb.params,
        DEFAULT_TEMP,
        nb.potential.beta,
        nb.potential.cutoff,
        radius,
        seed,
        proposals_per_move,
        targeted_water_sampling_interval,
    )

    intg = LangevinIntegrator(temperature, dt, 1.0, np.array(host_masses), seed)

    host_params = [bp.params for bp in host_fns]
    summed_pot = SummedPotential([bp.potential for bp in host_fns], host_params)
    bps = []
    ubps = []

    ref_pot = summed_pot.to_gpu(precision).bind_params_list(host_params).bound_impl
    for bp in host_fns:
        bound_impl = bp.to_gpu(precision=precision).bound_impl
        bps.append(bound_impl)  # get the bound implementation
        ubps.append(bound_impl.get_potential())  # Get unbound potential

    baro_impl = baro.impl(bps)
    num_steps = 200
    for movers in [None, [baro_impl], [tibdem], [baro_impl, tibdem]]:
        if movers is not None:
            for mover in movers:
                # Make sure we are actually running all of the movers
                mover.set_step(0)
                assert mover.get_interval() <= num_steps
        ctxt = custom_ops.Context(x0, v0, complex_box, intg.impl(), bps, movers=movers)
        xs, boxes = ctxt.multiple_steps(num_steps, 10)

        for x, b in zip(xs, boxes):
            ref_du_dx, ref_U = ref_pot.execute(x, b)
            minimizer.check_force_norm(-ref_du_dx)
            test_u = 0.0
            test_u_selective = 0.0
            test_U_fixed = np.uint64(0)
            for fn, unbound, bp in zip(host_fns, ubps, bps):
                U_fixed = bp.execute_fixed(x, b)
                assert not fixed_overflowed(U_fixed)
                test_U_fixed += U_fixed
                _, U = bp.execute(x, b)
                test_u += U
                _, _, U_selective = unbound.execute(x, fn.params, b, False, False, True)
                test_u_selective += U_selective
            assert ref_U == fixed_to_float(test_U_fixed), precision
            assert test_u == test_u_selective, str(movers)
            np.testing.assert_allclose(ref_U, test_u, rtol=rtol, atol=atol)
            np.testing.assert_allclose(ref_U, test_u_selective, rtol=rtol, atol=atol)
