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
from timemachine.potentials import SummedPotential
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology

pytestmark = [pytest.mark.memcheck]


def test_deterministic_energies():
    """Verify that recomputing the energies of frames that have already had energies computed
    before, will produce the same bitwise identical energy.
    """
    seed = 1234
    dt = 1.5e-3
    temperature = DEFAULT_TEMP
    pressure = DEFAULT_PRESSURE
    barostat_interval = 25
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

    harmonic_bond_potential = host_fns[0]
    bond_list = get_bond_list(harmonic_bond_potential.potential)
    group_idxs = get_group_indices(bond_list, len(host_masses))
    baro = MonteCarloBarostat(
        x0.shape[0],
        pressure,
        temperature,
        group_idxs,
        barostat_interval,
        seed,
    )

    intg = LangevinIntegrator(temperature, dt, 1.0, np.array(host_masses), seed)

    host_params = [bp.params for bp in host_fns]
    summed_pot = SummedPotential([bp.potential for bp in host_fns], host_params)
    for precision, rtol, atol in [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 1e-6)]:
        bps = []
        ubps = []

        ref_pot = summed_pot.to_gpu(precision).bind_params_list(host_params).bound_impl
        for bp in host_fns:
            bps.append(bp.to_gpu(precision=precision).bound_impl)  # get the bound implementation
            ubps.append(bp.potential.to_gpu(precision).unbound_impl)

        for barostat in [None, baro.impl(bps)]:
            ctxt = custom_ops.Context(x0, v0, complex_box, intg.impl(), bps, barostat=barostat)
            xs, boxes = ctxt.multiple_steps(200, 10)

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
                assert test_u == test_u_selective, str(barostat)
                np.testing.assert_allclose(ref_U, test_u, rtol=rtol, atol=atol)
                np.testing.assert_allclose(ref_U, test_u_selective, rtol=rtol, atol=atol)
