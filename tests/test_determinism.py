from importlib import resources

import numpy as np

from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.integrator import FIXED_TO_FLOAT
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat, custom_ops
from timemachine.md import builders, minimizer
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology


def test_deterministic_energies():
    """Verify that recomputing the energies of frames that have already had energies computed
    before, will produce the same bitwise identical energy.
    """
    seed = 1234
    dt = 1.5e-3
    temperature = 300
    pressure = 1.0
    barostat_interval = 25
    mol_a, mol_b, _ = get_hif2a_ligand_pair_single_topology()

    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    # build the protein system.
    with resources.path("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as path_to_pdb:
        complex_system, complex_coords, complex_box, _ = builders.build_protein_system(
            str(path_to_pdb), ff.protein_ff, ff.water_ff
        )
    host_fns, host_masses = openmm_deserializer.deserialize_system(complex_system, cutoff=1.0)

    # resolve host clashes
    min_coords = minimizer.minimize_host_4d([mol_a, mol_b], complex_system, complex_coords, ff, complex_box)

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

    intg = LangevinIntegrator(temperature, dt, 1.0, np.array(host_masses), seed).impl()

    for precision, decimals in [(np.float32, 4), (np.float64, 8)]:
        bps = []
        ubps = []

        for bp in host_fns:
            bps.append(bp.to_gpu(precision=precision).bound_impl)  # get the bound implementation
            ubps.append(bp.potential.to_gpu(precision).unbound_impl)

        for barostat in [None, baro.impl(bps)]:
            ctxt = custom_ops.Context(x0, v0, complex_box, intg, bps, barostat=barostat)
            us, xs, boxes = ctxt.multiple_steps_U(200, 10, 10)

            for ref_U, x, b in zip(us, xs, boxes):
                test_u = 0.0
                test_u_selective = 0.0
                test_U_fixed = np.uint64(0)
                for fn, unbound, bp in zip(host_fns, ubps, bps):
                    U_fixed = bp.execute_fixed(x, b)
                    test_U_fixed += U_fixed
                    _, U = bp.execute(x, b)
                    test_u += U
                    _, _, U_selective = unbound.execute_selective(x, fn.params, b, False, False, True)
                    test_u_selective += U_selective
                assert ref_U == FIXED_TO_FLOAT(test_U_fixed)
                assert test_u == test_u_selective
                np.testing.assert_almost_equal(ref_U, test_u, decimal=decimals)
                np.testing.assert_almost_equal(ref_U, test_u_selective, decimal=decimals)
