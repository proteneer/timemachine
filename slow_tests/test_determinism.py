import numpy as np

from ff.handlers import openmm_deserializer
from ff.handlers.deserialize import deserialize_handlers
from ff import Forcefield

from timemachine.lib import custom_ops, LangevinIntegrator, MonteCarloBarostat

from fe import free_energy
from fe.topology import SingleTopology

from md import builders, minimizer
from md.barostat.utils import get_bond_list, get_group_indices
from testsystems.relative import hif2a_ligand_pair as testsystem

def FIXED_TO_FLOAT(v):
    FIXED_EXPONENT = 0x1000000000
    return np.float64(np.int64(v))/FIXED_EXPONENT

def test_deterministic_energies():
    """Verify that recomputing the energies of frames that have already had energies computed
    before, will produce the same bitwise identical energy.
    """
    seed = 1234
    dt = 1.5e-3
    temperature = 300
    pressure = 1.0
    barostat_interval = 25
    mol_a, mol_b, core = testsystem.mol_a, testsystem.mol_b, testsystem.core

    ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_sc.py').read())
    ff = Forcefield(ff_handlers)

    single_topology = SingleTopology(mol_a, mol_b, core, ff)
    rfe = free_energy.RelativeFreeEnergy(single_topology)

    ff_params = ff.get_ordered_params()

    # build the protein system.
    complex_system, complex_coords, _, _, complex_box, _ = builders.build_protein_system('tests/data/hif2a_nowater_min.pdb')
    host_fns, host_masses = openmm_deserializer.deserialize_system(
        complex_system,
        cutoff=1.0
    )


    # resolve host clashes
    min_coords = minimizer.minimize_host_4d([mol_a, mol_b], complex_system, complex_coords, ff, complex_box)

    x0 = min_coords
    v0 = np.zeros_like(x0)

    harmonic_bond_potential = host_fns[0]
    bond_list = get_bond_list(harmonic_bond_potential)
    group_idxs = get_group_indices(bond_list)
    baro = MonteCarloBarostat(
        x0.shape[0],
        pressure,
        temperature,
        group_idxs,
        barostat_interval,
        seed,
    )

    intg = LangevinIntegrator(
        temperature,
        dt,
        1.0,
        np.array(host_masses),
        seed
    ).impl()

    for precision, decimals in [(np.float32, 4), (np.float64, 8)]:
        bps = []
        unbound_bps = []

        for potential in host_fns:
            bps.append(potential.bound_impl(precision=precision)) # get the bound implementation
            unbound_bps.append(potential.unbound_impl(precision))

        for barostat in [None, baro.impl(bps)]:
            ctxt = custom_ops.Context(
                x0,
                v0,
                complex_box,
                intg,
                bps,
                barostat=barostat
            )
            for lamb in [0.0, 0.4, 1.0]:
                us, xs, boxes = ctxt.multiple_steps_U(
                    lamb,
                    200,
                    np.array([lamb]),
                    10,
                    10
                )

                for ref_U, x, b in zip(us, xs, boxes):
                    test_u = 0.0
                    test_u_selective = 0.0
                    test_U_fixed = np.uint64(0)
                    for fn, unbound, bp in zip(host_fns, unbound_bps, bps):
                        U_fixed = bp.execute_fixed(x, b, lamb)
                        test_U_fixed += U_fixed
                        _, _, U = bp.execute(x, b, lamb)
                        test_u += U
                        _, _, _, U_selective = unbound.execute_selective(x, fn.params, b, lamb, False, False, False, True)
                        test_u_selective += U_selective
                    assert ref_U == FIXED_TO_FLOAT(test_U_fixed)
                    assert test_u == test_u_selective
                    np.testing.assert_almost_equal(ref_U, test_u, decimal=decimals)
                    np.testing.assert_almost_equal(ref_U, test_u_selective, decimal=decimals)