"""Run vanilla "apo" MD on DHFR and HIF2A test systems,
and running an intermediate lambda window "rbfe" MD for a
relative binding free energy edge from the HIF2A test system"""

import time
import numpy as np

from ff.handlers import openmm_deserializer
from ff.handlers.deserialize import deserialize_handlers
from ff import Forcefield

from simtk.openmm import app

from timemachine.lib import custom_ops, LangevinIntegrator, MonteCarloBarostat

from fe.utils import to_md_units
from fe.model_utils import apply_hmr
from fe import free_energy
from fe.topology import SingleTopology

from md import builders, minimizer
from md.barostat.utils import get_bond_list, get_group_indices


def recenter(conf, box):

    new_coords = []

    periodicBoxSize = box

    for atom in conf:
        diff = np.array([0.0, 0.0, 0.0])
        diff += periodicBoxSize[2] * np.floor(atom[2] / periodicBoxSize[2][2])
        diff += periodicBoxSize[1] * np.floor((atom[1] - diff[1]) / periodicBoxSize[1][1])
        diff += periodicBoxSize[0] * np.floor((atom[0] - diff[0]) / periodicBoxSize[0][0])
        new_coords.append(atom - diff)

    return np.array(new_coords)


def benchmark(
    label,
    masses,
    lamb,
    x0,
    v0,
    box,
    bound_potentials,
    hmr=False,
    verbose=True,
    num_batches=100,
    steps_per_batch=1000,
    compute_du_dl_interval=0,
    barostat_interval=0,
):
    """
    TODO: configuration blob containing num_batches, steps_per_batch, and any other options
    """

    seed = 1234
    dt = 1.5e-3
    temperature = 300
    pressure = 1.0
    seconds_per_day = 86400

    harmonic_bond_potential = bound_potentials[0]
    bond_list = get_bond_list(harmonic_bond_potential)
    if hmr:
        dt = 2.5e-3
        masses = apply_hmr(masses, bond_list)
    intg = LangevinIntegrator(temperature, dt, 1.0, np.array(masses), seed).impl()

    bps = []

    for potential in bound_potentials:
        bps.append(potential.bound_impl(precision=np.float32))  # get the bound implementation

    baro_impl = None
    if barostat_interval > 0:
        group_idxs = get_group_indices(bond_list)
        baro = MonteCarloBarostat(
            x0.shape[0],
            pressure,
            temperature,
            group_idxs,
            barostat_interval,
            seed,
        )
        baro_impl = baro.impl(bps)

    ctxt = custom_ops.Context(
        x0,
        v0,
        box,
        intg,
        bps,
        barostat=baro_impl,
    )

    batch_times = []

    lambda_schedule = np.ones(steps_per_batch) * lamb

    # run once before timer starts
    ctxt.multiple_steps(lambda_schedule, compute_du_dl_interval)

    start = time.time()

    for batch in range(num_batches):

        # time the current batch
        batch_start = time.time()
        du_dls, _, _ = ctxt.multiple_steps(lambda_schedule, compute_du_dl_interval)
        batch_end = time.time()

        delta = batch_end - batch_start

        batch_times.append(delta)

        steps_per_second = steps_per_batch / np.mean(batch_times)
        steps_per_day = steps_per_second * seconds_per_day

        ps_per_day = dt * steps_per_day
        ns_per_day = ps_per_day * 1e-3

        if verbose:
            print(f"steps per second: {steps_per_second:.3f}")
            print(f"ns per day: {ns_per_day:.3f}")

    assert np.all(np.abs(ctxt.get_x_t()) < 1000)

    print(
        f"{label}: N={x0.shape[0]} speed: {ns_per_day:.2f}ns/day dt: {dt*1e3}fs (ran {steps_per_batch * num_batches} steps in {(time.time() - start):.2f}s)"
    )


def benchmark_dhfr(verbose=False, num_batches=100, steps_per_batch=1000):

    pdb_path = "tests/data/5dfr_solv_equil.pdb"
    host_pdb = app.PDBFile(pdb_path)
    protein_ff = app.ForceField("amber99sbildn.xml", "tip3p.xml")
    host_system = protein_ff.createSystem(
        host_pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False
    )
    host_coords = host_pdb.positions
    box = host_pdb.topology.getPeriodicBoxVectors()
    box = np.asarray(box / box.unit)

    host_fns, host_masses = openmm_deserializer.deserialize_system(host_system, cutoff=1.0)

    host_conf = []
    for x, y, z in host_coords:
        host_conf.append([to_md_units(x), to_md_units(y), to_md_units(z)])
    host_conf = np.array(host_conf)

    x0 = host_conf
    v0 = np.zeros_like(host_conf)

    benchmark(
        "dhfr-apo",
        host_masses,
        0.0,
        x0,
        v0,
        box,
        host_fns,
        verbose=verbose,
        num_batches=num_batches,
        steps_per_batch=steps_per_batch,
    )
    benchmark(
        "dhfr-apo-barostat-interval-25",
        host_masses,
        0.0,
        x0,
        v0,
        box,
        host_fns,
        verbose=verbose,
        num_batches=num_batches,
        steps_per_batch=steps_per_batch,
        barostat_interval=25,
    )
    benchmark(
        "dhfr-apo-hmr-barostat-interval-25",
        host_masses,
        0.0,
        x0,
        v0,
        box,
        host_fns,
        verbose=verbose,
        hmr=True,
        num_batches=num_batches,
        steps_per_batch=steps_per_batch,
        barostat_interval=25,
    )


def benchmark_hif2a(verbose=False, num_batches=100, steps_per_batch=1000):

    from testsystems.relative import hif2a_ligand_pair as testsystem

    mol_a, mol_b, core = testsystem.mol_a, testsystem.mol_b, testsystem.core

    # this
    ff_handlers = deserialize_handlers(open("ff/params/smirnoff_1_1_0_sc.py").read())
    ff = Forcefield(ff_handlers)

    single_topology = SingleTopology(mol_a, mol_b, core, ff)
    rfe = free_energy.RelativeFreeEnergy(single_topology)

    ff_params = ff.get_ordered_params()

    # build the protein system.
    complex_system, complex_coords, _, _, complex_box, _ = builders.build_protein_system(
        "tests/data/hif2a_nowater_min.pdb"
    )

    solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(4.0)

    for stage, host_system, host_coords, host_box in [
        ("hif2a", complex_system, complex_coords, complex_box),
        ("solvent", solvent_system, solvent_coords, solvent_box),
    ]:

        host_fns, host_masses = openmm_deserializer.deserialize_system(host_system, cutoff=1.0)

        # resolve host clashes
        min_host_coords = minimizer.minimize_host_4d([mol_a, mol_b], host_system, host_coords, ff, host_box)

        x0 = min_host_coords
        v0 = np.zeros_like(x0)

        # lamb = 0.0
        benchmark(
            stage + "-apo",
            host_masses,
            0.0,
            x0,
            v0,
            host_box,
            host_fns,
            verbose=verbose,
            num_batches=num_batches,
            steps_per_batch=steps_per_batch,
        )
        benchmark(
            stage + "-apo-barostat-interval-25",
            host_masses,
            0.0,
            x0,
            v0,
            host_box,
            host_fns,
            verbose=verbose,
            num_batches=num_batches,
            steps_per_batch=steps_per_batch,
            barostat_interval=25,
        )

        # RBFE
        unbound_potentials, sys_params, masses, coords = rfe.prepare_host_edge(ff_params, host_system, x0)

        bound_potentials = [x.bind(y) for (x, y) in zip(unbound_potentials, sys_params)]

        x0 = coords
        v0 = np.zeros_like(x0)

        # lamb = 0.5
        benchmark(
            stage + "-rbfe-with-du-dp",
            masses,
            0.5,
            x0,
            v0,
            host_box,
            bound_potentials,
            verbose=verbose,
            num_batches=num_batches,
            steps_per_batch=steps_per_batch,
        )

        for du_dl_interval in [0, 1, 5]:
            benchmark(
                stage + "-rbfe-du-dl-interval-" + str(du_dl_interval),
                masses,
                0.5,
                x0,
                v0,
                host_box,
                bound_potentials,
                verbose=verbose,
                num_batches=num_batches,
                steps_per_batch=steps_per_batch,
                compute_du_dl_interval=du_dl_interval,
            )


def test_dhfr():
    benchmark_dhfr(verbose=True, num_batches=2, steps_per_batch=100)


def test_hif2a():
    benchmark_hif2a(verbose=True, num_batches=2, steps_per_batch=100)


if __name__ == "__main__":

    benchmark_dhfr(verbose=False, num_batches=100)
    benchmark_hif2a(verbose=False, num_batches=100)
