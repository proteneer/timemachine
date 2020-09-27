# test running a simple simulation
import time
import numpy as np

from training import water_box

from fe.utils import to_md_units
from ff.handlers import bonded, nonbonded, openmm_deserializer

from simtk.openmm import app

from timemachine.lib import custom_ops
from timemachine.lib import LangevinIntegrator

from fe.pdb_writer import PDBWriter

def benchmark_dhfr():



    pdb_path = 'tests/data/5dfr_solv_equil.pdb'
    host_pdb = app.PDBFile(pdb_path)
    protein_ff = app.ForceField('amber99sbildn.xml', 'tip3p.xml')
    host_system = protein_ff.createSystem(
        host_pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False
    )
    host_coords = host_pdb.positions
    box = host_pdb.topology.getPeriodicBoxVectors()
    box = np.asarray(box/box.unit)

    # box_width = 3.0
    # host_system, host_coords, box, _ = water_box.prep_system(box_width)

    host_fns, host_masses = openmm_deserializer.deserialize_system(
        host_system,
        precision=np.float32,
        cutoff=1.0
    )

    host_conf = []
    for x,y,z in host_coords:
        host_conf.append([to_md_units(x),to_md_units(y),to_md_units(z)])
    host_conf = np.array(host_conf)

    seed = 1234
    dt = 1.5e-3

    intg = LangevinIntegrator(
        300,
        dt,
        1.0,
        np.array(host_masses),
        seed
    ).impl()

    bps = []

    for potential in host_fns:
        bps.append(potential.bound_impl()) # get the bound implementation

    x0 = host_conf
    v0 = np.zeros_like(host_conf)

    ctxt = custom_ops.Context(
        x0,
        v0,
        box,
        intg,
        bps
    )

    lamb = 0.0

    start = time.time()
    num_steps = 200000

    writer = PDBWriter(open(pdb_path), "dhfr.pdb")

    writer.write_header()
    for step in range(num_steps):
        ctxt.step(lamb)
        if step % 5000 == 0:
            coords = ctxt.get_x_t()
            writer.write(coords*10)

    writer.close()

    delta = time.time()-start

    print("Delta", delta)

    steps_per_second = num_steps/delta
    seconds_per_day = 86400
    steps_per_day = steps_per_second*seconds_per_day
    ps_per_day = dt*steps_per_day
    ns_per_day = ps_per_day*1e-3

    print("ns/day", ns_per_day)
        # print("coords", ctxt.get_x_t())

    # print(host_conf)

if __name__ == "__main__":
    benchmark_dhfr()