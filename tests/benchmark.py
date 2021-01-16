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

def recenter(conf, box):

    new_coords = []

    periodicBoxSize = box

    for atom in conf:
        diff = np.array([0., 0., 0.])
        diff += periodicBoxSize[2]*np.floor(atom[2]/periodicBoxSize[2][2]);
        diff += periodicBoxSize[1]*np.floor((atom[1]-diff[1])/periodicBoxSize[1][1]);
        diff += periodicBoxSize[0]*np.floor((atom[0]-diff[0])/periodicBoxSize[0][0]);
        new_coords.append(atom - diff)

    return np.array(new_coords)


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

    host_fns, host_masses = openmm_deserializer.deserialize_system(
        host_system,
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
        bps.append(potential.bound_impl(precision=np.float32)) # get the bound implementation

    x0 = host_conf
    v0 = np.zeros_like(host_conf)

    ctxt = custom_ops.Context(
        x0,
        v0,
        box,
        intg,
        bps
    )

    # initialize observables
    obs = []
    for bp in bps:
        du_dp_obs = custom_ops.AvgPartialUPartialParam(bp, 100)
        ctxt.add_observable(du_dp_obs)
        obs.append(du_dp_obs)

    lamb = 0.0

    start = time.time()

    writer = PDBWriter([host_pdb.topology], "dhfr.pdb")

    num_batches = 100
    steps_per_batch = 1000

    for batch in range(num_batches):
        lambda_schedule = np.ones(steps_per_batch)*lamb
        ctxt.multiple_steps(lambda_schedule)

        delta = time.time()-start
        steps_per_second = (batch+1)*steps_per_batch/delta
        seconds_per_day = 86400
        steps_per_day = steps_per_second*seconds_per_day
        ps_per_day = dt*steps_per_day
        ns_per_day = ps_per_day*1e-3

        print((batch+1)*steps_per_batch, "steps @ ", ns_per_day, " ns/day")
        # coords = recenter(ctxt.get_x_t(), box)
        # writer.write_frame(coords*10)

    print("total time", time.time() - start)

    writer.close()


    # bond angle torsions nonbonded
    for potential, du_dp_obs in zip(host_fns, obs):
        dp = du_dp_obs.avg_du_dp()
        print(potential, dp.shape)
        print(dp)

if __name__ == "__main__":
    benchmark_dhfr()