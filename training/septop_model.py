import os
import pickle

from io import StringIO

from training import service_pb2
from training import bootstrap
from rdkit import Chem

import numpy as np



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


# we want this to work for protein ligand systems as well as solvation free energies
def simulate(
    simulation,
    num_host_atoms,
    n_steps,
    lambda_schedule,
    stubs,
    combined_pdb):
    """
    Compute the hydration free energy of a simulation system.

    Parameters
    ----------

    simulation: Simulation
        Simulation system

    n_steps: int
        Number of steps that we run the simulation for.

    lambda_schedule: array, np.float64
        The lambda windows we're interested in simulating over.

    stubs: grpc.service
        gRPC services that will be used to run each lambda window

    Returns
    -------
    (dG, dG_err), dG_grad, du_dls
        dG grad is with respect to system parameters Q, not forcefield parameters P.
        It correspond to the vjps of each potential function that backprops into
        the forcefield handler directly.

    """

    n_frames = 10

    simulate_futures = []

    for lamb_idx, lamb in enumerate(lambda_schedule):

        # endpoint lambda
        if lamb_idx == 0 or lamb_idx == len(lambda_schedule) - 1:
            observe_du_dl_freq = 5000 # this is analytically zero.
            observe_du_dp_freq = 25
        else:
            observe_du_dl_freq = 25 # this is analytically zero.
            observe_du_dp_freq = 0

        request = service_pb2.SimulateRequest(
            simulation=pickle.dumps(simulation),
            lamb=lamb,
            prep_steps=5000,
            prod_steps=n_steps,
            observe_du_dl_freq=observe_du_dl_freq,
            observe_du_dp_freq=observe_du_dp_freq,
            precision="single",
            n_frames=n_frames,
            num_host_atoms=num_host_atoms
        )

        stub = stubs[lamb_idx % len(stubs)]

        # launch asynchronously
        response_future = stub.Simulate.future(request)
        simulate_futures.append(response_future)

    du_dls = []

    for lamb_idx, (lamb, future) in enumerate(zip(lambda_schedule, simulate_futures)):
        response = future.result()
        # print("finishing up lambda", lamb)
        energies = pickle.loads(response.energies)

        # enable this later when we need simulation frames
        if n_frames > 0:
            frames = pickle.loads(response.frames)
            combined_pdb_str = StringIO(Chem.MolToPDBBlock(combined_pdb))

            out_file = "deubg_simulation_"+str(lamb_idx)+".pdb"
            pdb_writer = PDBWriter(combined_pdb_str, out_file)
            pdb_writer.write_header(simulation.box)
            for x in frames:
                x = recenter(x, simulation.box)
                pdb_writer.write(x*10)
            pdb_writer.close()

        du_dl = pickle.loads(response.avg_du_dls)
        du_dls.append(du_dl)

        if lamb_idx == 0:
            lambda_0_du_dqs = pickle.loads(response.avg_du_dps)
        elif lamb_idx == len(lambda_schedule) - 1:
            lambda_1_du_dqs = pickle.loads(response.avg_du_dps)

    pred_dG = np.trapz(du_dls, lambda_schedule)
    pred_dG_err = bootstrap.ti_ci(du_dls, lambda_schedule)

    
    grad_dG = []

    for source_grad, target_grad in zip(lambda_0_du_dqs, lambda_1_du_dqs):
        grad_dG.append(target_grad - source_grad)


    return (pred_dG, pred_dG_err), grad_dG, du_dls
