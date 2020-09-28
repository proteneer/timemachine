import os
import pickle

from training import service_pb2
from training import bootstrap


import numpy as np

# we want this to work for protein ligand systems as well as solvation free energies
def simulate(
    simulation,
    n_steps,
    lambda_schedule,
    stubs):
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
    (dG, dG_err), dG_grad
        dG grad is with respect to system parameters Q, not forcefield parameters P.
        It correspond to the vjps of each potential function that backprops into
        the forcefield handler directly.

    """

    n_frames = 0

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
            # prod_steps=100000,
            observe_du_dl_freq=observe_du_dl_freq,
            observe_du_dp_freq=observe_du_dp_freq,
            precision="single",
            n_frames=n_frames,
        )

        stub = stubs[lamb_idx % len(stubs)]

        # launch asynchronously
        response_future = stub.Simulate.future(request)
        simulate_futures.append(response_future)

    du_dls = []

    for lamb_idx, (lamb, future) in enumerate(zip(lambda_schedule, simulate_futures)):
        response = future.result()
        energies = pickle.loads(response.energies)

        # if n_frames > 0:
        #     frames = pickle.loads(response.frames)
        #     # combined_pdb_str = StringIO(Chem.MolToPDBBlock(combined_pdb))

        #     out_file = os.path.join(epoch_dir, "simulation_"+str(lamb_idx)+".pdb")
        #     pdb_writer = PDBWriter(combined_pdb_str, out_file)
        #     pdb_writer.write_header(box)
        #     for x in frames:
        #         x = recenter(x, box)
        #         pdb_writer.write(x*10)
        #     pdb_writer.close()

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
