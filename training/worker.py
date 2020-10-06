import argparse

import sys
import numpy as np
import time
import os

import logging
import pickle
from concurrent import futures
import copy
import grpc

import service_pb2
import service_pb2_grpc

from threading import Lock

import timemachine.lib

from timemachine.lib import custom_ops
from timemachine.lib import potentials

class Worker(service_pb2_grpc.WorkerServicer):

    def Simulate(self, request, context):

        if request.precision == 'single':
            precision = np.float32
        elif request.precision == 'double':
            precision = np.float64
        else:
            raise Exception("Unknown precision")

        simulation = pickle.loads(request.simulation)

        bps = []
        pots = []

        min_bps = []

        for potential in simulation.potentials:

            if isinstance(potential, timemachine.lib.potentials.Nonbonded):
                min_potential = copy.deepcopy(potential)
                loi = min_potential.get_lambda_offset_idxs()
                loi[:request.num_host_atoms] = 0
                loi[request.num_host_atoms:] = 1
                min_bps.append(min_potential.bound_impl())
            else:
                min_bps.append(potential.bound_impl())

            bps.append(potential.bound_impl()) # get the bound implementation

        intg = simulation.integrator.impl()
        lamb = request.lamb

        min_ctxt = custom_ops.Context(
            simulation.x,
            simulation.v,
            simulation.box,
            intg,
            min_bps
        )

        # minimization may use a different set of lambda indicies
        # for step, minimize_lamb in enumerate(np.linspace(1.0, lamb, request.prep_steps)):
        # should we minimiize to zero or to lambda? for sep top
        for step, minimize_lamb in enumerate(np.linspace(1.0, 0, request.prep_steps)):
            min_ctxt.step(minimize_lamb)

        energies = []
        frames = []

        intg = simulation.integrator.impl()

        ctxt = custom_ops.Context(
            min_ctxt.get_x_t(),
            simulation.v, # maybe use min_ctxt.get_v_t()?
            simulation.box,
            intg,
            bps
        )

        if request.observe_du_dl_freq > 0:
            du_dl_obs = custom_ops.AvgPartialUPartialLambda(bps, request.observe_du_dl_freq)
            ctxt.add_observable(du_dl_obs)

        if request.observe_du_dp_freq > 0:
            du_dps = []
            # for name, bp in zip(names, bps):
            # if name == 'LennardJones' or name == 'Electrostatics':
            for bp in bps:
                du_dp_obs = custom_ops.AvgPartialUPartialParam(bp, request.observe_du_dp_freq)
                ctxt.add_observable(du_dp_obs)
                du_dps.append(du_dp_obs)

        # dynamics
        for step in range(request.prod_steps):
            if step % 100 == 0:
                u = ctxt.get_u_t()
                energies.append(u)

            if request.n_frames > 0:
                interval = max(1, request.prod_steps//request.n_frames)
                if step % interval == 0:
                    frames.append(ctxt.get_x_t())


            ctxt.step(lamb)

        frames = np.array(frames)

        if request.observe_du_dl_freq > 0:
            avg_du_dls = du_dl_obs.avg_du_dl()
        else:
            avg_du_dls = None

        if request.observe_du_dp_freq > 0:
            avg_du_dps = []
            for obs in du_dps:
                avg_du_dps.append(obs.avg_du_dp())
        else:
            avg_du_dps = None

        return service_pb2.SimulateReply(
            avg_du_dls=pickle.dumps(avg_du_dls),
            avg_du_dps=pickle.dumps(avg_du_dps),
            energies=pickle.dumps(energies),
            frames=pickle.dumps(frames),
        )


def serve(args):

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1),
        options = [
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024)
        ]
    )
    service_pb2_grpc.add_WorkerServicer_to_server(Worker(), server)
    server.add_insecure_port('[::]:'+str(args.port))
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Worker Server')
    parser.add_argument('--gpu_idx', type=int, required=True, help='Location of all output files')
    parser.add_argument('--port', type=int, required=True, help='Either single or double precision. Double is 8x slower.')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_idx)

    logging.basicConfig()
    serve(args)
