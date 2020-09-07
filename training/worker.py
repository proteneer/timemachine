import argparse

import numpy as np
import time
import os

import logging
import pickle
from concurrent import futures

import grpc

import service_pb2
import service_pb2_grpc

from threading import Lock

from timemachine.lib import custom_ops, ops

class Worker(service_pb2_grpc.WorkerServicer):

    def __init__(self):
        # self.states = {}
        # responsible for making sure no more than 1 simulation
        # runs on the gpu.
        # self.gpu_mutex = Lock()
        # mutex isn't needed since we only allow one worker.
        pass

    def Simulate(self, request, context):

        if request.precision == 'single':
            precision = np.float32
        elif request.precision == 'double':
            precision = np.float64
        else:
            raise Exception("Unknown precision")

        system = pickle.loads(request.system)

        bps = []
        pots = []

        for name, args in system.potentials:
            params = args[-1]
            op_fn = getattr(ops, name)
            potential = op_fn(*args[:-1], precision=np.float32)
            pots.append(potential) # (ytz) needed for binding, else python decides to GC this
            # du_dx, du_dp, du_dl, u = potential.execute(x0, params, box, lamb)
            bp = custom_ops.BoundPotential(potential, params)
            bps.append(bp)

        intg = custom_ops.LangevinIntegrator(*system.integrator_args)

        ctxt = custom_ops.Context(
            system.x0,
            system.v0,
            system.box,
            intg,
            bps
        )

        lamb = request.lamb

        # minimize
        for step, minimize_lamb in enumerate(np.linspace(1.0, lamb, request.prep_steps)):
            ctxt.step(minimize_lamb)

        energies = []
        frames = []

        interval = max(1, request.prod_steps//request.n_frames)

        if request.observe_du_dl_freq > 0:
            du_dl_obs = custom_ops.AvgPartialUPartialLambda(bps, request.observe_du_dl_freq)
            ctxt.add_observable(du_dl_obs)

        # if request.observe_du_dp:

        # dynamics
        for step in range(request.prod_steps):
            if step % 100 == 0:
                u = ctxt.get_u_t()
                energies.append(u)
            if step % interval == 0:
                frames.append(ctxt.get_x_t())
            ctxt.step(lamb)

        frames = np.array(frames)

        if request.observe_du_dl_freq > 0:
            avg_du_dls = du_dl_obs.avg_du_dl()
        else:
            avg_du_dls = None

        return service_pb2.SimulateReply(
            avg_du_dls=pickle.dumps(avg_du_dls),
            avg_du_dps=None,
            energies=pickle.dumps(energies),
            frames=pickle.dumps(frames),
        )

        # start = time.time()

        # # ensure only one GPU can be running at given time.
        # total_size = 0 

        # with self.gpu_mutex:

        #     ctxt.forward_mode()
        #     full_du_dls = stepper.get_du_dl() # [FxT]
        #     stripped_du_dls = []
        #     energies = stepper.get_energies()

        #     for force_du_dls in full_du_dls:
        #         # zero out 
        #         if np.all(force_du_dls) == 0:
        #             stripped_du_dls.append(None)
        #         else:
        #             stripped_du_dls.append(force_du_dls)
        #             total_size += len(force_du_dls)

        #     keep_idxs = []

        #     if request.n_frames > 0:
        #         xs = ctxt.get_all_coords()
        #         interval = max(1, xs.shape[0]//request.n_frames)
        #         for frame_idx in range(xs.shape[0]):
        #             if frame_idx % interval == 0:
        #                 keep_idxs.append(frame_idx)
        #         frames = xs[keep_idxs]
        #     else:
        #         frames = np.zeros((0, *system.x0.shape), dtype=system.x0.dtype)


        #     # store and set state for backwards mode use.
        #     if request.inference is False:
        #         self.states[request.key] = (ctxt, gradients, force_names, stepper, system)

        #     return service_pb2.ForwardReply(
        #         du_dls=pickle.dumps(stripped_du_dls), # tbd strip zeros
        #         energies=pickle.dumps(energies),
        #         frames=pickle.dumps(frames),
        #     )

    def BackwardMode(self, request, context):

        ctxt, gradients, force_names, stepper, system = self.states[request.key]

        adjoint_du_dls = pickle.loads(request.adjoint_du_dls)

        stepper.set_du_dl_adjoint(adjoint_du_dls)
        ctxt.set_x_t_adjoint(np.zeros_like(system.x0))

        with self.gpu_mutex:

            ctxt.backward_mode()

            # note that we have multiple HarmonicBonds/Angles/Torsions that correspond to different parameters
            dl_dps = []
            for f_name, g in zip(force_names, gradients):
                if f_name == 'HarmonicBond':
                    # dl_dps.append(g.get_du_dp_tangents())
                    dl_dps.append(None)
                elif f_name == 'HarmonicAngle':
                    # dl_dps.append(g.get_du_dp_tangents())
                    dl_dps.append(None)
                elif f_name == 'PeriodicTorsion':
                    # dl_dps.append(g.get_du_dp_tangents())
                    dl_dps.append(None)
                elif f_name == 'Nonbonded':
                    dl_dps.append((g.get_du_dcharge_tangents(), g.get_du_dlj_tangents()))
                elif f_name == 'LennardJones':
                    # dl_dps.append(g.get_du_dlj_tangents())
                    dl_dps.append(None)
                elif f_name == 'Electrostatics':
                    # dl_dps.append(g.get_du_dcharge_tangents())
                    dl_dps.append(None)
                elif f_name == 'GBSA':
                    dl_dps.append((g.get_du_dcharge_tangents(), g.get_du_dgb_tangents()))
                elif f_name == 'CentroidRestraint':
                    dl_dps.append(None)
                else:
                    print("f_name")
                    raise Exception("Unknown Gradient")

            del self.states[request.key]

            return service_pb2.BackwardReply(dl_dps=pickle.dumps(dl_dps))


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
