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
        self.states = {}
        self.mutex = Lock()

    def ResetState(self, request, context):
        with self.mutex:
            self.states.clear()

        reply = service_pb2.EmptyMessage()
        return reply

    def ForwardMode(self, request, context):

        if request.precision == 'single':
            precision = np.float32
        elif request.precision == 'double':
            precision = np.float64
        else:
            raise Exception("Unknown precision")

        # systems = pickle.loads(request.systems)

        gradients = pickle.loads(request.gradients)
        integrators = pickle.loads(request.integrators)

        assert len(gradients) == len(integrators)

        with self.mutex:

            x0 = pickle.loads(request.x0)
            v0 = pickle.loads(request.v0)

            all_du_dls = []
            all_energies = []
            all_frames = []

            all_ctxts = []

            assert request.n_frames == 0

            for gradient, integrator in zip(gradients, integrators):

                grads = []
                force_names = []

                # instantiate gradients
                for grad_name, grad_args in gradient:
                    force_names.append(grad_name)
                    op_fn = getattr(ops, grad_name)
                    grad = op_fn(*grad_args, precision=precision)
                    grads.append(grad)
 
                stepper = custom_ops.AlchemicalStepper_f64(
                    grads,
                    integrator.lambs
                )

                ctxt = custom_ops.ReversibleContext_f64(
                    stepper,
                    x0,
                    v0,
                    integrator.cas,
                    integrator.cbs,
                    integrator.ccs,
                    integrator.dts,
                    integrator.seed
                )

                start = time.time()

                # ensure only one GPU can be running at given time.
                ctxt.forward_mode()

                all_du_dls.append(stepper.get_du_dl())
                all_energies.append(stepper.get_energies())

                keep_idxs = []

                if request.n_frames > 0:
                    xs = ctxt.get_all_coords()
                    interval = max(1, xs.shape[0]//request.n_frames)
                    for frame_idx in range(xs.shape[0]):
                        if frame_idx % interval == 0:
                            keep_idxs.append(frame_idx)
                    frames = xs[keep_idxs]
                else:
                    frames = np.zeros((0, *x0.shape), dtype=x0.dtype)

                x0 = ctxt.get_last_coords()
                v0 = ctxt.get_last_velocities()

                all_ctxts.append(ctxt)

            # store and set state for backwards mode use.
            if request.inference is False:
                self.states[request.key] = (all_ctxts, gradients, force_names, stepper, integrators)

            all_du_dls = np.concatenate(all_du_dls, axis=-1)
            energies = np.concatenate(all_energies)

            # stripped_du_dls = []
            # for force_du_dls in all_du_dls:
                # zero out 
                # if np.all(force_du_dls) == 0:
                    # stripped_du_dls.append(None)
                # else:
                    # stripped_du_dls.append(force_du_dls)

            return service_pb2.ForwardReply(
                du_dls=pickle.dumps(all_du_dls), # tbd strip zeros
                energies=pickle.dumps(energies),
                frames=pickle.dumps(frames)
            )


    def BackwardMode(self, request, context):

        ctxt, gradients, force_names, stepper, system = self.states[request.key]

        adjoint_du_dls = pickle.loads(request.adjoint_du_dls)

        stepper.set_du_dl_adjoint(adjoint_du_dls)
        # ctxt.set_x_t_adjoint(request.adjoint_x_t)
        # ctxt.set_v_t_adjoint(request.adjoint_v_t)

        with self.mutex:

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

            return service_pb2.BackwardReply(
                dl_dps=pickle.dumps(dl_dps),
                adjoint_x_t=adjoint_x_t,
                adjoint_v_t=adjoint_v_t
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
