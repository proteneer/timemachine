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


from timemachine.lib import custom_ops, ops

class Worker(service_pb2_grpc.WorkerServicer):

    def __init__(self):
        self.state = None
        # md_ctxt = None
        # self.gradients = None
        # self.force_names = None
        # self.

    def ForwardMode(self, request, context):
        assert self.state is None

        if request.precision == 'single':
            precision = np.float32
        elif request.precision == 'double':
            precision = np.float64
        else:
            raise Exception("Unknown precision")

        system = pickle.loads(request.system)

        gradients = []
        force_names = []

        for grad_name, grad_args in system.gradients:
            force_names.append(grad_name)
            op_fn = getattr(ops, grad_name)
            grad = op_fn(*grad_args, precision=precision)
            gradients.append(grad)

        integrator = system.integrator

        stepper = custom_ops.AlchemicalStepper_f64(
            gradients,
            integrator.lambs
        )

        ctxt = custom_ops.ReversibleContext_f64(
            stepper,
            system.x0,
            system.v0,
            integrator.cas,
            integrator.cbs,
            integrator.ccs,
            integrator.dts,
            integrator.seed
        )

        start = time.time()
        ctxt.forward_mode()
        print("fwd run time", time.time() - start)

        full_du_dls = stepper.get_du_dl() # [FxT]
        energies = stepper.get_energies()
        reply = service_pb2.ForwardReply(du_dls=pickle.dumps(full_du_dls), energies=pickle.dumps(energies))

        # store and set state for backwards mode use.
        if request.inference is False:
            self.state = (ctxt, gradients, force_names, stepper, system)


        return reply

    def BackwardMode(self, request, context):
        assert self.state is not None

        ctxt, gradients, force_names, stepper, system = self.state
        adjoint_du_dls = pickle.loads(request.adjoint_du_dls)

        stepper.set_du_dl_adjoint(adjoint_du_dls)
        ctxt.set_x_t_adjoint(np.zeros_like(system.x0))
        start = time.time()
        print("start backwards mode")
        ctxt.backward_mode()
        print("bkwd run time", time.time() - start)
        # not a valid method, grab directly from handlers

        # note that we have multiple HarmonicBonds/Angles/Torsions that correspond to different parameters
        dl_dps = []
        for f_name, g in zip(force_names, gradients):
            if f_name == 'HarmonicBond':
                dl_dps.append(g.get_du_dp_tangents())
            elif f_name == 'HarmonicAngle':
                dl_dps.append(g.get_du_dp_tangents())
            elif f_name == 'PeriodicTorsion':
                dl_dps.append(g.get_du_dp_tangents())
            elif f_name == 'Nonbonded':
                dl_dps.append((g.get_du_dcharge_tangents(), g.get_du_dlj_tangents()))
            elif f_name == 'GBSA':
                dl_dps.append((g.get_du_dcharge_tangents(), g.get_du_dgb_tangents()))
            elif f_name == 'Restraint':
                dl_dps.append(g.get_du_dp_tangents())
            else:
                raise Exception("Unknown Gradient")

        reply = service_pb2.BackwardReply(dl_dps=pickle.dumps(dl_dps))

        self.state = None

        return reply


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


# class Worker():

#   def __init__(self):
#       self.ctxt = None

#   def forward_mode(self, request):

#       assert self.ctxt is None

#       system_xml, precision = request.data()
#       reply = Reply()
#       self.ctxt = initialize()
#       du_dls = self.ctxt.simulate()
#       reply.du_dls = du_dls
    
#       if request.inference:
#           self.ctxt = None

#       self.send(reply)

#   def backwards_mode():

#       assert self.ctxt is not None
#       self.ctxt.reverse_mode()

#       parameter_adjoints = None
#       self.send(parameter_adjoints)

#       # ensure this can only ever be ran once
#       self.ctxt = None
