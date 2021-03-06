import argparse

import numpy as np
import time
import os

import logging
import pickle
from concurrent import futures

from parallel import service_pb2
from parallel import service_pb2_grpc

import grpc

class Worker(service_pb2_grpc.WorkerServicer):

    def Submit(self, request, context):
        task_fn, args = pickle.loads(request.binary)
        result = task_fn(args)
        return service_pb2.PickleData(binary=pickle.dumps(result))

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
