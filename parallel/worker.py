import argparse

import numpy as np
from datetime import datetime
import os

import logging
import pickle
from concurrent import futures

from parallel import service_pb2, service_pb2_grpc
from parallel.utils import get_worker_status
from parallel.constants import DEFAULT_GRPC_OPTIONS

import grpc

class Worker(service_pb2_grpc.WorkerServicer):

    def Submit(self, request, context):
        start = datetime.now()
        task_fn, args, kwargs = pickle.loads(request.binary)
        task_name = task_fn.__name__
        print(f"Started {task_name} at {start}", flush=True)
        result = task_fn(*args, **kwargs)
        finish_time = datetime.now()
        total_seconds = (finish_time - start).seconds
        print(f"Running {task_name} took {total_seconds}", flush=True)
        return service_pb2.PickleData(binary=pickle.dumps(result))

    def Status(self, request, context):
        return get_worker_status()


def serve(args):

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options=DEFAULT_GRPC_OPTIONS)
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
