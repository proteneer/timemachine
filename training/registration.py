import service_pb2
import service_pb2_grpc

import threading

class Registry(service_pb2_grpc.RegistrationServicer):

    def __init__(self, ip_list, expected_workers, event):

        self.ip_list = ip_list
        self.expected_workers = expected_workers
        self.shutdown_event = event
        self.lock = threading.Lock()

    def RegisterWorker(self, request, context):
        self.lock.acquire()
        self.ip_list.append(request.ip)
        if len(self.ip_list) == self.expected_workers:
            self.shutdown_event.set()
        self.lock.release()

        reply = service_pb2.EmptyMessage()
        return reply
