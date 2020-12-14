#!/usr/bin/env python3

import pickle
import socket

class Op:
    def __init__(self, f, *args):
        self.function = f
        self.arguments = args
        self.exit = False

    @classmethod
    def close(cls):
        op = Op(None)
        op.exit = True
        return op


class RPC:
    HOST = '127.0.0.1'  # The server's hostname or IP address
    PORT = 65432        # The port used by the server
    PICKLE_PROTOCOL = 5 # Experimentally seems to result in smaller payload

    socket = None

    @staticmethod
    def get_socket():
        if RPC.socket is None:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((RPC.HOST, RPC.PORT))
            RPC.socket = s

        return RPC.socket

    # TODO add async flavor - need to figure out how to track that request
    @staticmethod
    def send(func, *args):
        s = RPC.get_socket()
        req = pickle.dumps(Op(func, *args), protocol=RPC.PICKLE_PROTOCOL)
        print(f"sending {len(req)} bytes")
        s.sendall(req)
        res = s.recv(1024) # TODO what if this is not big enough?
        return pickle.loads(res)

    @staticmethod
    def done():
        close = Op.close()
        req = pickle.dumps(close, protocol=RPC.PICKLE_PROTOCOL)
        
        s = RPC.get_socket()
        s.sendall(req)
        s.close()
        

class Worker(object):

    HOST = '127.0.0.1'

    def __init__(self, port):
        self.port = port

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.HOST, self.port))
            s.listen()
            while True:
                conn, addr = s.accept()
                with conn:
                    #TODO clean up print statements
                    print('Connected by', addr)
                    while True:
                        data = conn.recv(4096) # TODO better default?
                        print(f"received bytes: {len(data):}: {data}")
                        if not data:
                            break

                        op = pickle.loads(data)
                        if op.exit:
                            print("exiting")
                            return

                        res = op.function(*op.arguments)
                        data = pickle.dumps(res, protocol=5)
                        conn.sendall(data)
