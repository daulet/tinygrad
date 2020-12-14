from multiprocessing import Process
import numpy as np
import time
import unittest

from rpc import Worker, RPC

def sum(a, b):
    return a + b

def run_remote(port):
    worker = Worker(port)
    worker.run()

class TestDistributed(unittest.TestCase):

    def test_simple(self):
        port = 65432
        proc = Process(target=run_remote, args=[port])
        proc.start()
        time.sleep(1)

        a = np.ones([3,3]) * 3
        b = np.ones_like(a) * 6

        res = RPC.send(sum, a, b)
        RPC.done()
        proc.join()

        res == np.ones_like(a) * 9

if __name__ == "__main__":
    unittest.main()