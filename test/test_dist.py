from multiprocessing import Process
import numpy as np
import time
import torch
import torch.distributed.rpc as rpc
import unittest

from distributed.rpc import Worker, RPC
from tinygrad.tensor import Tensor, GPU


x_init = np.random.randn(3,3).astype(np.float32)
y_init = np.random.randn(3,3).astype(np.float32)
z_init = np.random.randn(3,3).astype(np.float32)


def run_remote_tinygrad(port):
    worker = Worker(port)
    worker.run()


def run_remote_torch(rank, world_size):
    name = "worker{}".format(rank)
    rpc.init_rpc(
        name=name,
        rank=rank,
        world_size=world_size
    )
    rpc.shutdown()


def compute_tinygrad(t1, t2):
  return t1 + t2


def compute_torch(t1, t2):
  return torch.add(t1, t2)


class TestTinygrad(unittest.TestCase):
  gpu = False

  def test_distributed_grad(self):
    def test_tinygrad():
        """
        TODO
        assume the other agent is running exactly the same model,
        so can ref func by ID instead of trying to serialize it
        compare (timewise) tinygrad distributed over CPU and GPU vs pure pytorch on CPU
        """

        # Arrange

        port = 65432
        proc = Process(target=run_remote_tinygrad, args=[port])
        proc.start()
        time.sleep(0.1)

        # Act

        t1 = Tensor(x_init, gpu=self.gpu)
        t2 = Tensor(y_init, gpu=self.gpu)
        t3 = RPC.send(compute_tinygrad, t1, t2)
        t4 = Tensor(z_init, gpu=self.gpu)
        res = t3.mul(t4)

        # clean up

        RPC.done()
        proc.join()

        return res.cpu().data


    def test_pytorch():

      # Arrange

      world_size = 2
      proc = Process(target=run_remote_torch, args=[1, world_size])
      proc.start()

      rpc.init_rpc(
        name="worker0",
        rank=0,
        world_size=world_size
      )

      # Act

      t1 = torch.tensor(x_init, requires_grad=True)
      t2 = torch.tensor(y_init, requires_grad=True)
      t3 = rpc.rpc_sync("worker1", compute_torch, args=(t1, t2))
      t4 = torch.tensor(z_init, requires_grad=True)
      res = torch.mul(t3, t4)

      # clean up

      rpc.shutdown()
      proc.join()

      return res.detach().numpy()


    # Assert

    for x,y in zip(test_tinygrad(), test_pytorch()):
      np.testing.assert_allclose(x, y, atol=1e-5)


@unittest.skipUnless(GPU, "Requires GPU")
class TestTinygradGPU(TestTinygrad):
  gpu = True


if __name__ == '__main__':
  unittest.main()