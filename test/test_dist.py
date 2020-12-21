from multiprocessing import Process
import numpy as np
import time
import torch
import torch.distributed.rpc as rpc
import unittest

from tinygrad.tensor import Tensor, GPU

def run_remote_torch(rank, world_size):
    name = "worker{}".format(rank)
    rpc.init_rpc(
        name=name,
        rank=rank,
        world_size=world_size
    )
    rpc.shutdown()

def compute_torch(t1, t2):
  return torch.add(t1, t2)

class TestTinygrad(unittest.TestCase):
  gpu = False

  def test_distributed_grad(self):
    def test_tinygrad():
        """
        TODO
        dial workers
        assume the other agent is running exactly the same model,
        so can ref func by ID instead of trying to serialize it
        compare (timewise) tinygrad distributed over CPU and GPU vs pure pytorch on CPU
        """
        return np.random.randn(3,3)

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

      t1 = torch.rand((3, 3), requires_grad=True)
      t2 = torch.rand((3, 3), requires_grad=True)
      t3 = rpc.rpc_sync("worker1", compute_torch, args=(t1, t2))
      t4 = torch.rand((3, 3), requires_grad=True)
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