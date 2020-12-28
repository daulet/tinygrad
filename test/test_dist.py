from multiprocessing import Process
import numpy as np
import os
import time
import torch
import torch.distributed.rpc as rpc
import unittest

from distributed.rpc import Worker, RPC
from tinygrad.tensor import Device, Tensor, GPU


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
  device = Device.CPU
  port = 65432

  def test_distributed_forward(self):

    def test_tinygrad():
        # Arrange
        proc = Process(target=run_remote_tinygrad, args=[self.port])
        proc.start()
        time.sleep(0.1)

        # Act
        t1 = Tensor(x_init, device=self.device)
        t2 = Tensor(y_init, device=self.device)
        t3 = RPC.send(compute_tinygrad, t1, t2)
        t4 = Tensor(z_init, device=self.device)
        t5 = t3.mul(t4)

        # clean up
        RPC.done()
        proc.join()

        return t5.cpu().data


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
      t5 = torch.mul(t3, t4)

      # clean up
      rpc.shutdown()
      proc.join()

      return t5.detach().numpy()

    # Assert
    for x,y in zip(test_tinygrad(), test_pytorch()):
      np.testing.assert_allclose(x, y, atol=1e-5)


  def test_distributed_backward(self):

    def test_tinygrad():
        # Arrange
        proc = Process(target=run_remote_tinygrad, args=[self.port])
        proc.start()
        time.sleep(0.1)

        # Act
        t1 = Tensor(x_init, device=self.device)
        t2 = Tensor(y_init, device=self.device)
        t3 = RPC.send(compute_tinygrad, t1, t2)
        t4 = Tensor(z_init, device=self.device)
        t5 = t3.mul(t4)
        loss = t5.sum()
        loss.backward()

        # clean up
        RPC.done()
        proc.join()

        return loss.cpu().data, t1.grad.cpu().data, t2.grad.cpu().data, \
          t3.grad.cpu().data, t4.grad.cpu().data, t5.grad.cpu().data


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
      t5 = torch.mul(t3, t4)
      loss = t5.sum()
      loss.backward()

      # clean up
      rpc.shutdown()
      proc.join()

      return loss.detach().numpy(), t1.grad, t2.grad, \
        t3.grad, t4.grad, t5.grad


    # Assert
    for x,y in zip(test_tinygrad(), test_pytorch()):
      np.testing.assert_allclose(x, y, atol=1e-5)


  def setUp(self):
    # required by torch.rpc.init()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(self.port)


  def tearDown(self):
    del os.environ['MASTER_ADDR']
    del os.environ['MASTER_PORT']


@unittest.skipUnless(GPU, "Requires GPU")
class TestTinygradGPU(TestTinygrad):
  device = Device.GPU


if __name__ == '__main__':
  unittest.main()