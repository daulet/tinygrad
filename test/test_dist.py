import torch
import torch.distributed.rpc as rpc
import unittest

class TestTinygrad(unittest.TestCase):
  gpu = False

  def test_distributed_grad():
    def test_tinygrad():
        """
        TODO
        dial workers
        assume the other agent is running exactly the same model,
        so can ref func by ID instead of trying to serialize it
        """
        pass

    def test_pytorch():
      def my_add(t1, t2):
        return torch.add(t1, t2)

      # On worker 0:
      t1 = torch.rand((3, 3), requires_grad=True)
      t2 = torch.rand((3, 3), requires_grad=True)

      # Perform some computation remotely.
      t3 = rpc.rpc_sync("worker1", my_add, args=(t1, t2))

      # Perform some computation locally based on remote result.
      t4 = torch.rand((3, 3), requires_grad=True)
      t5 = torch.mul(t3, t4)

      # Compute some loss.
      loss = t5.sum()

    for x,y in zip(test_tinygrad(), test_pytorch()):
      np.testing.assert_allclose(x, y, atol=1e-5)


@unittest.skipUnless(GPU, "Requires GPU")
class TestTinygradGPU(TestTinygrad):
  gpu = True

if __name__ == '__main__':
  unittest.main()