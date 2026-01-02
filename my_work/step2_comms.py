import torch
import torch.distributed as dist
import os

def init_distributed():
    """
    Initializes the distributed process group.
    Reads state directly from environment variables set by torchrun.
    """
    # 1. Read Environment Variables (set by torchrun)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    # 2. Set Device
    device = None
    # 3. Initialize Group
    
    return rank, world_size, device

class PipelineComms:
    def __init__(self, rank, world_size):
        pass

    def send_forward(self, tensor):
        """Send activation to the next GPU."""
        pass

    def recv_forward(self, shape, device, dtype=torch.float32):
        """Receive activation from the previous GPU."""
        pass

    def send_backward(self, tensor):
        """Send gradients back to the previous GPU."""
        pass

    def recv_backward(self, shape, device, dtype=torch.float32):
        """Receive gradients from the next GPU."""
        pass