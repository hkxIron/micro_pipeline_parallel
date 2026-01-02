import os

import torch

from comms import PipelineComms, init_distributed


def ping_pong():
    """
    Send a tensor from device rank 0 to device rank 1 and print to verify.
    """
    rank, world_size, device = init_distributed()
    # play with the barrier!
    # torch.distributed.barrier() is mainly used to synchronize all processes,
    # forcing them to wait until each has reached the barrier; this effectively
    # makes async distributed code temporarily synchronous at that point.
    torch.distributed.barrier()
    print(rank, world_size, device, os.getpid())
    comms = PipelineComms(rank, world_size)

    if rank == 0:
        tensor = torch.rand(3).to(device)
        print(f"Rank 0: Sending {tensor}")
        comms.send_forward(tensor)
    elif rank == 1:
        # Must know shape in advance!
        shape = (3,)
        received = comms.recv_forward(shape, device)
        print(f"Rank 1: Received {received}")


if __name__ == "__main__":
    ping_pong()
