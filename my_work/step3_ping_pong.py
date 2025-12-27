import torch
from comms import init_distributed, PipelineComms

def ping_pong():
    '''
    Send a tensor from device rank 0 to device rank 1 and print to verify.
    '''
    rank, world_size, device = init_distributed()

if __name__ == "__main__":
    ping_pong()