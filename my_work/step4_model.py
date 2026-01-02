import torch.nn as nn


class ShardedMLP(nn.Module):
    def __init__(self, hidden_dim, total_layers, rank, world_size):
        super().__init__()

        # 1. Calculate how many layers THIS GPU is responsible for

        # 2. Build the local stack of layers

    def forward(self, x, targets=None):
        # Run the local chunk of the network
        x = self.net(x)

        # Only the last GPU calculates loss

        # Everyone else just returns the hidden states (activations)
