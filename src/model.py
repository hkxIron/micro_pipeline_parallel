import torch.nn as nn


class ShardedMLP(nn.Module):
    def __init__(self, dim, total_layers, rank, world_size):
        super().__init__()
        # 1. Calculate how many layers THIS GPU is responsible for
        layers_per_gpu = total_layers // world_size # 每个GPU负责多少层

        self.rank = rank # NOTE: 当前rank所在的GPU
        self.is_first = rank == 0
        self.is_last = rank + 1 == world_size

        # 2. Build the local stack of layers
        layers = []
        # NOTE: 当前rank上需要创建的layer
        for _ in range(layers_per_gpu):
            # For a simple MLP, every layer looks the same
            # NOTE: 注意，此处并没有将layer移动到GPU上
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.ReLU())

        # NOTE: 最后一层需要loss
        if self.is_last:
            self.loss_fn = nn.CrossEntropyLoss(reduction="mean") # 默认是在batch维度对loss求平均
            layers.append(nn.Linear(dim, 2))

        # 3. Build the local network on the rank
        self.net = nn.Sequential(*layers)

    def forward(self, x, targets=None):
        # Run the local chunk of the network
        x = self.net(x)

        # Only the last GPU calculates loss
        # NOTE: 如果是最后一个rank, 则需要计算loss, shape: scalar
        if self.is_last and targets is not None:
            return self.loss_fn(x, targets)

        # Everyone else just returns the hidden states (activations)
        return x
