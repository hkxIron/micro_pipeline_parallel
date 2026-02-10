"""
torchrun --nproc-per-node=4 src/main.py

main.py is a sheet of music.

torchrun is the conductor.

The Processes are 4 different musicians.

The conductor hands the same sheet of music to all 4 musicians.

local_rank is the instruction at the top of the sheet:

Musician 1's sheet says: "You are playing the Violin (GPU 0)."

Musician 2's sheet says: "You are playing the Viola (GPU 1)."

They all read the same notes (code), but they play on different instruments (GPUs) because of that initial setup instruction.

Literally, torchrun launches 4 copies of main.py.

It assigns RANK 0, 1, 2, 3 automatically.

Rank 0 creates random data and sends it to Rank 1...

Rank 3 calculates loss and sends gradients back to Rank 2...
"""

import time

import torch
import torch.optim as optim

# Import our modules
from comms import PipelineComms, init_distributed
from model import ShardedMLP
from schedule import one_forward_one_backward_pipeline_step

# Hyperparameters
BATCH_SIZE = 32
HIDDEN_DIM = 128
TOTAL_LAYERS = 16
STEPS = 50
CHUNKS = 8

# 1. Setup Distributed Environment
# device: 当前rank分配的GPU device
rank, world_size, device = init_distributed()
comms = PipelineComms(rank, world_size)

# Set base seed, then offset by rank to continue RNG sequence
torch.manual_seed(42)
# Each rank needs to "skip" the random numbers used by previous ranks
# Rough approximation: advance RNG state by rank * layers_per_rank * params_per_layer
# nn.Linear initialization uses a more complex pattern (Kaiming/He initialization
# typically samples from a normal distribution with specific parameters).
for i in range(
    rank * (TOTAL_LAYERS // world_size) * 2
):  # 2 params per layer (weight, bias)
    torch.randn(1)  # Consume RNG state

if rank == 0:
    print(f"--- Starting Micro PP on {world_size} Processes ({device}) ---")

# 2. Initialize the Sharded Model
# Each process only initializes its specific slice of layers
# rank为全局的id
model: ShardedMLP = ShardedMLP(HIDDEN_DIM, TOTAL_LAYERS, rank, world_size).to(device)

# 3. Setup Optimizer
# We only optimize the parameters present on THIS device
# NOTE: 将当前rank的参数传给optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Data Generation ---
# In Pipeline Parallelism, only Rank 0 loads the data.
if rank == 0:
    # rank =0 加载数据, [batch, hidden_dim]
    fixed_input = torch.randn(BATCH_SIZE, HIDDEN_DIM).to(device)
else:
    fixed_input = BATCH_SIZE

# Target (Labels)
# Only the Last Rank needs the targets to calc loss.
if rank == world_size - 1:
    # We want the model to learn to classify these random vectors into class '0' or '1'
    # target: [batch]
    fixed_target = torch.randint(0, 2, (BATCH_SIZE,)).to(device)
else:
    fixed_target = None

# 4. Training Loop
start_time = time.time()
model.train()

# 当前rank上的layer进行前向/反向
for step in range(STEPS):
    optimizer.zero_grad()
    if model.is_last:
        # 只有最后一层才需要计算loss
        # This function handles the Send/Recv/Compute orchestration
        loss = one_forward_one_backward_pipeline_step(
            model, comms, fixed_input, fixed_target, HIDDEN_DIM, CHUNKS, device
        )
    else:
        # This GPU doesn't know the loss; it just finished its communication/compute cycle
        one_forward_one_backward_pipeline_step(
            model, comms, fixed_input, fixed_target, HIDDEN_DIM, CHUNKS, device
        )

    # NOTE: 将当前rank的参数进行更新
    # Optimizer Step (All ranks do this locally after backward pass completes)
    optimizer.step()

    # --- Logging ---
    # Only the last rank (who calculates loss) can print the loss value
    # 只有最后一层所在的rank才需要计算loss
    if rank == world_size - 1 and step % 5 == 0:
        print(f"Step {step:02d} | Loss: {loss.item():.6f}")

# Clean up
if rank == world_size - 1:
    print("--- Training Complete ---")
    duration = time.time() - start_time
    print(f"Final Loss: {loss.item():.6f} | Time: {duration:.3f}s")

# Clean up
torch.distributed.destroy_process_group()
