import time

import torch
import torch.optim as optim

# Import our modules
from step6_schedule import naive_pipeline_step

# Hyperparameters
BATCH_SIZE = 32
HIDDEN_DIM = 128
TOTAL_LAYERS = 16
STEPS = 50

# 1. Setup Distributed Environment TODO
rank, world_size, device = None
comms = None

torch.manual_seed(42)
# Each rank needs to "skip" the random numbers used by previous ranks
for i in range(
    rank * (TOTAL_LAYERS // world_size) * 2
):  # 2 params per layer (weight, bias)
    torch.randn(1)  # Consume RNG state


if rank == 0:
    print(f"--- Starting Micro PP on {world_size} Processes (Mac/CPU) ---")

# 2. Initialize the Sharded Model TODO
model = None

# 3. Setup Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 4. Only Rank 0 loads the data. TODO
if rank == 0:
    fixed_input = None
else:
    pass
# 5. Only the Last Rank needs the targets to calc loss. TODO
if rank == world_size - 1:
    # We want the model to learn to classify these random vectors into class '0' or '1'
    fixed_target = None
else:
    pass

model.train()
for step in range(STEPS):
    optimizer.zero_grad()
    start_time = time.time()
    loss = naive_pipeline_step(
        model, comms, fixed_input, fixed_target, HIDDEN_DIM, device
    )
    optimizer.step()
    if rank == world_size - 1 and step % 5 == 0:
        print(f"Step {step:02d} | Loss: {loss.item():.6f}")

# Clean up
if rank == 0:
    print("--- Training Complete ---")
    duration = time.time() - start_time
    print(f"Final Loss: {loss.item():.6f} | Time: {duration:.3f}s")
torch.distributed.destroy_process_group()
