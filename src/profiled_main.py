import time

import torch
import torch.optim as optim

# Import our modules
from comms import PipelineComms, init_distributed
from model import ShardedMLP
from profiled_schedule import *
from profiler import PipelineProfiler
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument("--pp_type", type=str, default='naive', help="流水线的类型，有naive, gpipe, 1f1b")
args = parser.parse_args()

# Hyperparameters
BATCH_SIZE = 32
HIDDEN_DIM = 128
TOTAL_LAYERS = 64
STEPS = 50
CHUNKS = 8

# 1. Setup Distributed Environment
rank, world_size, device = init_distributed()
comms = PipelineComms(rank, world_size)
# NOTE: 用于记录每个rank的通信时间, 每个rank都有一个单独的profiler
profiler = PipelineProfiler(rank)

torch.manual_seed(42)
# Each rank needs to "skip" the random numbers used by previous ranks
for i in range(
    rank * (TOTAL_LAYERS // world_size) * 2
):  # 2 params per layer (weight, bias)
    torch.randn(1)  # Consume RNG state

if rank == 0:
    print(f"--- Starting Micro PP on {world_size} Processes ({device}) ---")
    print(f"pp_type:{args.pp_type} profiler")

# 2. Initialize the Sharded Model
model = ShardedMLP(HIDDEN_DIM, TOTAL_LAYERS, rank, world_size).to(device)

# 3. Setup Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 4. Only Rank 0 loads the data.
if rank == 0:
    fixed_input = torch.randn((BATCH_SIZE, HIDDEN_DIM)).to(device)
else:
    fixed_input = BATCH_SIZE
# 5. Only the Last Rank needs the targets to calc loss.
if rank == world_size - 1:
    # We want the model to learn to classify these random vectors into class '0' or '1'
    fixed_target = torch.randint(0, 2, (BATCH_SIZE,)).to(device)
else:
    fixed_target = None

start_time = time.time()
model.train()
for step in range(STEPS):
    optimizer.zero_grad()
    if rank == world_size - 1:
        if args.pp_type == 'naive':
            loss = naive_pipeline_step(
                model,
                comms,
                profiler,
                fixed_input,
                fixed_target,
                HIDDEN_DIM,
                device,
            )
        elif args.pp_type == 'gpipe':
            loss = gpipe_pipeline_step(
                model,
                comms,
                profiler,
                fixed_input,
                fixed_target,
                HIDDEN_DIM,
                CHUNKS,
                device,
            )
        elif args.pp_type == '1f1b':
            loss = onef_oneb_pipeline_step(
                model,
                comms,
                profiler,
                fixed_input,
                fixed_target,
                HIDDEN_DIM,
                CHUNKS,
                device,
            )
    else:
        if args.pp_type == 'naive':
            naive_pipeline_step(
                model,
                comms,
                profiler,
                fixed_input,
                fixed_target,
                HIDDEN_DIM,
                device,
            )
        elif args.pp_type == 'gpipe':
            gpipe_pipeline_step(
                model,
                comms,
                profiler,
                fixed_input,
                fixed_target,
                HIDDEN_DIM,
                CHUNKS,
                device,
            )
        elif args.pp_type == '1f1b':
            onef_oneb_pipeline_step(
                model,
                comms,
                profiler,
                fixed_input,
                fixed_target,
                HIDDEN_DIM,
                CHUNKS,
                device,
            )

    # ================
    optimizer.step()
    if rank == world_size - 1 and step % 5 == 0:
        print(f"Step {step:02d} | Loss: {loss.item():.6f}")
# Clean up
if rank == world_size - 1:
    print("--- Training Complete ---")
    duration = time.time() - start_time
    print(f"Final Loss: {loss.item():.6f} | Time: {duration:.3f}s")

# 每个profile单独打印自己所在rank上的时间
profiler.print_summary()
torch.distributed.destroy_process_group()

"""
 torchrun --nproc-per-node=2 src/profiled_main.py --pp_type=1f1b
"""