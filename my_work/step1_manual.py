import time

import torch
import torch.nn as nn
import torch.optim as optim

# 1. Hyperparameters
BATCH_SIZE = 32
HIDDEN_DIM = 128
TOTAL_LAYERS = 16
STEPS = 50


class Part1(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        # TODO

    def forward(self, x):
        return self.net(x)


class Part2(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        # TODO

    def forward(self, x, targets):
        logits = self.net(x)
        return self.loss_fn(logits, targets)


# 3. Setup
torch.manual_seed(42)
cuda = False
if torch.cuda.is_available() and torch.cuda.device_count >= 2:
    cuda = True
# TODO
part1 = Part1(HIDDEN_DIM, TOTAL_LAYERS)
part2 = Part2(HIDDEN_DIM, TOTAL_LAYERS)

optimizer = optim.Adam(part1 + part2, lr=0.001)

# TODO
fixed_input = torch.randn(BATCH_SIZE, HIDDEN_DIM)
fixed_target = torch.randint(0, 2, (BATCH_SIZE,))

# 4. Training Loop
print("--- Training Manual Split (Bridge to Distributed) ---")
start_time = time.time()
part1.train()
part2.train()
for step in range(STEPS):
    optimizer.zero_grad()
    # --- FORWARD PASS ---
    hidden = part1(fixed_input)
    # TODO: device switch for cuda and retain_grad
    loss = part2(hidden, fixed_target)
    # --- BACKWARD PASS ---
    loss.backward()
    if step == 0:
        print(hidden.requires_grad, hidden.grad is not None, hidden.grad)
    optimizer.step()
    if step % 5 == 0:
        print(f"Step {step:02d} | Loss: {loss.item():.6f}")

duration = time.time() - start_time
print(f"Final Loss: {loss.item():.6f} | Time: {duration:.3f}s")
