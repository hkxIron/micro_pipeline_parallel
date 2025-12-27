import time
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Hyperparameters
BATCH_SIZE = 32
HIDDEN_DIM = 128
TOTAL_LAYERS = 16
STEPS = 50

# 2. Manual Split Classes
class Part1(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        layers = []
        for _ in range(depth // 2):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Part2(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        layers = []
        for _ in range(depth // 2):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dim, 2))
        self.net = nn.Sequential(*layers)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, targets):
        logits = self.net(x)
        return self.loss_fn(logits, targets)

# 3. Setup
torch.manual_seed(42)

part1 = Part1(HIDDEN_DIM, TOTAL_LAYERS)
part2 = Part2(HIDDEN_DIM, TOTAL_LAYERS)

# IMPORTANT: The optimizer must track parameters from BOTH parts
optimizer = optim.Adam(list(part1.parameters()) + list(part2.parameters()), lr=0.001)

# Generate fixed data
fixed_input = torch.randn(BATCH_SIZE, HIDDEN_DIM)
fixed_target = torch.randint(0, 2, (BATCH_SIZE,))

# 4. Training Loop
print("--- Training Manual Split (Bridge to Distributed) ---")
start_time = time.time()
part1.train(); part2.train()
for step in range(STEPS):
    optimizer.zero_grad()
    # --- FORWARD PASS ---
    # Step A: Run first half
    hidden = part1(fixed_input)
    # In PP, we'd send 'hidden' to the next GPU here.
    # We must ensure 'hidden' is ready to receive gradients later.
    hidden.retain_grad() 
    # Step B: Run second half using output of first
    loss = part2(hidden, fixed_target)
    # --- BACKWARD PASS ---
    loss.backward()
    # Because we called .retain_grad(), we can now see hidden.grad.
    # In PP, this is the tensor we would 'send_backward' to Rank 0.
    optimizer.step()
    if step % 5 == 0:
        print(f"Step {step:02d} | Loss: {loss.item():.6f}")

duration = time.time() - start_time
print(f"Final Loss: {loss.item():.6f} | Time: {duration:.3f}s")