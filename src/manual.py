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
# As we can see, pipeline stages aren't always identical;
# here Part2 contains the model head
# In a transformer, the first stage has an Embedding/Input layer,
# the middle has Transformer/MLP blocks, and the
# last has the Classifier/Loss
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
        # model head
        layers.append(nn.Linear(dim, 2))
        self.net = nn.Sequential(*layers)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, targets):
        logits = self.net(x)
        return self.loss_fn(logits, targets)

# 3. Setup
torch.manual_seed(42)
cuda = False
if torch.cuda.is_available() and torch.cuda.device_count >= 2:
    cuda = True
if cuda:
    part1 = Part1(HIDDEN_DIM, TOTAL_LAYERS).to(torch.cuda.device(0))
    part2 = Part2(HIDDEN_DIM, TOTAL_LAYERS).to(torch.cuda.device(1))
else:
    part1 = Part1(HIDDEN_DIM, TOTAL_LAYERS)
    part2 = Part2(HIDDEN_DIM, TOTAL_LAYERS)

# IMPORTANT: The optimizer must track parameters from BOTH parts
optimizer = optim.Adam(list(part1.parameters()) + list(part2.parameters()), lr=0.001)

# Generate fixed data
if cuda:
    fixed_input = torch.randn(BATCH_SIZE, HIDDEN_DIM).to(torch.cuda.device(0))
    fixed_target = torch.randint(0, 2, (BATCH_SIZE,)).to(torch.cuda.device(1))
else:
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
    if cuda:
        hidden = hidden.to(torch.cuda.device(1))
    # We call hidden.retain_grad() strictly for teaching purposes—to
    # see exactly what value would have been sent over the network 
    # if this were distributed. Without it, hidden.grad would be None.
    hidden.retain_grad()
    # Step B: Run second half using output of first
    loss = part2(hidden, fixed_target)
    # --- BACKWARD PASS ---
    '''
    loss.backward() backpropagates through both parts automatically.
    PyTorch's autograd maintains the computation graph across the forward pass:
    hidden = part1(fixed_input) → graph includes part1
    hidden = hidden.to(device(1)) → .to() is part of the graph
    loss = part2(hidden, fixed_target) → graph extends through part2
    loss.backward() → backpropagates through the entire graph:
    Computes gradients for part2's parameters
    Computes hidden.grad (input gradient to part2)
    Continues back through .to() operation
    Computes gradients for part1's parameters
    The graph connects automatically because:
    hidden from part1() has requires_grad=True
    .to() operations preserve the graph connection
    The optimizer tracks both parts' parameters
    '''
    loss.backward()
    # Verify both parts get gradients
    # The distributed version (in schedule.py) manually handles
    # this because it's split across processes; in a single
    # process, autograd handles gradients automatically.
    if step == 0:  # Check once
        p1_has_grad = any(p.grad is not None for p in part1.parameters())
        p2_has_grad = any(p.grad is not None for p in part2.parameters())
        print(f"Part1 has grads: {p1_has_grad}, Part2 has grads: {p2_has_grad}")
    # Because we called .retain_grad(), we can now see hidden.grad.
    # In PP, this is the tensor we would 'send_backward' to Rank 0.
    optimizer.step()
    if step % 5 == 0:
        print(f"Step {step:02d} | Loss: {loss.item():.6f}")

duration = time.time() - start_time
print(f"Final Loss: {loss.item():.6f} | Time: {duration:.3f}s")