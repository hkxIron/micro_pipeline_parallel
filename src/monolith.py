import torch
import torch.nn as nn
import torch.optim as optim

# 1. Hyperparameters
HIDDEN_DIM = 128
TOTAL_LAYERS = 16
STEPS = 50
BATCH_SIZE = 32

# 2. The Monolithic Model
# This is what we will eventually "shard" across multiple GPUs
class MonolithicMLP(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, targets):
        logits = self.net(x)
        return self.loss_fn(logits, targets)

# 3. Setup
torch.manual_seed(42)
device = torch.device("cpu")

model = MonolithicMLP(HIDDEN_DIM, TOTAL_LAYERS).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Generate one fixed batch to overfit
fixed_input = torch.randn(BATCH_SIZE, HIDDEN_DIM).to(device)
fixed_target = torch.randint(0, 2, (BATCH_SIZE,)).to(device)

# 4. Training Loop
print("--- Training Monolith (Ground Truth) ---")
model.train()
for step in range(STEPS):
    optimizer.zero_grad()
    
    # Simple forward and backward
    loss = model(fixed_input, fixed_target)
    loss.backward()
    optimizer.step()
    
    if step % 5 == 0:
        print(f"Step {step:02d} | Loss: {loss.item():.6f}")

print(f"Final Monolith Loss: {loss.item():.6f}")