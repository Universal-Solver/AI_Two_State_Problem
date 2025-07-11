import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. Create toy descriptors (e.g., x-coordinates in 1D double well)
n_samples = 500
x = np.linspace(-2, 2, n_samples).reshape(-1, 1)

# 2. Define A and B states (e.g., left and right wells)
stateA = (x < -0.5).astype(np.float32)  # label 0
stateB = (x > 0.5).astype(np.float32)   # label 1

# For simplicity, label A as 0, B as 1, and ignore others
labels = np.full_like(x, -1, dtype=np.float32)
labels[stateA == 1.0] = 0.0
labels[stateB == 1.0] = 1.0

# Filter out unlabeled samples
mask = labels != -1
x_labeled = x[mask.flatten()]
y_labeled = labels[mask.flatten()]

# Convert to PyTorch tensors
descriptors = torch.tensor(x_labeled, dtype=torch.float32)
targets = torch.tensor(y_labeled, dtype=torch.float32).unsqueeze(1)  # binary classification

# 3. Define feedforward neural network
class FFNet(nn.Module):
    def __init__(self, n_in, n_hidden, activation=nn.ELU()):
        super(FFNet, self).__init__()
        layers = []
        prev = n_in
        for h in n_hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(activation)
            prev = h
        layers.append(nn.Linear(prev, 1))  # Single output: predicts p_B
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# 4. Instantiate the network and optimizer
n_in = 1
ffnet = FFNet(n_in=n_in, n_hidden=[4, 2], activation=nn.ELU())
optimizer = optim.Adam(ffnet.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()  # binary classification loss

# 5. Define the descriptor transform (identity here)
def descriptor_transform(snapshot):
    return snapshot

# 6. Simulate a training loop
for step in range(10):
    # simulate snapshot (random sample)
    idx = torch.randint(0, descriptors.shape[0], (1,))
    snapshot = descriptors[idx]
    label = targets[idx].view(-1, 1)

    # forward pass
    pred_logit = ffnet(snapshot)
    loss = loss_fn(pred_logit, label)

    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # prediction
    with torch.no_grad():
        pred_prob = torch.sigmoid(pred_logit)  # p_B
        print(f"Step {step:2d}: x = {snapshot.item():+.3f} → p_B ≈ {pred_prob.item():.3f}")

