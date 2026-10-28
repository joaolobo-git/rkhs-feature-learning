import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold

# GPU if available, CPU otherwise (the original hard-coded "cuda")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)
DTYPE = torch.float64    # the one-step network below is computed in double precision


# =========================================================
# A. The network of Sec. 2, trained by ONE full-batch GD step on the first layer (Eq. 2.3)
# =========================================================
class OneStepNet(nn.Module):
    """
    f(x) = m^{-1/2} sum_j a_j relu(<w_j, x>), no biases, w_j ~ N(0, I_d/d), a ~ N(0, I_m/m)  (Sec. 2).
    Only W is trained, by one full-batch gradient step on L(W) = (1/2n) sum_i (f(x_i) - y_i)^2 with a fixed:
        W_1 = W_0 - eta * grad_W L(W_0).
    The second layer is then refitted by ridge regression on the features relu(<w_j, x>)/sqrt(m) (Eq. 2.4),
    which is kernel ridge regression with the feature kernel F F^T.
    """

    def __init__(self, d, m, seed=0):
        super().__init__()
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        W = torch.randn(m, d, generator=g, dtype=DTYPE, device="cpu") / math.sqrt(d)
        a = torch.randn(m, generator=g, dtype=DTYPE, device="cpu") / math.sqrt(m)
        self.W = nn.Parameter(W.to(DEVICE))       # layer 1, rows are the w_j
        self.register_buffer("a", a.to(DEVICE))    # layer 2, fixed during the step
        self.m = m

    def forward(self, X):
        return torch.relu(X @ self.W.T) @ self.a / math.sqrt(self.m)

    def first_layer_gradient(self, X, y):
        """grad_W of (1/2n) sum_i (f(x_i) - y_i)^2 at initialization."""
        X = torch.as_tensor(X, dtype=DTYPE, device=DEVICE)
        y = torch.as_tensor(y, dtype=DTYPE, device=DEVICE)
        self.W.grad = None
        loss = 0.5 * torch.mean((self.forward(X) - y) ** 2)
        loss.backward()
        return self.W.grad.detach().clone()

    @torch.no_grad()
    def features_after_step(self, X, grad, eta):
        """relu(X W_1^T)/sqrt(m) with W_1 = W_0 - eta * grad (eta = 0: random features at initialization)."""
        X = torch.as_tensor(X, dtype=DTYPE, device=DEVICE)
        W1 = self.W - eta * grad
        return (torch.relu(X @ W1.T) / math.sqrt(self.m)).cpu().numpy()


# =========================================================
# B. Original code (Adam-trained network used for the old Fig. 1b), unchanged except for the device
# =========================================================
class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


def find_lr_kfold(X_train, y_train, lr_candidates, k=5, epochs=1, batch_size=64):
    kf = KFold(n_splits=k, shuffle=True, random_state=123114)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    dataset = TensorDataset(X_train, y_train)
    input_size, hidden_size, output_size = X_train.shape[1], 400, 1
    lr_to_avg_loss = {}
    criterion = nn.MSELoss()
    for lr in lr_candidates:
        fold_losses = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True,
                                      generator=torch.Generator(device=DEVICE))
            val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size,
                                    generator=torch.Generator(device=DEVICE))
            model = TwoLayerNet(input_size, hidden_size, output_size).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            for epoch in range(epochs):
                model.train()
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    optimizer.zero_grad()
                    loss = criterion(model(inputs), targets)
                    loss.backward()
                    optimizer.step()
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    total_val_loss += criterion(model(inputs), targets).item()
            fold_losses.append(total_val_loss / len(val_loader))
        lr_to_avg_loss[lr] = np.mean(fold_losses)
    return min(lr_to_avg_loss, key=lr_to_avg_loss.get)


def train_model(X_train, y_train, lr, epochs=1):
    input_size, hidden_size, output_size = X_train.shape[1], 400, 1
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    model = TwoLayerNet(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True,
                              generator=torch.Generator(device=DEVICE))
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            loss = criterion(model(batch_X), batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    return model
