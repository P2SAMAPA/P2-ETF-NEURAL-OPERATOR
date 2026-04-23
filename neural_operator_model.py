"""
Neural Operator models: Fourier Neural Operator (FNO), DeepONet, and MLP fallback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        m1 = min(self.modes1, x_ft.size(-2))
        m2 = min(self.modes2, x_ft.size(-1))
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, n_layers=4, padding=0.1):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = padding
        self.fc0 = nn.Linear(1, self.width)
        self.convs = nn.ModuleList([SpectralConv2d(self.width, self.width, modes1, modes2) for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(n_layers)])
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize, H, W = x.shape
        x = x.unsqueeze(1)
        x = self.fc0(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        pad_h = int(self.padding * H)
        pad_w = int(self.padding * W)
        x = F.pad(x, [0, pad_w, 0, pad_h])
        for conv, w in zip(self.convs, self.ws):
            x1 = conv(x)
            x2 = w(x)
            x = x1 + x2
            x = F.gelu(x)
        x = x[..., :H, :W]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x).squeeze(-1)
        return x

class DeepONet(nn.Module):
    def __init__(self, branch_input_dim, trunk_input_dim=1, hidden_dim=128, n_layers=3):
        super().__init__()
        self.branch_net = self._build_mlp(branch_input_dim, hidden_dim, hidden_dim, n_layers)
        self.trunk_net = self._build_mlp(trunk_input_dim, hidden_dim, hidden_dim, n_layers)
        self.final_bias = nn.Parameter(torch.zeros(1))

    def _build_mlp(self, in_dim, hidden_dim, out_dim, n_layers):
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, out_dim))
        return nn.Sequential(*layers)

    def forward(self, x_branch, x_trunk):
        b = self.branch_net(x_branch)
        t = self.trunk_net(x_trunk)
        return torch.sum(b * t, dim=1, keepdim=True) + self.final_bias

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

class NeuralOperatorTrainer:
    def __init__(self, model_type="FNO", n_assets=23, modes=16, width=64, n_layers=4,
                 lr=0.001, weight_decay=1e-4, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model_type = model_type
        self.n_assets = n_assets
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_type == "FNO":
            effective_modes = min(modes, n_assets // 2 + 1)
            self.model = FNO2d(effective_modes, effective_modes, width, n_layers, padding=0.1).to(self.device)
        elif model_type == "DeepONet":
            branch_dim = n_assets * n_assets
            self.model = DeepONet(branch_dim, trunk_input_dim=2, hidden_dim=width, n_layers=n_layers).to(self.device)
        else:
            input_dim = n_assets * n_assets
            self.model = MLPRegressor(input_dim, hidden_dims=[256, 128, 64]).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.start_time = None

    def _flatten_covariance(self, cov_batch):
        batch_size = cov_batch.shape[0]
        return cov_batch.reshape(batch_size, -1)

    def fit(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, patience=10):
        self.start_time = time.time()
        self.model.train()
        dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                  torch.tensor(y_train, dtype=torch.float32))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                if self.model_type == "FNO":
                    pred = self.model(batch_X)
                    pred = pred.reshape(batch_X.size(0), -1)   # flatten to (batch, n_assets*n_assets)
                elif self.model_type == "DeepONet":
                    batch_X_flat = self._flatten_covariance(batch_X)
                    n_pairs = batch_X.shape[1] * batch_X.shape[2]
                    trunk = torch.arange(n_pairs, device=self.device).float().unsqueeze(0).expand(batch_X.size(0), -1)
                    pred = self.model(batch_X_flat, trunk)
                else:
                    batch_X_flat = self._flatten_covariance(batch_X)
                    pred = self.model(batch_X_flat)
                loss = self.criterion(pred, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
            train_loss /= len(dataset)

            self.model.eval()
            with torch.no_grad():
                X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self.device)
                if self.model_type == "FNO":
                    pred_val = self.model(X_val_t).reshape(X_val.shape[0], -1)
                elif self.model_type == "DeepONet":
                    X_val_flat = self._flatten_covariance(X_val_t)
                    n_pairs = X_val.shape[1] * X_val.shape[2]
                    trunk = torch.arange(n_pairs, device=self.device).float().unsqueeze(0).expand(X_val.shape[0], -1)
                    pred_val = self.model(X_val_flat, trunk)
                else:
                    X_val_flat = self._flatten_covariance(X_val_t)
                    pred_val = self.model(X_val_flat)
                val_loss = self.criterion(pred_val, y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

            if (epoch+1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            if time.time() - self.start_time > config.MAX_RUNTIME_SECONDS:
                print(f"    Max runtime exceeded. Switching to fallback model: {config.FALLBACK_MODEL}")
                self.model_type = config.FALLBACK_MODEL
                return False

        if best_state:
            self.model.load_state_dict(best_state)
        return True

    def predict(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            if self.model_type == "FNO":
                pred = self.model(X_t).reshape(X.shape[0], -1).cpu().numpy()
            elif self.model_type == "DeepONet":
                X_flat = self._flatten_covariance(X_t)
                n_pairs = X.shape[1] * X.shape[2]
                trunk = torch.arange(n_pairs, device=self.device).float().unsqueeze(0).expand(X.shape[0], -1)
                pred = self.model(X_flat, trunk).cpu().numpy()
            else:
                X_flat = self._flatten_covariance(X_t)
                pred = self.model(X_flat).cpu().numpy()
        return pred
