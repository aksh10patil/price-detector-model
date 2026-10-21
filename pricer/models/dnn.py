import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from torch.optim.lr_scheduler import CosineAnnealingLR
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = type("nn", (), {"Module": object})
    _TORCH_AVAILABLE = False
from sklearn.feature_extraction.text import HashingVectorizer
from pricer.data.schema import Item


class ResidualBlock(nn.Module):
    """Residual MLP block with LayerNorm, ReLU, and skip connection."""

    def __init__(self, hidden_size: int, dropout_prob: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class DeepNeuralNetwork(nn.Module):
    """10-layer Deep Residual Neural Network for price regression."""

    def __init__(self, input_size: int, num_layers: int = 10, hidden_size: int = 4096, dropout_prob: float = 0.2):
        super().__init__()
        self.initial_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(hidden_size, dropout_prob) for _ in range(num_layers)]
        )
        self.final_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.initial_layer(x)
        for block in self.residual_blocks:
            x = block(x)
        return self.final_layer(x)


class DeepNeuralNetworkRunner:
    """Manages feature hashing, PyTorch training loop, and weight persistence."""

    def __init__(self, train: list[Item], val: list[Item], n_features: int = 100_000):
        self.vectorizer = HashingVectorizer(n_features=n_features, alternate_sign=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

        train_texts = [it.summary or it.full for it in train]
        val_texts = [it.summary or it.full for it in val]

        X_train = self.vectorizer.fit_transform(train_texts)
        X_val = self.vectorizer.transform(val_texts)

        y_train = np.array([it.price for it in train], dtype=np.float32)
        y_val = np.array([it.price for it in val], dtype=np.float32)

        self.train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train.toarray()).float(), torch.from_numpy(y_train).float()),
            batch_size=64,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val.toarray()).float(), torch.from_numpy(y_val).float()),
            batch_size=64,
            shuffle=False,
        )

        self.model = DeepNeuralNetwork(input_size=n_features).to(self.device)

    def train(self, epochs: int = 5, lr: float = 1e-3):
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for X_b, y_b in self.train_loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                pred = self.model(X_b).squeeze(-1)
                loss = criterion(pred, y_b)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch + 1}/{epochs} - Train MSE Loss: {avg_loss:.2f}")

    def predict(self, item: Item) -> float:
        self.model.eval()
        text = item.summary or item.full
        X = self.vectorizer.transform([text]).toarray()
        with torch.no_grad():
            tensor = torch.from_numpy(X).float().to(self.device)
            out = self.model(tensor).item()
        return float(out)

    def __call__(self, item: Item) -> float:
        return self.predict(item)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
