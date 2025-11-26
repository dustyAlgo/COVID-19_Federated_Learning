# client.py
import torch
from utils import extract_features, sparsify, add_dp_noise
from torch.utils.data import DataLoader

class Client:
    def __init__(self, client_id, model, train_dataset, device):
        self.id = client_id
        self.model = model.to(device)
        self.train_data = DataLoader(train_dataset, batch_size=16, shuffle=True)
        self.device = device

    # ----------------------------
    # Local training step
    # ----------------------------
    def train_local(self, epochs=1, lr=0.001):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        self.model.train()
        for _ in range(epochs):
            for x, y in self.train_data:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.model(x)
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()

    # ----------------------------
    # Prepare secure feature vector
    # ----------------------------
    def get_secure_features(self, epsilon=1.0, top_k_ratio=0.1):
        features, labels = extract_features(self.model, self.train_data, self.device)

        sparse = sparsify(features, top_k_ratio)
        dp_features = add_dp_noise(sparse, epsilon)

        return dp_features
