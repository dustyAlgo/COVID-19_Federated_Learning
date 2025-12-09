# client.py
import torch
import torch.nn.functional as F
from utils import extract_features, sparsify, add_dp_noise
from torch.utils.data import DataLoader

class Client:
    def __init__(self, client_id, model, train_dataset, device):
        self.id = client_id
        self.model = model.to(device)
        self.train_data = DataLoader(train_dataset, batch_size=16, shuffle=True)
        self.device = device
    # client.py (Modification for Class Imbalance)

    def train_local(self, global_features=None, epochs=1, lr=0.001, alignment_weight=1.0, class_weights=None):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # FIX: Use Weighted Loss if weights are provided
        if class_weights is not None:
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

        self.model.train()
        for _ in range(epochs):
            for x, y in self.train_data:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                features = self.model.extract_features(x)
                out = self.model.classifier(features)

                # Weighted classification loss
                task_loss = loss_fn(out, y)

                # Feedback Loop (Feature Alignment)
                total_loss = task_loss
                if global_features is not None:
                    global_target = torch.tensor(global_features, dtype=torch.float32).to(self.device)
                    batch_mean = torch.mean(features, dim=0)
                    align_loss = F.mse_loss(batch_mean, global_target)
                    total_loss += alignment_weight * align_loss

                total_loss.backward()
                optimizer.step()

    # Prepare secure feature vector
    def get_secure_features(self, epsilon=1.0, top_k_ratio=0.1):
        features, labels = extract_features(self.model, self.train_data, self.device)

        sparse = sparsify(features, top_k_ratio)
        dp_features = add_dp_noise(sparse, epsilon)

        return dp_features