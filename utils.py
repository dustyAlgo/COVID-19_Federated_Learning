# utils.py
import torch
import numpy as np

# ----------------------------
# Extract features from model
# ----------------------------
def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            f = model.extract_features(x)
            features.append(f.cpu().numpy())
            labels.append(y.numpy())

    return np.vstack(features), np.hstack(labels)


# ----------------------------
# Top-K sparsification (10%)
# ----------------------------
def sparsify(features, top_k_ratio=0.1):
    k = int(features.shape[1] * top_k_ratio)
    topk_indices = np.argsort(np.abs(features), axis=1)[:, -k:]

    sparse = np.zeros_like(features)
    for i in range(features.shape[0]):
        sparse[i, topk_indices[i]] = features[i, topk_indices[i]]

    return sparse


# ----------------------------
# Differential Privacy
# Gaussian Mechanism
# ----------------------------
def add_dp_noise(features, epsilon=1.0):
    sensitivity = np.linalg.norm(features, ord=2)
    sigma = sensitivity / epsilon

    noise = np.random.normal(0, sigma, size=features.shape)
    return features + noise


# ----------------------------
# Server aggregation
# ----------------------------
def aggregate_feature_vectors(client_vectors):
    client_means = [np.mean(vec, axis=0) for vec in client_vectors]
    return np.mean(client_means, axis=0)

# Model evaluation
def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy