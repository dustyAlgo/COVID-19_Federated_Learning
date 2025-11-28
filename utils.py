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

def add_dp_noise(features, epsilon=1.0, delta=1e-5):
    """Improved differential privacy with Gaussian mechanism"""
    # Calculate L2 sensitivity more accurately
    sensitivity = np.max(np.linalg.norm(features, ord=2, axis=1))
    
    # Gaussian mechanism parameters
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    
    noise = np.random.normal(0, sigma, size=features.shape)
    noisy_features = features + noise
    
    # Clip to maintain reasonable values
    noisy_features = np.clip(noisy_features, -10, 10)
    
    return noisy_features


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

# privacy estimation

def calculate_privacy_metrics(original_features, noisy_features):
    """Calculate privacy metrics like MSE and correlation"""
    mse = np.mean((original_features - noisy_features) ** 2)
    correlation = np.corrcoef(original_features.flatten(), noisy_features.flatten())[0, 1]
    
    return {
        'mse': mse,
        'correlation': correlation,
        'privacy_score': 1 / (1 + mse)  # Higher score = better privacy
    }

def measure_inversion_success(original_images, recovered_images):
    """Measure inversion attack success qualitatively"""
    # This would compare original vs recovered images
    # For now, return some metrics
    if original_images.shape != recovered_images.shape:
        return {"success_rate": 0.0, "similarity": 0.0}
    
    # Simple similarity measure (SSIM would be better)
    similarity = np.mean(np.abs(original_images - recovered_images))
    success_rate = 1.0 if similarity < 0.5 else 0.0  # Arbitrary threshold
    
    return {
        "success_rate": success_rate,
        "similarity": similarity
    }