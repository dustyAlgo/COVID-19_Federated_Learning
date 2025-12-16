# utils.py
import torch
import numpy as np
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, matthews_corrcoef
)
from sklearn.preprocessing import label_binarize


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
    k = max(1, int(features.shape[1] * top_k_ratio))
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
def evaluate_full_metrics(model, dataloader, device, num_classes=3):
    model.eval()
    
    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            probs = F.softmax(logits, dim=1)

            preds = torch.argmax(probs, dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # Core metrics
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # IoU / mIoU (classification IoU)
    ious = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else 0.0
        ious.append(iou)

    metrics["iou_per_class"] = ious
    metrics["miou"] = np.mean(ious)

    # ROC-AUC (One-vs-Rest)
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    auc_per_class = roc_auc_score(
        y_true_bin, y_prob, average=None, multi_class="ovr"
    )
    auc_macro = roc_auc_score(
        y_true_bin, y_prob, average="macro", multi_class="ovr"
    )

    metrics["auc_per_class"] = auc_per_class
    metrics["auc_macro"] = auc_macro

    return metrics, cm, y_true, y_prob

def plot_roc(y_true, y_prob, round_num, client_id, save_dir, class_names):
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))

    plt.figure()
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, label=f"{cls}")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve | Client {client_id} | Round {round_num}")
    plt.legend()
    plt.grid(True)

    plt.savefig(f"{save_dir}/roc_client_{client_id}_round_{round_num}.png", dpi=300)
    plt.close()

def plot_confusion_matrix(cm, round_num, client_id, save_dir, class_names):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues"
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix | Client {client_id} | Round {round_num}")

    plt.savefig(f"{save_dir}/cm_client_{client_id}_round_{round_num}.png", dpi=300)
    plt.close()


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
