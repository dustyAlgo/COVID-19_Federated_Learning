import torch
import numpy as np
from server import Server
from client import Client
from models import LightweightCOVIDNet, ResNet18COVID, VGG11COVID
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import evaluate_model
import collections

# 1. Hyperparameters & Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_ROUNDS = 5
LOCAL_EPOCHS = 3          # Increased from 1 to allow better local learning
LEARNING_RATE = 0.001
ALIGNMENT_WEIGHT = 2.0    # Controls how much clients listen to the server
DP_EPSILON = 1.0          # Privacy budget
TOP_K_RATIO = 0.1         # Compression ratio

print(f"Running on {DEVICE}")
print(f"Config: {LOCAL_EPOCHS} Local Epochs, Align Weight: {ALIGNMENT_WEIGHT}")

# 2. Data Loading & Distribution Check

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load Datasets
hospital_a_dataset = datasets.ImageFolder(root='data/Hospital_A/', transform=transform)
hospital_b_dataset = datasets.ImageFolder(root='data/Hospital_B/', transform=transform)
hospital_c_dataset = datasets.ImageFolder(root='data/Hospital_C/', transform=transform)
client_datasets = [hospital_a_dataset, hospital_b_dataset, hospital_c_dataset]

# Load Validation Set
validation_dataset = datasets.ImageFolder(root='data/Validation_Set/', transform=transform)
validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False)

# --- Helper to Check Data Distribution (IID vs Non-IID) ---
def print_class_distribution(dataset, client_name):
    targets = dataset.targets
    counts = collections.Counter(targets)
    total = len(targets)
    dist_str = ", ".join([f"Class {k}: {v} ({v/total:.1%})" for k, v in counts.items()])
    print(f"[{client_name}] Total: {total} | {dist_str}")

print("\n--- Data Distribution Check ---")
print_class_distribution(hospital_a_dataset, "Hospital A")
print_class_distribution(hospital_b_dataset, "Hospital B")
print_class_distribution(hospital_c_dataset, "Hospital C")
print("-------------------------------\n")