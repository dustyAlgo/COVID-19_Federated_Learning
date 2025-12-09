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
NUM_ROUNDS = 2
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

# We normalize so minority classes have higher weight.

weights_tensor = torch.tensor([1.0, 0.33, 1.0]).float().to(DEVICE)

# 3. Setup System

server = Server()

# Initialize clients with different architectures (System Heterogeneity)
clients = [
    Client(0, LightweightCOVIDNet(), client_datasets[0], DEVICE),  # Low resource
    Client(1, ResNet18COVID(), client_datasets[1], DEVICE),        # Standard
    Client(2, VGG11COVID(), client_datasets[2], DEVICE)            # High performance
]
# 4. Federated Training Loop

# Initialize global knowledge as None for the first round
global_features = None

for round_num in range(NUM_ROUNDS):
    print(f"\n=== Round {round_num + 1}/{NUM_ROUNDS} ===")

    client_vectors = []

    # CLIENT PHASE
    for i, client in enumerate(clients):
        print(f" > Client {i} training...", end='\r')
        
        # 1. Train locally using Global Knowledge from previous round
        client.train_local(
            global_features=global_features, 
            epochs=LOCAL_EPOCHS, 
            lr=LEARNING_RATE,
            alignment_weight=ALIGNMENT_WEIGHT,
            class_weights=weights_tensor
        )
        
        # 2. Extract and sanitize features
        fv = client.get_secure_features(epsilon=DP_EPSILON, top_k_ratio=TOP_K_RATIO)
        client_vectors.append(fv)
    
    print("\nAll clients finished training.")

    # SERVER PHASE
    # 1. Aggregate features
    server.receive(client_vectors)
    
    # 2. Update Global Knowledge for the NEXT round
    global_features = server.send_global()
    
    print(f" > Global Prototype Shape: {global_features.shape}")


    print("\nValidation Accuracy:")
    avg_acc = 0
    for i, client in enumerate(clients):
        acc = evaluate_model(client.model, validation_loader, DEVICE)
        avg_acc += acc
        print(f"   Client {i} ({type(client.model).__name__}): {acc:.2f}%")
    print(f"   Average System Accuracy: {avg_acc / len(clients):.2f}%")


# 5. Final Summary
print("\n=== Training Complete ===")
# Optional: Save models
torch.save(clients[0].model.state_dict(), "client_0_final.pth")