import torch
from server import Server
from client import Client
from models import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import evaluate_model  # New import

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------
# Real Dataset Loading for 3 Hospitals
# -----------------------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load TRAINING datasets for each hospital
hospital_a_dataset = datasets.ImageFolder(root='data/Hospital_A/', transform=transform)
hospital_b_dataset = datasets.ImageFolder(root='data/Hospital_B/', transform=transform)
hospital_c_dataset = datasets.ImageFolder(root='data/Hospital_C/', transform=transform)

client_datasets = [hospital_a_dataset, hospital_b_dataset, hospital_c_dataset]

# Load VALIDATION dataset (shared test set)
validation_dataset = datasets.ImageFolder(root='data/Validation_Set/', transform=transform)
validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False)

print("Dataset Info:")
print(f"Hospital A: {len(hospital_a_dataset)} training images")
print(f"Hospital B: {len(hospital_b_dataset)} training images")
print(f"Hospital C: {len(hospital_c_dataset)} training images")
print(f"Validation: {len(validation_dataset)} test images")
print(f"Classes: {hospital_a_dataset.classes}")

# -----------------------------------------
# Create server + clients with different architectures
# -----------------------------------------
server = Server()
clients = [
    Client(0, LightweightCOVIDNet(), client_datasets[0], device),  # Custom CNN
    Client(1, ResNet18COVID(), client_datasets[1], device),        # ResNet18
    Client(2, VGG11COVID(), client_datasets[2], device)            # VGG11
]

# -----------------------------------------
# Federated Rounds with Evaluation
# -----------------------------------------
for round in range(5):
    print(f"\n--- Round {round+1} ---")

    client_vectors = []

    # CLIENT SIDE: Training and feature extraction
    for client in clients:
        client.train_local(epochs=1)
        fv = client.get_secure_features(epsilon=1.0, top_k_ratio=0.1)
        client_vectors.append(fv)

    # SERVER SIDE: Aggregation
    server.receive(client_vectors)
    global_features = server.send_global()
    print("Global feature vector shape:", global_features.shape)

    # MODEL EVALUATION
    print("\nModel Accuracy on Validation Set:")
    for i, client in enumerate(clients):
        accuracy = evaluate_model(client.model, validation_loader, device)
        print(f"Client {i} ({type(client.model).__name__}): {accuracy:.2f}%")

# Final evaluation
print("\n=== Final Results ===")
for i, client in enumerate(clients):
    accuracy = evaluate_model(client.model, validation_loader, device)
    print(f"Final - Client {i}: {accuracy:.2f}%")