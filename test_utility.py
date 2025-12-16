# test_utility.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from server import Server
from client import Client
from models import LightweightCOVIDNet, ResNet18COVID, VGG11COVID
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import evaluate_model

# --- 1. Setup ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running Utility Test on: {DEVICE}")

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load Data (Safe fallback included)
try:
    if not os.path.exists('data'): raise FileNotFoundError("Data folder missing")
    hospital_a_dataset = datasets.ImageFolder(root='data/Hospital_A/', transform=transform)
    hospital_b_dataset = datasets.ImageFolder(root='data/Hospital_B/', transform=transform)
    hospital_c_dataset = datasets.ImageFolder(root='data/Hospital_C/', transform=transform)
    validation_dataset = datasets.ImageFolder(root='data/Validation_Set/', transform=transform)
    
    client_datasets = [hospital_a_dataset, hospital_b_dataset, hospital_c_dataset]
    validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False)
except Exception as e:
    print(f"Warning: Using Dummy Data. Reason: {e}")
    hospital_a_dataset = datasets.FakeData(size=100, transform=transform)
    client_datasets = [hospital_a_dataset, hospital_a_dataset, hospital_a_dataset]
    validation_loader = DataLoader(datasets.FakeData(size=50, transform=transform), batch_size=16)

# --- 2. Experiment Logic ---
def run_noise_vs_accuracy():
    epsilon_values = [0.5, 1.0, 5.0, 10.0]  # Compare High Noise (0.5) vs Low Noise (10.0)
    results = {eps: [] for eps in epsilon_values}
    
    print("\n=== Experiment: Noise Scale (ε) vs Accuracy ===")
    
    for epsilon in epsilon_values:
        print(f"\n--- Testing with epsilon={epsilon} ---")
        
        # Fresh System for each epsilon
        server = Server()
        clients = [
            Client(0, LightweightCOVIDNet(), client_datasets[0], DEVICE),
            Client(1, ResNet18COVID(), client_datasets[1], DEVICE),
            Client(2, VGG11COVID(), client_datasets[2], DEVICE)
        ]
        
        global_features = None  # Reset global knowledge
        
        # Short training loop (5 rounds is enough to see the trend)
        for r in range(5):
            client_vectors = []
            
            # Client Phase
            for client in clients:
                client.train_local(
                    global_features=global_features, 
                    epochs=3,  # Good balance for speed/learning
                    alignment_weight=1.0
                )
                
                # Get noisy features
                fv = client.get_secure_features(epsilon=epsilon, top_k_ratio=0.1)
                client_vectors.append(fv)
            
            # Server Phase
            server.receive(client_vectors)
            global_features = server.send_global()
            
            # Evaluation Phase
            acc_list = [evaluate_model(c.model, validation_loader, DEVICE) for c in clients]
            avg_acc = np.mean(acc_list)
            results[epsilon].append(avg_acc)
            
            print(f"   Round {r+1}: Avg Accuracy = {avg_acc:.2f}%")
    
    # --- 3. Plotting ---
    plt.figure(figsize=(10, 6))
    for eps, accs in results.items():
        plt.plot(range(1, 6), accs, marker='o', label=f'ε={eps}')
    
    plt.xlabel('Federated Round')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Impact of Differential Privacy (ε) on Model Utility')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    filename = 'utility_results.png'
    plt.savefig(filename)
    print(f"\nDone! Results saved to {filename}")
    # plt.show() # Uncomment if running locally with a screen

if __name__ == "__main__":
    run_noise_vs_accuracy()