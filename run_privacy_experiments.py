import torch
import numpy as np
import matplotlib.pyplot as plt
import copy

# Import your local modules
from server import Server
from client import Client
from models import * # Assuming LightweightCOVIDNet, ResNet18COVID, VGG11COVID are here
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import evaluate_model

# 1. Setup Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Data Loading Setup
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# NOTE: Update these paths to match your actual folder structure
try:
    hospital_a_dataset = datasets.ImageFolder(root='data/Hospital_A/', transform=transform)
    hospital_b_dataset = datasets.ImageFolder(root='data/Hospital_B/', transform=transform)
    hospital_c_dataset = datasets.ImageFolder(root='data/Hospital_C/', transform=transform)
    client_datasets = [hospital_a_dataset, hospital_b_dataset, hospital_c_dataset]

    validation_dataset = datasets.ImageFolder(root='data/Validation_Set/', transform=transform)
    validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False)
except Exception as e:
    print(f"Warning: Could not load datasets. Check paths. Error: {e}")
    # Create dummy datasets for testing if files are missing
    hospital_a_dataset = datasets.FakeData(transform=transform)
    hospital_b_dataset = datasets.FakeData(transform=transform)
    hospital_c_dataset = datasets.FakeData(transform=transform)
    client_datasets = [hospital_a_dataset, hospital_b_dataset, hospital_c_dataset]
    validation_loader = DataLoader(datasets.FakeData(transform=transform), batch_size=16)

# ==========================================
# Core Helper Functions
# ==========================================

def simple_feature_inversion(model, target_features, device="cpu", steps=500):
    """
    Real Optimization-Based Inversion Attack.
    Attempts to reconstruct the input image 'x' such that Model(x) matches 'target_features'.
    """
    model.eval()
    
    # Batch size detection
    batch_size = target_features.shape[0]
    
    # 1. Start with random noisy images (The 'Dummy' input)
    # Shape: (Batch, Channels=1, H=128, W=128)
    dummy_input = torch.randn(batch_size, 1, 128, 128, requires_grad=True, device=device)
    
    # 2. Optimizer to adjust the dummy input
    optimizer = torch.optim.Adam([dummy_input], lr=0.1)
    
    # Convert target features to tensor
    target = torch.tensor(target_features).to(device)
    
    print(f"  > Running inversion optimization for {steps} steps...")
    
    # 3. Optimization Loop
    for i in range(steps):
        optimizer.zero_grad()
        
        # Forward pass on dummy input
        dummy_features = model.extract_features(dummy_input)
        
        # Loss: Mean Squared Error between dummy features and real (noisy) features
        loss = ((dummy_features - target)**2).mean()
        
        # Backprop (update dummy_input, NOT the model)
        loss.backward()
        optimizer.step()
        
    # Return reconstructed images as numpy
    return dummy_input.detach().cpu().numpy()

def visualize_inversion_results(original_images, recovered_dict, labels, class_names):
    """
    Visualizes the Original Image vs Reconstructed Images at different Epsilon levels.
    """
    plt.figure(figsize=(12, 8))
    
    num_samples = len(original_images)
    num_cols = 1 + len(recovered_dict) # Original + Num of Privacy Levels
    
    for i in range(num_samples):
        # 1. Plot Original Image
        ax = plt.subplot(num_samples, num_cols, i * num_cols + 1)
        # Handle shape (1, 128, 128) -> (128, 128)
        img_disp = original_images[i].squeeze()
        plt.imshow(img_disp, cmap='gray')
        if i == 0: plt.title("Original\nInput", fontsize=10, fontweight='bold')
        plt.ylabel(f"Class: {labels[i]}", fontsize=9)
        plt.xticks([]); plt.yticks([])
        
        # 2. Plot Recovered Images for each Epsilon
        for j, (eps, recovered_batch) in enumerate(recovered_dict.items()):
            ax = plt.subplot(num_samples, num_cols, i * num_cols + j + 2)
            
            # Recovered batch is numpy array (Batch, 1, 128, 128)
            rec_disp = recovered_batch[i].squeeze()
            
            plt.imshow(rec_disp, cmap='gray')
            if i == 0: plt.title(f"Reconstructed\n(ε = {eps})", fontsize=10)
            plt.xticks([]); plt.yticks([])
            
    plt.tight_layout()
    plt.savefig('inversion_attack_results.png', dpi=300)
    print("Saved inversion_attack_results.png")
    plt.show()

# ==========================================
# Experiment 1: Noise vs Accuracy
# ==========================================

def run_noise_vs_accuracy_experiment():
    """Test different noise levels (epsilon values) against model accuracy."""
    epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] 
    results = {eps: [] for eps in epsilon_values}
    
    print("\n=== Experiment 1: Noise Scale vs Accuracy ===")
    
    for epsilon in epsilon_values:
        print(f"\nRunning training with epsilon={epsilon}...")
        
        # Initialize Fresh Server & Clients
        server = Server()
        # Note: Assuming Client() takes (id, model, dataset, device)
        clients = [
            Client(0, LightweightCOVIDNet(), client_datasets[0], device),
            Client(1, ResNet18COVID(), client_datasets[1], device),
            Client(2, VGG11COVID(), client_datasets[2], device)
        ]
        
        round_accuracies = []
        
        # Short training loop for experiment (e.g., 5 rounds)
        for r in range(5):
            client_vectors = []
            for client in clients:
                client.train_local(epochs=1) 
                # Assuming get_secure_features handles Sparsification + DP
                fv = client.get_secure_features(epsilon=epsilon, top_k_ratio=0.1)
                client_vectors.append(fv)
            
            server.receive(client_vectors)
            _ = server.send_global() # Broadcast global prototype
            
            # Evaluate
            acc_list = []
            for client in clients:
                acc = evaluate_model(client.model, validation_loader, device)
                acc_list.append(acc)
            
            avg_acc = np.mean(acc_list)
            round_accuracies.append(avg_acc)
            print(f"  Round {r+1}: Avg Accuracy = {avg_acc:.2f}%")
        
        results[epsilon] = round_accuracies
    
    # Plotting
    plt.figure(figsize=(10, 6))
    for eps, accs in results.items():
        plt.plot(range(1, 6), accs, marker='o', label=f'ε={eps}')
    
    plt.xlabel('Federated Round')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Impact of Differential Privacy Noise (ε) on Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('noise_vs_accuracy.png', dpi=300)
    print("Saved noise_vs_accuracy.png")
    plt.show()
    
    return results

# ==========================================
# Experiment 2: Inversion Attack Recovery
# ==========================================

def inversion_attack_experiment():
    """Qualitative inversion attack recovery demonstration."""
    print("\n=== Experiment 2: Inversion Attack Recovery ===")
    
    # Use Client 0 for this test
    client = Client(0, LightweightCOVIDNet(), client_datasets[0], device)
    
    # Get a specific batch of images to attack
    # We use a fresh loader to guarantee we get the same images for 'Original' and 'Attack'
    loader = DataLoader(client.train_data, batch_size=5, shuffle=True)
    images, labels = next(iter(loader))
    images = images.to(device)
    
    # 1. Get Original Images (for visualization)
    original_images_np = images.cpu().numpy()
    
    # 2. Extract CLEAN features (Target for reconstruction reference)
    with torch.no_grad():
        clean_features = client.model.extract_features(images)
    
    privacy_levels = [0.1, 1.0, 10.0] # Strong to Weak
    recovered_images_dict = {}
    
    for eps in privacy_levels:
        print(f"\nAttacking privacy level epsilon={eps}...")
        
        # --- SIMULATE CLIENT NOISE MECHANISM ---
        # We manually apply noise here to guarantee it applies to THIS batch
        # Sparsification (Top 10%)
        top_k_ratio = 0.1
        k = int(top_k_ratio * clean_features.numel() / clean_features.shape[0])
        
        # Flatten per batch item to find threshold
        flat_feats = clean_features.abs().flatten(1)
        topk_vals, _ = torch.topk(flat_feats, k)
        threshold = topk_vals[:, -1].unsqueeze(1)
        
        # Apply mask
        mask = flat_feats >= threshold
        sparse_features = clean_features.flatten(1) * mask
        
        # Add DP Noise
        sensitivity = 1.0 
        noise_scale = sensitivity / eps
        noise = torch.normal(0, noise_scale, size=sparse_features.shape).to(device)
        
        # Final Noisy Features
        noisy_features = (sparse_features + noise).reshape(clean_features.shape)
        # ---------------------------------------
        
        # 3. Perform Inversion Attack
        # We try to reconstruct the image from 'noisy_features'
        recovered = simple_feature_inversion(client.model, noisy_features, device=device)
        recovered_images_dict[eps] = recovered
    
    # 4. Visualize
    try:
        class_names = client.train_data.classes
    except:
        class_names = ["Class 0", "Class 1", "Class 2"] # Fallback
        
    visualize_inversion_results(original_images_np, recovered_images_dict, labels.numpy(), class_names)

# ==========================================
# Main Execution
# ==========================================

if __name__ == "__main__":
    # Ensure directories exist
    import os
    if not os.path.exists('data'):
        print("Note: 'data' directory not found. Ensure datasets are present.")

    # Run Experiments
    try:
        run_noise_vs_accuracy_experiment()
        inversion_attack_experiment()
        print("\nAll experiments completed successfully.")
    except KeyboardInterrupt:
        print("\nExperiments stopped by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()