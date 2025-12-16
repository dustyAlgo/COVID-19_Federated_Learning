import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from models import LightweightCOVIDNet, ResNet18COVID, VGG11COVID # Import your models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- 1. Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "client_0_final.pth"  # <--- Point this to your saved file
MODEL_TYPE = "LightweightCOVIDNet"      # Change to "ResNet18COVID" etc. if needed

print(f"Loading model from {CHECKPOINT_PATH} on {DEVICE}")

# --- 2. Setup Data (Just for input samples) ---
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# We need some images to attack
try:
    if not os.path.exists('data'): raise FileNotFoundError
    dataset = datasets.ImageFolder(root='data/Hospital_A/', transform=transform)
except:
    print("Warning: Using Dummy Data")
    dataset = datasets.FakeData(size=20, transform=transform)

# --- 3. Load the Saved Model ---
# Initialize the empty architecture first
if MODEL_TYPE == "LightweightCOVIDNet":
    model = LightweightCOVIDNet()
elif MODEL_TYPE == "ResNet18COVID":
    model = ResNet18COVID()
elif MODEL_TYPE == "VGG11COVID":
    model = VGG11COVID()

model.to(DEVICE)

# Load weights
try:
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"Error: Could not find {CHECKPOINT_PATH}. Did you run the training script first?")
    exit()

# --- 4. The Attack Logic (Same as before) ---
def simple_feature_inversion(model, target_features, steps=500):
    model.eval()
    dummy_input = torch.randn(target_features.shape[0], 1, 128, 128, requires_grad=True, device=DEVICE)
    optimizer = torch.optim.Adam([dummy_input], lr=0.1)
    target = torch.tensor(target_features).to(DEVICE)
    
    print(f"   > Reconstructing ({steps} steps)...")
    for _ in range(steps):
        optimizer.zero_grad()
        feat = model.extract_features(dummy_input)
        loss = ((feat - target)**2).mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            dummy_input.data.clamp_(0, 1)
    return dummy_input.detach().cpu().numpy()

# --- 5. Run Experiment ---
loader = DataLoader(dataset, batch_size=5, shuffle=True)
images, _ = next(iter(loader))
images = images.to(DEVICE)

# Get Ground Truth
with torch.no_grad():
    clean_features = model.extract_features(images)

privacy_levels = [0.1, 1.0, 10.0]
recovered_dict = {}

for eps in privacy_levels:
    print(f"Attacking ε={eps}...")
    
    # Noise Logic
    k = int(0.1 * clean_features.shape[1])
    topk_indices = torch.topk(clean_features.abs(), k, dim=1).indices
    sparse = torch.zeros_like(clean_features)
    for i in range(clean_features.shape[0]):
        sparse[i, topk_indices[i]] = clean_features[i, topk_indices[i]]
        
    noise = torch.normal(0, 1.0/eps, size=sparse.shape).to(DEVICE)
    noisy = sparse + noise
    
    # Invert
    recovered_dict[eps] = simple_feature_inversion(model, noisy)

# --- 6. Plot ---
plt.figure(figsize=(12, 6))
num_cols = 1 + len(privacy_levels)
for i in range(len(images)):
    # Original
    plt.subplot(len(images), num_cols, i * num_cols + 1)
    plt.imshow(images[i].cpu().squeeze(), cmap='gray')
    if i==0: plt.title("Original")
    plt.axis('off')
    
    # Recovered
    for j, eps in enumerate(sorted(privacy_levels)):
        plt.subplot(len(images), num_cols, i * num_cols + j + 2)
        plt.imshow(recovered_dict[eps][i].squeeze(), cmap='gray')
        if i==0: plt.title(f"ε={eps}")
        plt.axis('off')

plt.tight_layout()
plt.savefig('saved_model_attack.png')
print("Saved saved_model_attack.png")