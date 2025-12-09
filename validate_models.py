# validate_models.py
import torch
from models import LightweightCOVIDNet, ResNet18COVID, VGG11COVID

# Test each model with a sample input
test_input = torch.randn(2, 1, 128, 128)  # batch_size=2, grayscale 128x128

models = [LightweightCOVIDNet(), ResNet18COVID(), VGG11COVID()]
model_names = ["Custom CNN", "ResNet18", "VGG11"]

print("🔍 Validating model feature dimensions:")
for i, (model, name) in enumerate(zip(models, model_names)):
    features = model.extract_features(test_input)
    print(f"{name}: {features.shape}")
    
    # Should be torch.Size([2, 256]) for all models
    if features.shape[1] != 256:
        print(f"ERROR: {name} outputs {features.shape[1]} features, expected 256!")
    else:
        print(f"{name}: Correct 256 features!")