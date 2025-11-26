# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, vgg11

class LightweightCOVIDNet(nn.Module):
    def __init__(self, feature_dim=256, num_classes=3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # for CT/X-ray
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Linear(64, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def extract_features(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def forward(self, x):
        f = self.extract_features(x)
        out = self.classifier(f)
        return out
# Modified ResNet18 for single-channel input
class ResNet18COVID(nn.Module):
    def __init__(self, feature_dim=256, num_classes=3):
        super().__init__()
        # Load pre-trained ResNet18
        self.model = resnet18(pretrained=True)
        
        # Modify first layer for single channel input
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace final layer for feature extraction
        self.model.fc = nn.Linear(512, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def extract_features(self, x):
        x = self.model(x)
        return x

    def forward(self, x):
        f = self.extract_features(x)
        out = self.classifier(f)
        return out

# Modified VGG11 for single-channel input  
class VGG11COVID(nn.Module):
    def __init__(self, feature_dim=256, num_classes=3):
        super().__init__()
        # Load pre-trained VGG11
        self.model = vgg11(pretrained=True)
        
        # Modify first layer for single channel input
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        
        # Replace classifier for feature extraction
        self.model.classifier[6] = nn.Linear(4096, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def extract_features(self, x):
        x = self.model(x)
        return x

    def forward(self, x):
        f = self.extract_features(x)
        out = self.classifier(f)
        return out