import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

class CheXNetModel(nn.Module):
    """
    Standard CheXNet Architecture: 
    DenseNet121 backbone with a Linear classifier for multi-label pathology detection.
    """
    def __init__(self, num_classes=15):
        super(CheXNetModel, self).__init__()
        # Load DenseNet121 with ImageNet weights
        self.densenet121 = models.densenet121(weights='IMAGENET1K_V1')
        
        # Replace the final linear layer (classifier) with 15 target pathologies
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.densenet121(x)

def get_chexnet_pipeline():
    # Pipeline steps documentation for the professor
    pipeline = {
        "1. Image Preprocessing": "Standard normalization (ImageNet stats), Resize to 224x224.",
        "2. Backbone": "DenseNet-121 (Effective for deep feature reuse).",
        "3. Global Pooling": "Global Average Pooling before the classification head.",
        "4. Output": "15 nodes with Sigmoid activation (for Multi-label Classification).",
        "5. Loss Function": "BCEWithLogitsLoss."
    }
    return pipeline

if __name__ == "__main__":
    model = CheXNetModel()
    print("CheXNet Model Structure Initialized.")
    for step, desc in get_chexnet_pipeline().items():
        print(f"{step}: {desc}")
