import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

def get_resnet(depth, num_classes=10):
    """Create ResNet with CIFAR-10 adaptations."""
    
    if depth == 18:
        model = resnet18(weights=None)
    elif depth == 34:
        model = resnet34(weights=None)
    elif depth == 50:
        model = resnet50(weights=None)
    elif depth == 101:
        model = resnet101(weights=None)
    elif depth == 152:
        model = resnet152(weights=None)
    else:
        raise ValueError(f"Unsupported depth: {depth}")
    
    # Adapt for CIFAR-10 (32x32 images)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
