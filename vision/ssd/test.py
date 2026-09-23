import torch
from .mb_tiny_fd import create_mb_tiny_fd

# Set number of classes (including background)
num_classes = 3  # e.g., BACKGROUND, palm, one

# Create network
net = create_mb_tiny_fd(num_classes=num_classes, is_test=True, device="cpu")

# Dummy input (batch_size=1, 3 channels, 320x320 image)
x = torch.randn(1, 3, 320, 320)

# Forward pass
with torch.no_grad():
    conf, loc = net(x)

# Check shapes
print("Classification output shape:", conf.shape)  # [batch_size, num_anchors, num_classes]
print("Regression output shape:", loc.shape)       # [batch_size, num_anchors, 4]
