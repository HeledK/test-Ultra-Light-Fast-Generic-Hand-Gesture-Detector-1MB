import torch

model = torch.load(r"C:\Users\Heled\Ultra-Light-Fast-Generic-Face-Detector-1MB\models\train-version-slim\slim-Epoch-39-Loss-0.694770356442066\slim-Epoch-39-Loss-0.694770356442066.pth", map_location="cpu")

# If it's a state dict (most common)
if isinstance(model, dict):
    for key, tensor in model.items():
        print(f"{key:60s} {str(tensor.shape):30s} {tensor.dtype}")

# If it's a full model object
else:
    print(model)