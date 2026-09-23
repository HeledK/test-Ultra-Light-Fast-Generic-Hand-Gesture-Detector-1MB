import torch

obj = torch.load(r"C:\Users\Heled\Ultra-Light-Fast-Generic-Face-Detector-1MB\models\train-version-slim\slim-Epoch-39-Loss-0.694770356442066\slim-Epoch-39-Loss-0.694770356442066.pth", map_location="cpu")

print("=== TYPE ===")
print(type(obj))

print("\n=== KEYS (top level) ===")
if isinstance(obj, dict):
    for k in obj.keys():
        print(f"  {k}")
else:
    print("  Not a dict — full model object")