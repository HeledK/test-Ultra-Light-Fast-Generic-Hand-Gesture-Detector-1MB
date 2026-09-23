import torch
import torch.nn as nn
import torch_pruning as tp

# =========================
# 1. Load your model
# =========================
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd  

model = create_mb_tiny_fd(num_classes=3, is_test=False)
checkpoint = torch.load("slim-Epoch-19-Loss-1.3659555289149283.pth", map_location="cpu")
model.load_state_dict(checkpoint)
model.eval()

# =========================
# 2. Define example input
# =========================
example_inputs = torch.randn(1, 3, 128, 128)

# =========================
# 3. Ignore detection heads
# =========================
ignored_layers = []

# ignore all regression and classification headers
for header in model.regression_headers + model.classification_headers:
    for m in header.modules():
        if isinstance(m, nn.Conv2d):
            ignored_layers.append(m)

print(f"Ignoring {len(ignored_layers)} detection head conv layers")

# =========================
# 4. Define importance
# =========================
imp = tp.importance.MagnitudeImportance(p=1)  # L1 norm

# =========================
# 5. Create pruner
# =========================
pruner = tp.pruner.MetaPruner(
    model,
    example_inputs,
    importance=imp,
    pruning_ratio=0.15,   
    ignored_layers=ignored_layers,
)

# =========================
# 6. Apply pruning
# =========================
pruner.step()

print("Pruning done!")

# =========================
# 7. Save pruned model
# =========================
torch.save(model.state_dict(), "pruned_model.pth")