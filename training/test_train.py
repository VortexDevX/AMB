import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from amb.data import AMBDataset
from amb.models import BuilderTransformer
from amb.training.train import train

dataset = AMBDataset(
    schematic_dir=str(project_root / "datasets" / "organized" / "house"),
    max_size=16,
    max_structures=50,  # Need more attempts since many are skipped
    augment_rotations=False,
    cache_samples=True,
)

print(f"Samples: {len(dataset)}")
# Should be ~100-500 depending on structure size

# Create small model for fast testing
model = BuilderTransformer(
    max_size=16,
    d_model=128,
    n_layers=4,
)

# Train
train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

train(
    model=model,
    train_loader=train_loader,
    val_loader=None,
    epochs=100,  # Overfit
    lr=1e-3,
    checkpoint_dir=str(project_root / "checkpoints" / "overfit_test"),
    device="cuda",
)
