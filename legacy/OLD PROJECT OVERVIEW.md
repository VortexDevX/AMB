# Minecraft AI Builder - Project Overview

## What We Built

An AI-powered Minecraft structure generator that creates buildings based on text prompts. The system uses a 3D U-Net neural network trained on procedurally generated synthetic structures to generate voxel-based buildings.

---

## Current Status (Jan 2026)

### ML Training Approach: Per-Type Models

After extensive experimentation, we've adopted a **single-type training strategy**:

- Each structure type (house, tower, castle, etc.) has its own dedicated model
- No conditioning complexity - models take only dimensions as input
- Trained on 10K synthetic samples per type with Dice+Focal loss

**Models in Training:**
| Model | Structure | Status |
|-------|-----------|--------|
| `house_model.pth` | 🏠 Houses | Training |
| `tower_model.pth` | 🗼 Towers | Training |
| More to come... | | Planned |

---

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Frontend   │────>│   Backend   │────>│  Minecraft  │
│   (React)   │     │  (FastAPI)  │     │   Agent     │
└─────────────┘     └─────────────┘     └─────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │  ML Models  │
                    │ (Per-Type)  │
                    └─────────────┘
```

### ML Model: SimpleUNet

- **Input:** Dimensions (width, height, depth)
- **Output:** 16x16x16 voxel grid with 6 role classes
- **Loss:** Dice + Focal (balanced for rare classes)

---

## Role Classes (Voxel Types)

| ID  | Role   | Description          |
| --- | ------ | -------------------- |
| 0   | Air    | Empty space          |
| 1   | Wall   | Solid walls          |
| 2   | Floor  | Ground level         |
| 3   | Roof   | Top covering         |
| 4   | Window | Transparent openings |
| 5   | Door   | Entry points         |

---

## Project Structure

```
minecraft-ai-builder/
├── backend/           # FastAPI server
├── frontend/          # React UI
├── minecraft-agent/   # Minecraft bot integration
├── ml/
│   ├── models/        # Model definitions
│   ├── training/      # Training notebooks & scripts
│   │   ├── house_training.ipynb
│   │   ├── tower_training.ipynb
│   │   └── train.py
│   ├── checkpoints/   # Saved model weights
│   ├── data/          # Dataset loaders
│   ├── inference/     # Inference utilities
│   ├── archive/       # Archived experiments
│   └── test_model.py  # Model testing script
└── datasets/          # Training data utilities
```

---

## Training Configuration

| Parameter         | Value               |
| ----------------- | ------------------- |
| Samples           | 10,000 per type     |
| Epochs            | 80                  |
| Batch Size        | 32                  |
| Learning Rate     | 1e-3 (cosine decay) |
| Max Voxel Size    | 16x16x16            |
| Loss              | Dice + Focal        |
| Window/Door Boost | 10x weight          |

---

## Testing Models

```bash
cd ml
py test_model.py house checkpoints/house_model.pth
py test_model.py tower checkpoints/tower_model.pth
py test_model.py all checkpoints/best_model.pth  # Test all types
```

---

## Performance Targets

| Metric     | Target | Current |
| ---------- | ------ | ------- |
| mIoU       | >40%   | TBD     |
| Window IoU | >10%   | TBD     |
| Door IoU   | >5%    | TBD     |

---

## History

### Phase 1: Real Data (Failed)

- Attempted training on ~1700 real schematics
- Problem: Extreme class imbalance (99% Air/Wall)
- Result: Model collapsed to trivial solutions

### Phase 2: Multi-Type Synthetic (Partial Success)

- Generated 16 structure types synthetically
- Problem: Conditioning mechanism broken
- Result: Some types work (Church), others collapse (House)

### Phase 3: Per-Type Models (Current)

- Separate model per structure type
- No conditioning, simpler learning task
- Uses Dice+Focal loss for class balance
