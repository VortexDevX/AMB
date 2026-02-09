# AMB Test Suite

Test scripts for the trained AMB model.

## Setup

1. Download `best.pt` from Kaggle
2. Place it in `checkpoints/best.pt`

## Scripts

### test_model.py

Generate structures and print stats.

```bash
python test/test_model.py --checkpoint checkpoints/best.pt
```

Options:

- `--max-steps`: Maximum generation steps (default 500)
- `--export FILE`: Export to JSON
- `--device`: cpu or cuda

### visualize.py

Create 3D visualizations (requires matplotlib).

```bash
pip install matplotlib
python test/visualize.py --checkpoint checkpoints/best.pt --show
```

## Block Types

| ID  | Role   | Color  |
| --- | ------ | ------ |
| 0   | AIR    | -      |
| 1   | WALL   | Gray   |
| 2   | FLOOR  | Brown  |
| 3   | ROOF   | Red    |
| 4   | WINDOW | Cyan   |
| 5   | DOOR   | Purple |
