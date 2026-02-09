"""
Test script for trained AMB model.
Generates structures and exports to various formats.

Usage:
    python test/test_model.py --checkpoint checkpoints/best.pt
"""

import argparse
import sys
import numpy as np
from pathlib import Path

import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from test.model import SmallModel, load_model


# Role names for display
ROLE_NAMES = ['AIR', 'WALL', 'FLOOR', 'ROOF', 'WINDOW', 'DOOR']
ROLE_COLORS = {
    0: '\033[90m',   # Gray for AIR
    1: '\033[97m',   # White for WALL
    2: '\033[93m',   # Yellow for FLOOR
    3: '\033[91m',   # Red for ROOF
    4: '\033[96m',   # Cyan for WINDOW
    5: '\033[95m',   # Magenta for DOOR
}
RESET = '\033[0m'


def generate_structure(model: SmallModel, device: str, max_steps: int = 500):
    """
    Generate a structure using the trained model.
    
    Returns:
        voxels: numpy array [sz, sz, sz]
        stats: dict with generation stats
    """
    sz = model.sz
    state = torch.zeros(1, sz, sz, sz, dtype=torch.long, device=device)
    phase = torch.tensor([0], dtype=torch.long, device=device)
    
    model.eval()
    placed = 0
    stopped = False
    history = []
    
    print(f"\nGenerating (max {max_steps} steps)...")
    
    for step in range(max_steps):
        prog = torch.tensor([step / max_steps], dtype=torch.float32, device=device)
        
        with torch.no_grad():
            pos_log, blk_log = model(state, phase, prog)
        
        pos_idx = pos_log.argmax().item()
        block = blk_log.argmax().item()
        
        # Decode position
        z = pos_idx % sz
        y = (pos_idx // sz) % sz
        x = pos_idx // (sz ** 2)
        
        if block == 0:  # STOP
            stopped = True
            print(f"  STOP at step {step}")
            break
        
        # Place block
        if state[0, x, y, z] == 0:
            state[0, x, y, z] = block
            placed += 1
            history.append((x, y, z, block))
            
            # Update phase based on position
            if placed % 50 == 0:
                print(f"  Step {step}: placed {placed} blocks")
    
    voxels = state[0].cpu().numpy()
    
    stats = {
        'placed': placed,
        'stopped': stopped,
        'steps': step + 1,
        'non_air': int((voxels > 0).sum()),
    }
    
    return voxels, stats, history


def print_layer(voxels: np.ndarray, y: int):
    """Print a single Y layer of the voxel grid."""
    sz = voxels.shape[0]
    print(f"\n  Layer Y={y}:")
    print("    ", end="")
    for x in range(sz):
        print(f"{x:2}", end="")
    print()
    
    for z in range(sz):
        print(f"  {z:2} ", end="")
        for x in range(sz):
            block = voxels[x, y, z]
            if block == 0:
                print(" .", end="")
            else:
                color = ROLE_COLORS.get(block, '')
                print(f"{color}{block:2}{RESET}", end="")
        print()


def print_structure(voxels: np.ndarray):
    """Print the voxel structure layer by layer."""
    sz = voxels.shape[0]
    
    # Find layers with blocks
    non_empty = []
    for y in range(sz):
        if np.any(voxels[:, y, :] > 0):
            non_empty.append(y)
    
    print(f"\nStructure ({len(non_empty)} non-empty layers):")
    
    for y in non_empty[:5]:  # Show first 5 layers
        print_layer(voxels, y)
    
    if len(non_empty) > 5:
        print(f"\n  ... and {len(non_empty) - 5} more layers")


def count_blocks(voxels: np.ndarray):
    """Count blocks by type."""
    counts = {}
    for role_id in range(6):
        count = int((voxels == role_id).sum())
        if count > 0 or role_id == 0:
            counts[ROLE_NAMES[role_id]] = count
    return counts


def export_to_json(voxels: np.ndarray, output_path: str):
    """Export voxels to JSON format."""
    import json
    
    blocks = []
    for x in range(voxels.shape[0]):
        for y in range(voxels.shape[1]):
            for z in range(voxels.shape[2]):
                if voxels[x, y, z] > 0:
                    blocks.append({
                        'x': int(x),
                        'y': int(y),
                        'z': int(z),
                        'type': int(voxels[x, y, z]),
                        'role': ROLE_NAMES[voxels[x, y, z]]
                    })
    
    data = {
        'size': list(voxels.shape),
        'block_count': len(blocks),
        'blocks': blocks
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nExported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Test trained AMB model')
    parser.add_argument('--checkpoint', '-c', type=str, 
                        default='checkpoints/best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--max-steps', '-s', type=int, default=500,
                        help='Maximum generation steps')
    parser.add_argument('--export', '-e', type=str, default=None,
                        help='Export to JSON file')
    parser.add_argument('--device', '-d', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cpu/cuda)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("AMB Model Test")
    print("=" * 60)
    
    # Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("\nMake sure to download best.pt from Kaggle and place it in:")
        print(f"  {checkpoint_path.absolute()}")
        return 1
    
    print(f"\nLoading model from: {checkpoint_path}")
    model = load_model(str(checkpoint_path), args.device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")
    print(f"Device: {args.device}")
    
    # Generate
    voxels, stats, history = generate_structure(model, args.device, args.max_steps)
    
    # Stats
    print("\n" + "=" * 60)
    print("Generation Results")
    print("=" * 60)
    print(f"  Blocks placed: {stats['placed']}")
    print(f"  Stopped naturally: {stats['stopped']}")
    print(f"  Total steps: {stats['steps']}")
    
    # Block counts
    counts = count_blocks(voxels)
    print("\nBlock counts:")
    for role, count in counts.items():
        if role != 'AIR' and count > 0:
            print(f"  {role}: {count}")
    
    # Print structure
    print_structure(voxels)
    
    # Export
    if args.export:
        export_to_json(voxels, args.export)
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
