"""
Visualize generated structures in 3D using matplotlib.

Usage:
    python test/visualize.py --checkpoint checkpoints/best.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from test.model import load_model

# Block colors for visualization
BLOCK_COLORS = {
    1: '#808080',  # WALL - gray
    2: '#8B4513',  # FLOOR - brown
    3: '#B22222',  # ROOF - red
    4: '#00CED1',  # WINDOW - cyan
    5: '#9932CC',  # DOOR - purple
}


def generate_structure(model, device, max_steps=500):
    """Generate structure using model."""
    sz = model.sz
    state = torch.zeros(1, sz, sz, sz, dtype=torch.long, device=device)
    phase = torch.tensor([0], dtype=torch.long, device=device)
    
    model.eval()
    
    for step in range(max_steps):
        prog = torch.tensor([step / max_steps], dtype=torch.float32, device=device)
        
        with torch.no_grad():
            pos_log, blk_log = model(state, phase, prog)
        
        pos_idx = pos_log.argmax().item()
        block = blk_log.argmax().item()
        
        if block == 0:
            break
        
        z = pos_idx % sz
        y = (pos_idx // sz) % sz
        x = pos_idx // (sz ** 2)
        
        if state[0, x, y, z] == 0:
            state[0, x, y, z] = block
    
    return state[0].cpu().numpy()


def visualize_3d(voxels):
    """Create 3D visualization of voxel structure."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("ERROR: matplotlib not installed. Run: pip install matplotlib")
        return
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create colors array
    colors = np.empty(voxels.shape, dtype=object)
    for block_type, color in BLOCK_COLORS.items():
        colors[voxels == block_type] = color
    
    # Plot voxels
    filled = voxels > 0
    ax.voxels(filled, facecolors=colors, edgecolor='k', linewidth=0.1)
    
    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Generated Structure')
    
    # Equal aspect ratio
    max_range = max(voxels.shape)
    ax.set_xlim(0, max_range)
    ax.set_ylim(0, max_range)
    ax.set_zlim(0, max_range)
    
    plt.tight_layout()
    return fig


def visualize_layers(voxels, output_dir=None):
    """Create 2D layer-by-layer visualization."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        print("ERROR: matplotlib not installed")
        return
    
    # Find non-empty layers
    non_empty = []
    for y in range(voxels.shape[1]):
        if np.any(voxels[:, y, :] > 0):
            non_empty.append(y)
    
    if not non_empty:
        print("No blocks to visualize")
        return
    
    # Create colormap
    cmap = ListedColormap(['white', 'gray', 'brown', 'red', 'cyan', 'purple'])
    
    # Plot each layer
    n_layers = len(non_empty)
    cols = min(4, n_layers)
    rows = (n_layers + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    if n_layers == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, y in enumerate(non_empty):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]
        
        layer = voxels[:, y, :].T  # Transpose for correct orientation
        ax.imshow(layer, cmap=cmap, vmin=0, vmax=5, origin='lower')
        ax.set_title(f'Y={y}')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
    
    # Hide unused axes
    for idx in range(n_layers, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis('off')
    
    plt.suptitle('Layer-by-Layer View')
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'layers.png'
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    
    return fig


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    parser = argparse.ArgumentParser(description='Visualize AMB structures')
    parser.add_argument('--checkpoint', '-c', type=str,
                        default='checkpoints/best.pt')
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--output', '-o', type=str, default='test/output',
                        help='Output directory for saved images')
    parser.add_argument('--show', action='store_true',
                        help='Show interactive plot')
    args = parser.parse_args()
    
    # Check checkpoint
    if not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        return 1
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint)
    
    # Generate
    print("Generating structure...")
    voxels = generate_structure(model, 'cpu')
    
    block_count = int((voxels > 0).sum())
    print(f"Generated {block_count} blocks")
    
    # Create output dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize
    try:
        import matplotlib.pyplot as plt
        
        # 3D view
        fig_3d = visualize_3d(voxels)
        if fig_3d:
            fig_3d.savefig(output_dir / 'structure_3d.png', dpi=150)
            print(f"Saved: {output_dir / 'structure_3d.png'}")
        
        # Layer view
        visualize_layers(voxels, output_dir)
        
        if args.show:
            plt.show()
        else:
            plt.close('all')
            
    except Exception as e:
        print(f"Visualization error: {e}")
    
    print("\nDone!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
