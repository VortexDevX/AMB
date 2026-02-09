"""Test all stage checkpoints"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from test.model import SmallModel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_SIZE = 16
ROLE_NAMES = ['STOP', 'WALL', 'FLOOR', 'ROOF', 'WINDOW', 'DOOR']

def generate(model, max_steps=300):
    model.eval()
    state = torch.zeros(1, MAX_SIZE, MAX_SIZE, MAX_SIZE, dtype=torch.long, device=DEVICE)
    phase = torch.tensor([0], dtype=torch.long, device=DEVICE)
    
    blocks = {}
    for step in range(max_steps):
        prog = torch.tensor([step/max_steps], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            pos_log, blk_log = model(state, phase, prog)
        
        pos_idx = pos_log.argmax().item()
        blk = blk_log.argmax().item()
        
        if blk == 0:
            return blocks, step, True
        
        z = pos_idx % MAX_SIZE
        y = (pos_idx // MAX_SIZE) % MAX_SIZE
        x = pos_idx // (MAX_SIZE ** 2)
        
        if state[0,x,y,z] == 0:
            state[0,x,y,z] = blk
            blocks[blk] = blocks.get(blk, 0) + 1
            
            if y > 8: phase[0] = 2
            elif y > 2: phase[0] = 1
    
    return blocks, max_steps, False

def test_checkpoint(path):
    print(f"\n{'='*50}")
    print(f"Testing: {path}")
    print('='*50)
    
    model = SmallModel(MAX_SIZE, 128).to(DEVICE)
    try:
        state_dict = torch.load(path, map_location=DEVICE)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"ERROR loading: {e}")
        return
    
    blocks, steps, stopped = generate(model)
    total = sum(blocks.values())
    
    print(f"Generated {total} blocks in {steps} steps")
    print(f"Stopped naturally: {stopped}")
    print("Block distribution:")
    for b, c in sorted(blocks.items()):
        print(f"  {ROLE_NAMES[b]}: {c}")

if __name__ == '__main__':
    checkpoints = [
        'checkpoints/stage1.pt',
        'checkpoints/stage2.pt', 
        'checkpoints/stage3.pt',
        'checkpoints/stage4.pt',
        'checkpoints/best.pt',
    ]
    
    for cp in checkpoints:
        if Path(cp).exists():
            test_checkpoint(cp)
        else:
            print(f"Not found: {cp}")
