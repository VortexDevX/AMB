"""
AMB Model Architecture - For local testing
Must match the architecture used during training on Kaggle
"""

import torch
import torch.nn as nn


class SmallModel(nn.Module):
    """
    Small AMB model matching Kaggle training architecture.
    
    Args:
        sz: Grid size (default 16)
        d: Model dimension (default 128)
    """
    
    def __init__(self, sz: int = 16, d: int = 128):
        super().__init__()
        self.sz = sz
        self.d_model = d
        
        # Block embedding
        self.embed = nn.Embedding(6, 16)
        
        # 3D ConvNet encoder
        self.conv = nn.Sequential(
            nn.Conv3d(16, 32, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv3d(64, d, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        
        # Conditioning
        self.phase_emb = nn.Embedding(5, d // 2)
        self.prog_fc = nn.Linear(1, d // 2)
        self.fc = nn.Linear(d * 2, d)
        
        # Output heads
        self.pos_head = nn.Linear(d, sz ** 3)
        self.blk_head = nn.Linear(d, 6)
    
    def forward(self, state, phase, prog):
        """
        Forward pass.
        
        Args:
            state: Voxel grid [B, sz, sz, sz] (Long)
            phase: Build phase [B] (Long, 0-4)
            prog: Build progress [B] (Float, 0-1)
            
        Returns:
            pos_logits: Position logits [B, sz^3]
            blk_logits: Block type logits [B, 6]
        """
        # Encode state
        x = self.embed(state).permute(0, 4, 1, 2, 3).float()
        x = self.conv(x).flatten(1)
        
        # Conditioning
        p = self.phase_emb(phase)
        g = self.prog_fc(prog.unsqueeze(-1))
        x = self.fc(torch.cat([x, p, g], -1))
        
        return self.pos_head(x), self.blk_head(x)
    
    def predict_action(self, state, phase, prog):
        """
        Predict next action.
        
        Returns:
            (x, y, z, block_type) tuple
        """
        self.eval()
        with torch.no_grad():
            pos_log, blk_log = self(state, phase, prog)
            
            pos_idx = pos_log.argmax(dim=-1).item()
            block = blk_log.argmax(dim=-1).item()
            
            z = pos_idx % self.sz
            y = (pos_idx // self.sz) % self.sz
            x = pos_idx // (self.sz ** 2)
            
            return x, y, z, block


def load_model(checkpoint_path: str, device: str = 'cpu') -> SmallModel:
    """Load trained model from checkpoint."""
    model = SmallModel(sz=16, d=128)
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Handle both direct state dict and wrapped checkpoint
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
