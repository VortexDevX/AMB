"""
Action Heads for AMB
--------------------
Output heads for position and block type prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionHead(nn.Module):
    """
    Predicts WHERE to place the next block.
    
    Outputs a probability distribution over all positions in the voxel grid.
    Uses the global features + position features from state encoder.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        max_size: int = 16,
        hidden_dim: int = 512,
    ):
        """
        Args:
            d_model: Input feature dimension
            max_size: Grid size (assumes cubic)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.d_model = d_model
        self.max_size = max_size
        self.num_positions = max_size ** 3
        
        # MLP for position prediction
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.num_positions),
        )
    
    def forward(
        self, 
        features: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Predict position distribution.
        
        Args:
            features: (B, d_model) global features
            mask: (B, W*H*D) optional mask for invalid positions
            
        Returns:
            (B, W*H*D) logits over positions
        """
        logits = self.mlp(features)
        
        if mask is not None:
            # Mask out invalid positions with large negative value
            logits = logits.masked_fill(~mask, -1e9)
        
        return logits
    
    def get_position(
        self, 
        logits: torch.Tensor, 
        temperature: float = 1.0,
        sample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert logits to (x, y, z) coordinates.
        
        Args:
            logits: (B, W*H*D) position logits
            temperature: Sampling temperature
            sample: If True, sample; else argmax
            
        Returns:
            x, y, z: each (B,) tensors of coordinates
        """
        B = logits.shape[0]
        
        if sample:
            probs = F.softmax(logits / temperature, dim=-1)
            idx = torch.multinomial(probs, 1).squeeze(-1)
        else:
            idx = logits.argmax(dim=-1)
        
        # Convert flat index to (x, y, z)
        z = idx % self.max_size
        y = (idx // self.max_size) % self.max_size
        x = idx // (self.max_size ** 2)
        
        return x, y, z


class BlockTypeHead(nn.Module):
    """
    Predicts WHAT block to place (including STOP).
    
    Block types:
        0 = STOP (terminate generation)
        1-5 = Role IDs (wall, floor, roof, window, door)
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_block_types: int = 6,  # STOP + 5 roles
        hidden_dim: int = 256,
    ):
        """
        Args:
            d_model: Input feature dimension
            num_block_types: Number of block types (including STOP)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.num_block_types = num_block_types
        
        # MLP for block type prediction
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_block_types),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict block type distribution.
        
        Args:
            features: (B, d_model) global features
            
        Returns:
            (B, num_block_types) logits
        """
        return self.mlp(features)
    
    def get_block_type(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        sample: bool = True
    ) -> torch.Tensor:
        """
        Convert logits to block type.
        
        Args:
            logits: (B, num_block_types) logits
            temperature: Sampling temperature
            sample: If True, sample; else argmax
            
        Returns:
            (B,) block type indices
        """
        if sample:
            probs = F.softmax(logits / temperature, dim=-1)
            return torch.multinomial(probs, 1).squeeze(-1)
        else:
            return logits.argmax(dim=-1)


class CombinedActionHead(nn.Module):
    """
    Combined head that predicts position and block type together.
    Uses position-conditioned block prediction.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        max_size: int = 16,
        num_block_types: int = 6,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.max_size = max_size
        self.num_block_types = num_block_types
        
        # Position prediction
        self.position_head = PositionHead(d_model, max_size, hidden_dim)
        
        # Block type prediction (takes global features + position embedding)
        self.pos_embed = nn.Linear(3, 64)
        self.block_head = nn.Sequential(
            nn.Linear(d_model + 64, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_block_types),
        )
    
    def forward(
        self,
        features: torch.Tensor,
        target_pos: torch.Tensor = None,  # (B, 3) for teacher forcing
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict action (position, block_type).
        
        Args:
            features: (B, d_model) global features
            target_pos: (B, 3) target position for teacher forcing
            
        Returns:
            position_logits: (B, W*H*D)
            block_logits: (B, num_block_types)
        """
        B = features.shape[0]
        device = features.device
        
        # Position prediction
        position_logits = self.position_head(features)
        
        # Get predicted position for block prediction
        if target_pos is not None:
            # Teacher forcing: use ground truth position
            pos = target_pos.float()
        else:
            # Use predicted position
            x, y, z = self.position_head.get_position(position_logits, sample=False)
            pos = torch.stack([x, y, z], dim=-1).float()
        
        # Normalize position to [0, 1]
        pos = pos / self.max_size
        
        # Block type prediction
        pos_features = self.pos_embed(pos)
        combined = torch.cat([features, pos_features], dim=-1)
        block_logits = self.block_head(combined)
        
        return position_logits, block_logits
