"""
Builder Transformer for AMB
---------------------------
Main model that predicts sequential build actions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from .state_encoder import StateEncoder, StateEncoderWithPositions
from .action_heads import PositionHead, BlockTypeHead, CombinedActionHead


class PhaseProgressEncoder(nn.Module):
    """Encodes build phase and progress into features."""
    
    def __init__(self, num_phases: int = 5, d_model: int = 256):
        super().__init__()
        self.num_phases = num_phases
        
        # Phase embedding
        self.phase_embed = nn.Embedding(num_phases, d_model // 2)
        
        # Progress encoding (continuous)
        self.progress_mlp = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model // 2),
        )
        
        # Combine
        self.combine = nn.Linear(d_model, d_model)
    
    def forward(
        self, 
        phase: torch.Tensor,      # (B,) int
        progress: torch.Tensor    # (B,) float
    ) -> torch.Tensor:
        """
        Encode phase and progress.
        
        Returns:
            (B, d_model) conditioning features
        """
        phase_feat = self.phase_embed(phase)
        progress_feat = self.progress_mlp(progress.unsqueeze(-1))
        
        combined = torch.cat([phase_feat, progress_feat], dim=-1)
        return self.combine(combined)


class TransformerBlock(nn.Module):
    """Standard transformer decoder block."""
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model) input features
            condition: (B, d_model) optional conditioning
        """
        # Add conditioning
        if condition is not None:
            x = x + condition.unsqueeze(1)
        
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN
        x = self.norm2(x + self.ffn(x))
        
        return x


class BuilderTransformer(nn.Module):
    """
    Main model for sequential Minecraft building.
    
    Architecture:
        1. StateEncoder: 3D conv to encode current voxel grid
        2. PhaseProgressEncoder: Encode build phase and progress
        3. TransformerBlocks: Process combined features
        4. ActionHeads: Predict position and block type
    
    Usage:
        model = BuilderTransformer()
        pos_logits, block_logits = model(state, phase, progress)
    """
    
    def __init__(
        self,
        max_size: int = 16,
        num_block_types: int = 6,  # STOP + 5 roles
        num_phases: int = 5,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ):
        """
        Args:
            max_size: Maximum voxel grid size
            num_block_types: Number of block types (including STOP)
            num_phases: Number of build phases
            d_model: Model dimension
            n_heads: Attention heads
            n_layers: Transformer layers
            d_ff: FFN hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.max_size = max_size
        self.num_block_types = num_block_types
        self.d_model = d_model
        
        # Encoders
        self.state_encoder = StateEncoder(
            num_block_types=num_block_types,
            d_model=d_model,
            max_size=max_size,
        )
        
        self.phase_progress_encoder = PhaseProgressEncoder(
            num_phases=num_phases,
            d_model=d_model,
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output heads
        self.action_head = CombinedActionHead(
            d_model=d_model,
            max_size=max_size,
            num_block_types=num_block_types,
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self,
        state: torch.Tensor,           # (B, W, H, D) block type IDs
        phase: torch.Tensor,           # (B,) phase IDs
        progress: torch.Tensor,        # (B,) progress values 0-1
        target_pos: torch.Tensor = None,  # (B, 3) for teacher forcing
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: Current voxel grid (B, W, H, D)
            phase: Build phase (B,)
            progress: Build progress (B,)
            target_pos: Target position for teacher forcing (B, 3)
            
        Returns:
            position_logits: (B, W*H*D)
            block_logits: (B, num_block_types)
        """
        # Encode state
        state_feat = self.state_encoder(state)  # (B, d_model)
        
        # Encode phase and progress
        cond_feat = self.phase_progress_encoder(phase, progress)  # (B, d_model)
        
        # Combine features (as single token for transformer)
        x = state_feat + cond_feat
        x = x.unsqueeze(1)  # (B, 1, d_model)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, cond_feat)
        
        # Get output features
        x = x.squeeze(1)  # (B, d_model)
        
        # Predict action
        position_logits, block_logits = self.action_head(x, target_pos)
        
        return position_logits, block_logits
    
    def predict_action(
        self,
        state: torch.Tensor,
        phase: torch.Tensor,
        progress: torch.Tensor,
        temperature: float = 1.0,
        sample: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict next action for inference.
        
        Returns:
            x, y, z: Position coordinates (B,)
            block_type: Block type (B,)
        """
        position_logits, block_logits = self(state, phase, progress)
        
        # Sample position
        x, y, z = self.action_head.position_head.get_position(
            position_logits, temperature=temperature, sample=sample
        )
        
        # Sample block type
        block_type = self.action_head.block_head(
            torch.cat([
                self.state_encoder(state),
                self.action_head.pos_embed(
                    torch.stack([x, y, z], dim=-1).float() / self.max_size
                )
            ], dim=-1)
        )
        block_type = block_logits.argmax(dim=-1) if not sample else \
            torch.multinomial(F.softmax(block_logits / temperature, dim=-1), 1).squeeze(-1)
        
        return x, y, z, block_type


class BuilderTransformerSmall(BuilderTransformer):
    """Smaller version for testing and fast experiments."""
    
    def __init__(self, max_size: int = 16, **kwargs):
        super().__init__(
            max_size=max_size,
            d_model=128,
            n_heads=4,
            n_layers=4,
            d_ff=512,
            **kwargs
        )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    model = BuilderTransformer(max_size=16)
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test forward
    B = 4
    state = torch.randint(0, 6, (B, 16, 16, 16))
    phase = torch.randint(0, 5, (B,))
    progress = torch.rand(B)
    
    pos_logits, block_logits = model(state, phase, progress)
    print(f"Position logits: {pos_logits.shape}")
    print(f"Block logits: {block_logits.shape}")
