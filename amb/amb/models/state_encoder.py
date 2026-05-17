"""
State Encoder for AMB
---------------------
3D convolutional encoder that converts voxel grids to feature representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StateEncoder(nn.Module):
    """
    Encodes 3D voxel grids into flat feature vectors.
    
    Uses 3D convolutions to capture spatial patterns in the current build state.
    
    Architecture:
        Input: (B, 1, W, H, D) - one-hot encoded blocks
        -> 3D Conv layers with residual connections
        -> Global average pooling
        -> Output: (B, d_model)
    """
    
    def __init__(
        self,
        num_block_types: int = 6,  # air + 5 roles
        d_model: int = 256,
        max_size: int = 16,
    ):
        """
        Args:
            num_block_types: Number of distinct block types (including air)
            d_model: Output feature dimension
            max_size: Maximum voxel grid size
        """
        super().__init__()
        self.num_block_types = num_block_types
        self.d_model = d_model
        self.max_size = max_size
        
        # Embedding for block types
        self.block_embed = nn.Embedding(num_block_types, 32)
        
        # 3D Convolutional layers
        self.conv1 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm3d(128)
        
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm3d(256)
        
        self.conv4 = nn.Conv3d(256, d_model, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(d_model)
        
        # Global pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool3d(1)
        
        # Final projection
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode a voxel grid.
        
        Args:
            state: (B, W, H, D) tensor of block type IDs
            
        Returns:
            (B, d_model) feature vectors
        """
        B = state.shape[0]
        
        # Embed block types: (B, W, H, D) -> (B, W, H, D, 32)
        x = self.block_embed(state)
        
        # Permute for conv: (B, W, H, D, C) -> (B, C, W, H, D)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        
        # Conv layers
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        x = F.gelu(self.bn4(self.conv4(x)))
        
        # Global pooling: (B, d_model, *, *, *) -> (B, d_model, 1, 1, 1)
        x = self.adaptive_pool(x)
        
        # Flatten and project
        x = x.view(B, self.d_model)
        x = self.proj(x)
        
        return x


class StateEncoderWithPositions(nn.Module):
    """
    Extended state encoder that also outputs per-position features.
    Useful for position prediction head.
    """
    
    def __init__(
        self,
        num_block_types: int = 6,
        d_model: int = 256,
        max_size: int = 16,
    ):
        super().__init__()
        self.num_block_types = num_block_types
        self.d_model = d_model
        self.max_size = max_size
        
        # Embedding for block types
        self.block_embed = nn.Embedding(num_block_types, 32)
        
        # 3D Convolutional layers (no stride to preserve spatial dims)
        self.conv1 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        
        self.conv4 = nn.Conv3d(256, d_model, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(d_model)
        
        # Global features
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.global_proj = nn.Linear(d_model, d_model)
        
    def forward(
        self, 
        state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a voxel grid with position features.
        
        Args:
            state: (B, W, H, D) tensor of block type IDs
            
        Returns:
            global_features: (B, d_model) 
            position_features: (B, d_model, W, H, D)
        """
        B = state.shape[0]
        
        # Embed block types
        x = self.block_embed(state)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        
        # Conv layers
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        x = F.gelu(self.bn4(self.conv4(x)))
        
        # Position features: (B, d_model, W, H, D)
        position_features = x
        
        # Global features
        global_features = self.global_pool(x).view(B, self.d_model)
        global_features = self.global_proj(global_features)
        
        return global_features, position_features
