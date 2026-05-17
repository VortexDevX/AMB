"""
Loss Functions for AMB
----------------------
Combined loss for position and block type prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AMBLoss(nn.Module):
    """
    Combined loss for AMB training.
    
    L_total = L_position + L_block
    
    With optional weighting for STOP token.
    """
    
    def __init__(
        self,
        max_size: int = 16,
        num_block_types: int = 6,
        stop_weight: float = 50.0,  # Higher weight for STOP token
        position_weight: float = 1.0,
        block_weight: float = 1.0,
    ):
        """
        Args:
            max_size: Voxel grid size
            num_block_types: Number of block types
            stop_weight: Extra weight for STOP token (class 0)
            position_weight: Weight for position loss
            block_weight: Weight for block type loss
        """
        super().__init__()
        
        self.max_size = max_size
        self.num_positions = max_size ** 3
        self.num_block_types = num_block_types
        self.position_weight = position_weight
        self.block_weight = block_weight
        
        # Block class weights (emphasize STOP)
        block_weights = torch.ones(num_block_types)
        block_weights[0] = stop_weight  # STOP token
        self.register_buffer('block_class_weights', block_weights)
        
        # Position loss (cross entropy)
        self.position_loss_fn = nn.CrossEntropyLoss()
        
        # Block loss (weighted cross entropy)
        self.block_loss_fn = nn.CrossEntropyLoss(weight=block_weights)
    
    def forward(
        self,
        position_logits: torch.Tensor,  # (B, W*H*D)
        block_logits: torch.Tensor,     # (B, num_block_types)
        target_x: torch.Tensor,         # (B,)
        target_y: torch.Tensor,         # (B,)
        target_z: torch.Tensor,         # (B,)
        target_block: torch.Tensor,     # (B,)
    ) -> dict:
        """
        Compute losses.
        
        Returns:
            Dict with 'total', 'position', 'block' losses
        """
        B = position_logits.shape[0]
        
        # Convert (x, y, z) to flat index
        target_pos_idx = (
            target_x * self.max_size * self.max_size +
            target_y * self.max_size +
            target_z
        )
        
        # Position loss
        loss_position = self.position_loss_fn(position_logits, target_pos_idx)
        
        # Block loss
        loss_block = self.block_loss_fn(block_logits, target_block)
        
        # Total
        loss_total = (
            self.position_weight * loss_position +
            self.block_weight * loss_block
        )
        
        return {
            'total': loss_total,
            'position': loss_position,
            'block': loss_block,
        }


class PositionAccuracy:
    """Compute position prediction accuracy."""
    
    def __init__(self, max_size: int = 16):
        self.max_size = max_size
    
    def __call__(
        self,
        position_logits: torch.Tensor,
        target_x: torch.Tensor,
        target_y: torch.Tensor,
        target_z: torch.Tensor,
    ) -> float:
        pred_idx = position_logits.argmax(dim=-1)
        
        target_idx = (
            target_x * self.max_size * self.max_size +
            target_y * self.max_size +
            target_z
        )
        
        correct = (pred_idx == target_idx).float().mean()
        return correct.item()


class BlockAccuracy:
    """Compute block type prediction accuracy."""
    
    def __call__(
        self,
        block_logits: torch.Tensor,
        target_block: torch.Tensor,
    ) -> float:
        pred = block_logits.argmax(dim=-1)
        correct = (pred == target_block).float().mean()
        return correct.item()


class StopAccuracy:
    """Compute STOP token accuracy (precision and recall)."""
    
    def __call__(
        self,
        block_logits: torch.Tensor,
        target_block: torch.Tensor,
    ) -> dict:
        pred = block_logits.argmax(dim=-1)
        
        # STOP = class 0
        pred_stop = (pred == 0)
        true_stop = (target_block == 0)
        
        # Precision: of predicted STOPs, how many are correct
        if pred_stop.sum() > 0:
            precision = (pred_stop & true_stop).float().sum() / pred_stop.float().sum()
        else:
            precision = torch.tensor(0.0)
        
        # Recall: of true STOPs, how many were predicted
        if true_stop.sum() > 0:
            recall = (pred_stop & true_stop).float().sum() / true_stop.float().sum()
        else:
            recall = torch.tensor(1.0)  # No STOPs to miss
        
        return {
            'stop_precision': precision.item(),
            'stop_recall': recall.item(),
        }
