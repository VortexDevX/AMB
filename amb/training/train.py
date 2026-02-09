"""
Training Loop for AMB
---------------------
Training and validation functions.
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .losses import AMBLoss, PositionAccuracy, BlockAccuracy, StopAccuracy


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: AMBLoss,
    device: torch.device,
    epoch: int = 0,
    log_interval: int = 100,
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: BuilderTransformer model
        dataloader: Training dataloader
        optimizer: Optimizer
        loss_fn: AMBLoss instance
        device: Device to train on
        epoch: Current epoch number
        log_interval: Log every N batches
        
    Returns:
        Dict with average losses and metrics
    """
    model.train()
    
    total_loss = 0.0
    total_pos_loss = 0.0
    total_block_loss = 0.0
    total_pos_acc = 0.0
    total_block_acc = 0.0
    num_batches = 0
    
    pos_acc_fn = PositionAccuracy(loss_fn.max_size)
    block_acc_fn = BlockAccuracy()
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        state = batch['state'].to(device)
        phase = batch['phase'].to(device)
        progress = batch['progress'].to(device)
        target_x = batch['action_x'].to(device)
        target_y = batch['action_y'].to(device)
        target_z = batch['action_z'].to(device)
        target_block = batch['action_block'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        target_pos = torch.stack([target_x, target_y, target_z], dim=-1)
        position_logits, block_logits = model(
            state, phase, progress, target_pos=target_pos
        )
        
        # Compute loss
        losses = loss_fn(
            position_logits, block_logits,
            target_x, target_y, target_z, target_block
        )
        
        # Backward pass
        losses['total'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track metrics
        total_loss += losses['total'].item()
        total_pos_loss += losses['position'].item()
        total_block_loss += losses['block'].item()
        total_pos_acc += pos_acc_fn(position_logits, target_x, target_y, target_z)
        total_block_acc += block_acc_fn(block_logits, target_block)
        num_batches += 1
        
        # Log progress
        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] "
                  f"Loss: {losses['total']:.4f} "
                  f"PosAcc: {pos_acc_fn(position_logits, target_x, target_y, target_z):.3f} "
                  f"BlkAcc: {block_acc_fn(block_logits, target_block):.3f} "
                  f"({elapsed:.1f}s)")
    
    return {
        'loss': total_loss / num_batches,
        'position_loss': total_pos_loss / num_batches,
        'block_loss': total_block_loss / num_batches,
        'position_accuracy': total_pos_acc / num_batches,
        'block_accuracy': total_block_acc / num_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: AMBLoss,
    device: torch.device,
) -> Dict[str, float]:
    """
    Validate the model.
    
    Returns:
        Dict with average losses and metrics
    """
    model.eval()
    
    total_loss = 0.0
    total_pos_acc = 0.0
    total_block_acc = 0.0
    stop_stats = {'stop_precision': 0.0, 'stop_recall': 0.0}
    num_batches = 0
    
    pos_acc_fn = PositionAccuracy(loss_fn.max_size)
    block_acc_fn = BlockAccuracy()
    stop_acc_fn = StopAccuracy()
    
    for batch in dataloader:
        state = batch['state'].to(device)
        phase = batch['phase'].to(device)
        progress = batch['progress'].to(device)
        target_x = batch['action_x'].to(device)
        target_y = batch['action_y'].to(device)
        target_z = batch['action_z'].to(device)
        target_block = batch['action_block'].to(device)
        
        target_pos = torch.stack([target_x, target_y, target_z], dim=-1)
        position_logits, block_logits = model(
            state, phase, progress, target_pos=target_pos
        )
        
        losses = loss_fn(
            position_logits, block_logits,
            target_x, target_y, target_z, target_block
        )
        
        total_loss += losses['total'].item()
        total_pos_acc += pos_acc_fn(position_logits, target_x, target_y, target_z)
        total_block_acc += block_acc_fn(block_logits, target_block)
        
        stop_result = stop_acc_fn(block_logits, target_block)
        stop_stats['stop_precision'] += stop_result['stop_precision']
        stop_stats['stop_recall'] += stop_result['stop_recall']
        
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'position_accuracy': total_pos_acc / num_batches,
        'block_accuracy': total_block_acc / num_batches,
        'stop_precision': stop_stats['stop_precision'] / num_batches,
        'stop_recall': stop_stats['stop_recall'] / num_batches,
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int = 50,
    lr: float = 1e-4,
    checkpoint_dir: str = 'checkpoints',
    device: str = 'cuda',
):
    """
    Full training loop.
    
    Args:
        model: BuilderTransformer model
        train_loader: Training dataloader
        val_loader: Optional validation dataloader
        epochs: Number of epochs
        lr: Learning rate
        checkpoint_dir: Directory to save checkpoints
        device: Device to train on
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = AMBLoss(max_size=model.max_size).to(device)
    
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    
    print(f"Training on {device}")
    print(f"Train samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"Val samples: {len(val_loader.dataset)}")
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch
        )
        print(f"Train: Loss={train_metrics['loss']:.4f} "
              f"PosAcc={train_metrics['position_accuracy']:.3f} "
              f"BlkAcc={train_metrics['block_accuracy']:.3f}")
        
        # Validate
        if val_loader:
            val_metrics = validate(model, val_loader, loss_fn, device)
            print(f"Val:   Loss={val_metrics['loss']:.4f} "
                  f"PosAcc={val_metrics['position_accuracy']:.3f} "
                  f"BlkAcc={val_metrics['block_accuracy']:.3f} "
                  f"StopP={val_metrics['stop_precision']:.3f} "
                  f"StopR={val_metrics['stop_recall']:.3f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                }, checkpoint_path / 'best_model.pt')
                print(f"Saved best model (val_loss={best_val_loss:.4f})")
        
        # Step scheduler
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path / f'checkpoint_epoch_{epoch+1}.pt')
    
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    # Example usage
    from amb.models import BuilderTransformerSmall
    from amb.data import create_dataloader
    
    model = BuilderTransformerSmall()
    train_loader = create_dataloader(
        'datasets/organized',
        batch_size=32,
        max_structures=10,  # Small test
    )
    
    train(model, train_loader, None, epochs=5)
