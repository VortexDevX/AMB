"""
AMB Quick Test Script
---------------------
Run this to verify the model can forward pass and generate.
"""

import sys
import numpy as np
import torch

sys.path.insert(0, ".")


def test_data_pipeline():
    """Test data preprocessing components."""
    print("=" * 50)
    print("Testing Data Pipeline")
    print("=" * 50)

    from amb.data.simplifier import BlockSimplifier, Role
    from amb.data.phase_segmenter import PhaseSegmenter, Phase
    from amb.data.sequence_generator import BuildSequenceGenerator

    # Create a simple test structure
    blocks = np.zeros((8, 8, 8), dtype=np.int32)

    # Add a simple house shape
    # Floor
    blocks[1:7, 0, 1:7] = 4  # cobblestone (maps to WALL)
    # Walls
    blocks[1, 1:4, 1:7] = 4
    blocks[6, 1:4, 1:7] = 4
    blocks[1:7, 1:4, 1] = 4
    blocks[1:7, 1:4, 6] = 4
    # Window
    blocks[3, 2, 1] = 20  # glass
    blocks[4, 2, 1] = 20

    # Simplify
    simplifier = BlockSimplifier()
    roles = simplifier.simplify_blocks(blocks)
    roles = simplifier.assign_floor_from_position(roles)
    print(f"Roles shape: {roles.shape}")
    print(f"Non-air blocks: {np.count_nonzero(roles)}")

    # Segment phases
    segmenter = PhaseSegmenter()
    phases = segmenter.segment(roles)
    for phase in range(Phase.num_phases()):
        count = np.sum((phases == phase) & (roles != Role.AIR))
        print(f"  Phase {Phase.names()[phase]}: {count} blocks")

    # Generate sequence
    generator = BuildSequenceGenerator(seed=42)
    sequence = generator.generate_sequence(roles, phases)
    print(f"Build sequence length: {len(sequence)} (incl. STOP)")

    # Generate samples
    samples = list(generator.generate_samples(roles, phases))
    print(f"Training samples: {len(samples)}")

    print("✓ Data pipeline OK\n")
    return roles, phases


def test_model():
    """Test model forward pass."""
    print("=" * 50)
    print("Testing Model")
    print("=" * 50)

    from amb.models.builder_transformer import BuilderTransformer, count_parameters

    # Create small model
    model = BuilderTransformer(max_size=8, d_model=128, n_layers=2)
    print(f"Parameters: {count_parameters(model):,}")

    # Test forward pass
    B = 2
    state = torch.randint(0, 6, (B, 8, 8, 8))
    phase = torch.randint(0, 5, (B,))
    progress = torch.rand(B)

    pos_logits, block_logits = model(state, phase, progress)
    print(f"Position logits shape: {pos_logits.shape}")  # (B, 512)
    print(f"Block logits shape: {block_logits.shape}")  # (B, 6)

    # Test with teacher forcing
    target_pos = torch.randint(0, 8, (B, 3))
    pos_logits, block_logits = model(state, phase, progress, target_pos=target_pos)
    print(f"With teacher forcing: OK")

    print("✓ Model OK\n")
    return model


def test_loss():
    """Test loss computation."""
    print("=" * 50)
    print("Testing Loss")
    print("=" * 50)

    from amb.training.losses import AMBLoss, PositionAccuracy, BlockAccuracy

    loss_fn = AMBLoss(max_size=8)

    B = 4
    pos_logits = torch.randn(B, 512)
    block_logits = torch.randn(B, 6)
    target_x = torch.randint(0, 8, (B,))
    target_y = torch.randint(0, 8, (B,))
    target_z = torch.randint(0, 8, (B,))
    target_block = torch.randint(0, 6, (B,))

    losses = loss_fn(
        pos_logits, block_logits, target_x, target_y, target_z, target_block
    )
    print(f"Total loss: {losses['total']:.4f}")
    print(f"Position loss: {losses['position']:.4f}")
    print(f"Block loss: {losses['block']:.4f}")

    print("✓ Loss OK\n")


def test_inference():
    """Test inference generator."""
    print("=" * 50)
    print("Testing Inference")
    print("=" * 50)

    from amb.models.builder_transformer import BuilderTransformer
    from amb.inference.generator import StructureGenerator

    model = BuilderTransformer(max_size=8, d_model=64, n_layers=1)
    generator = StructureGenerator(model, device="cpu", max_size=8)

    result = generator.generate(max_steps=50, temperature=0.5, sample=True)

    print(f"Steps taken: {result.num_steps}")
    print(f"Stopped naturally: {result.stopped_naturally}")
    print(f"Blocks placed: {len(result.history)}")
    print(f"Non-air in final: {np.count_nonzero(result.structure)}")

    print("✓ Inference OK\n")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("AMB VERIFICATION TEST")
    print("=" * 50 + "\n")

    test_data_pipeline()
    test_model()
    test_loss()
    test_inference()

    print("=" * 50)
    print("ALL TESTS PASSED ✓")
    print("=" * 50)
