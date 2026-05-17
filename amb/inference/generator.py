"""
Structure Generator for AMB
---------------------------
Inference loop that generates structures step-by-step.
"""

import torch
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

from amb.data.simplifier import Role
from amb.data.phase_segmenter import Phase


@dataclass
class GenerationResult:
    """Result of structure generation."""

    structure: np.ndarray  # Final voxel grid (W, H, D)
    num_steps: int  # Number of steps taken
    stopped_naturally: bool  # Whether model predicted STOP
    history: List[Tuple[int, int, int, int]]  # (x, y, z, block) placements


class StructureGenerator:
    """
    Generates Minecraft structures using trained model.

    Starts from empty world and places blocks one at a time
    until model predicts STOP or max steps reached.

    Usage:
        generator = StructureGenerator(model)
        result = generator.generate(max_steps=500)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        max_size: int = 16,
    ):
        """
        Args:
            model: Trained BuilderTransformer model
            device: Device to run inference on
            max_size: Voxel grid size
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        self.max_size = max_size

    @torch.no_grad()
    def generate(
        self,
        max_steps: int = 1000,
        temperature: float = 0.8,
        sample: bool = True,
        starting_state: Optional[np.ndarray] = None,
        starting_phase: int = Phase.FOUNDATION,
    ) -> GenerationResult:
        """
        Generate a structure.

        Args:
            max_steps: Maximum number of placement steps
            temperature: Sampling temperature (lower = more deterministic)
            sample: If True, sample actions; else use argmax
            starting_state: Optional initial state (for continuation)
            starting_phase: Starting build phase

        Returns:
            GenerationResult with final structure and metadata
        """
        # Initialize state
        if starting_state is not None:
            state = starting_state.copy()
        else:
            state = np.zeros(
                (self.max_size, self.max_size, self.max_size), dtype=np.int32
            )

        history = []
        current_phase = starting_phase

        for step in range(max_steps):
            progress = step / max_steps

            # Convert to tensors
            state_tensor = torch.from_numpy(state).unsqueeze(0).long().to(self.device)
            phase_tensor = torch.tensor([current_phase], dtype=torch.long).to(
                self.device
            )
            progress_tensor = torch.tensor([progress], dtype=torch.float32).to(
                self.device
            )

            # Get model prediction
            position_logits, block_logits = self.model(
                state_tensor, phase_tensor, progress_tensor
            )

            # Sample position
            x, y, z = self._sample_position(position_logits, temperature, sample)

            # Sample block type
            block_type = self._sample_block(block_logits, temperature, sample)

            # Check for STOP
            if block_type == 0:
                return GenerationResult(
                    structure=state,
                    num_steps=step,
                    stopped_naturally=True,
                    history=history,
                )

            # Validate and place block
            if self._is_valid_placement(state, x, y, z):
                state[x, y, z] = block_type
                history.append((x, y, z, block_type))

            # Update phase based on progress/state
            current_phase = self._update_phase(state, current_phase, progress)

        # Max steps reached
        return GenerationResult(
            structure=state,
            num_steps=max_steps,
            stopped_naturally=False,
            history=history,
        )

    def _sample_position(
        self,
        logits: torch.Tensor,  # (1, W*H*D)
        temperature: float,
        sample: bool,
    ) -> Tuple[int, int, int]:
        """Sample position from logits."""
        logits = logits.squeeze(0)

        if sample and temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            idx = torch.multinomial(probs, 1).item()
        else:
            idx = logits.argmax().item()

        # Convert flat index to (x, y, z)
        z = idx % self.max_size
        y = (idx // self.max_size) % self.max_size
        x = idx // (self.max_size**2)

        return x, y, z

    def _sample_block(
        self,
        logits: torch.Tensor,  # (1, num_block_types)
        temperature: float,
        sample: bool,
    ) -> int:
        """Sample block type from logits."""
        logits = logits.squeeze(0)

        if sample and temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            return torch.multinomial(probs, 1).item()
        else:
            return logits.argmax().item()

    def _is_valid_placement(self, state: np.ndarray, x: int, y: int, z: int) -> bool:
        """Check if placement is valid (not already occupied, has support)."""
        # Bounds check
        if not (
            0 <= x < self.max_size and 0 <= y < self.max_size and 0 <= z < self.max_size
        ):
            return False

        # Already occupied
        if state[x, y, z] != 0:
            return False

        # Support check (block below or at ground level)
        if y == 0:
            return True

        if state[x, y - 1, z] != 0:
            return True

        # Adjacent support (for walls/overhangs)
        for dx, dz in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, nz = x + dx, z + dz
            if 0 <= nx < self.max_size and 0 <= nz < self.max_size:
                if state[nx, y, nz] != 0:
                    return True

        return False

    def _update_phase(
        self,
        state: np.ndarray,
        current_phase: int,
        progress: float,
    ) -> int:
        """Update phase based on build progress."""
        # Simple progress-based phase transitions
        if progress < 0.2:
            return Phase.FOUNDATION
        elif progress < 0.5:
            return Phase.WALL
        elif progress < 0.7:
            return Phase.ROOF
        elif progress < 0.85:
            return Phase.WINDOW
        else:
            return Phase.DETAIL

    @torch.no_grad()
    def reconstruct(
        self,
        target: np.ndarray,
        max_steps: Optional[int] = None,
        temperature: float = 0.0,  # Deterministic for reconstruction
    ) -> GenerationResult:
        """
        Test reconstruction of a target structure.
        Uses ground truth to guide generation for debugging.

        Args:
            target: Target structure to reconstruct
            max_steps: Max steps (default: count of blocks + 10)
            temperature: Sampling temperature

        Returns:
            GenerationResult
        """
        num_blocks = np.count_nonzero(target)
        if max_steps is None:
            max_steps = num_blocks + 10

        return self.generate(
            max_steps=max_steps,
            temperature=temperature,
            sample=False,
        )


def load_generator(
    checkpoint_path: str,
    model_class=None,
    device: str = "cuda",
    max_size: int = 16,
) -> StructureGenerator:
    """
    Load a trained model and create generator.

    Args:
        checkpoint_path: Path to model checkpoint (.pt)
        model_class: Model class (default: BuilderTransformer)
        device: Device
        max_size: Grid size

    Returns:
        StructureGenerator instance
    """
    from amb.models import BuilderTransformer

    if model_class is None:
        model_class = BuilderTransformer

    # Create model
    model = model_class(max_size=max_size)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return StructureGenerator(model, device=device, max_size=max_size)
