"""
Build Sequence Generator for AMB
---------------------------------
Converts static schematics into valid build sequences.
Each sequence is a trajectory of (state, action) pairs.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Generator
import numpy as np
import random

from .simplifier import Role
from .phase_segmenter import Phase, PhaseSegmenter


@dataclass
class BuildAction:
    """Represents a single build action."""

    x: int
    y: int
    z: int
    block_type: int  # Role ID (0=STOP, 1-5=actual roles)
    phase: int

    def is_stop(self) -> bool:
        return self.block_type == 0

    @staticmethod
    def stop_action() -> "BuildAction":
        return BuildAction(x=0, y=0, z=0, block_type=0, phase=0)


@dataclass
class TrainingSample:
    """A single training sample: (state, action) pair."""

    state: np.ndarray  # Current world state (W, H, D)
    phase: int  # Current build phase
    progress: float  # Build progress (0.0 to 1.0)
    action: BuildAction  # Next action to take


class BuildSequenceGenerator:
    """
    Generates valid build sequences from static structures.

    A build sequence is an ordered list of block placements that:
    1. Respects phase ordering (foundation → wall → roof → window → detail)
    2. Builds bottom-up within each phase
    3. Ensures no floating blocks (each block has support)
    4. Ends with an explicit STOP action

    Usage:
        generator = BuildSequenceGenerator()
        sequence = generator.generate_sequence(roles, phases)
        samples = generator.generate_samples(roles, phases)
    """

    def __init__(
        self,
        randomize_within_phase: bool = True,
        require_support: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Args:
            randomize_within_phase: Randomize block order within each phase
            require_support: Ensure blocks have support before placement
            seed: Random seed for reproducibility
        """
        self.randomize_within_phase = randomize_within_phase
        self.require_support = require_support
        self.rng = random.Random(seed)
        self.phase_segmenter = PhaseSegmenter()

    def generate_sequence(
        self, roles: np.ndarray, phases: Optional[np.ndarray] = None
    ) -> List[BuildAction]:
        """
        Generate a build sequence from a structure.

        Args:
            roles: 3D array of role IDs (W, H, D)
            phases: Optional 3D array of phase IDs

        Returns:
            List of BuildAction objects, ending with STOP
        """
        if phases is None:
            phases = self.phase_segmenter.segment(roles)

        W, H, D = roles.shape
        sequence = []

        # Collect blocks by phase
        phase_blocks = self._collect_blocks_by_phase(roles, phases)

        # Process each phase in order
        for phase_id in range(Phase.num_phases()):
            blocks = phase_blocks[phase_id]

            if not blocks:
                continue

            # Sort by Y (bottom-up), then optionally randomize X,Z
            blocks = self._order_blocks(
                blocks, randomize_xz=self.randomize_within_phase
            )

            # Add to sequence
            for x, y, z, role in blocks:
                # Map role to block_type (shift by 1, 0 is reserved for STOP)
                block_type = role  # roles 1-5 map to block_types 1-5
                action = BuildAction(
                    x=x, y=y, z=z, block_type=block_type, phase=phase_id
                )
                sequence.append(action)

        # Add STOP action
        sequence.append(BuildAction.stop_action())

        return sequence

    def generate_samples(
        self,
        roles: np.ndarray,
        phases: Optional[np.ndarray] = None,
        max_samples: Optional[int] = None,
    ) -> Generator[TrainingSample, None, None]:
        """
        Generate training samples from a structure.

        Each sample is a (state, action) pair where:
        - state: partial build at step t
        - action: next block to place

        Args:
            roles: 3D array of role IDs
            phases: Optional phase assignments
            max_samples: Limit samples (for memory)

        Yields:
            TrainingSample objects
        """
        sequence = self.generate_sequence(roles, phases)
        total_actions = len(sequence)

        if max_samples is not None:
            total_actions = min(total_actions, max_samples + 1)

        # Start with empty world
        W, H, D = roles.shape
        state = np.zeros((W, H, D), dtype=np.int32)

        for t in range(total_actions):
            action = sequence[t]
            progress = t / (len(sequence) - 1) if len(sequence) > 1 else 1.0

            yield TrainingSample(
                state=state.copy(), phase=action.phase, progress=progress, action=action
            )

            # Update state (place block)
            if not action.is_stop():
                state[action.x, action.y, action.z] = action.block_type

    def _collect_blocks_by_phase(
        self, roles: np.ndarray, phases: np.ndarray
    ) -> List[List[Tuple[int, int, int, int]]]:
        """Collect non-air blocks grouped by phase."""
        result = [[] for _ in range(Phase.num_phases())]

        W, H, D = roles.shape
        for x in range(W):
            for y in range(H):
                for z in range(D):
                    role = roles[x, y, z]
                    if role != Role.AIR:
                        phase = phases[x, y, z]
                        result[phase].append((x, y, z, role))

        return result

    def _order_blocks(
        self, blocks: List[Tuple[int, int, int, int]], randomize_xz: bool = True
    ) -> List[Tuple[int, int, int, int]]:
        """
        Order blocks for construction.

        Primary sort: Y ascending (bottom-up)
        Secondary sort: Random X,Z (for data augmentation) or sweep pattern
        """
        if randomize_xz:
            # Group by Y, shuffle within each Y level
            by_y = {}
            for block in blocks:
                y = block[1]
                if y not in by_y:
                    by_y[y] = []
                by_y[y].append(block)

            result = []
            for y in sorted(by_y.keys()):
                level_blocks = by_y[y]
                self.rng.shuffle(level_blocks)
                result.extend(level_blocks)

            return result
        else:
            # Deterministic: Y, then X, then Z
            return sorted(blocks, key=lambda b: (b[1], b[0], b[2]))

    def has_support(self, state: np.ndarray, x: int, y: int, z: int) -> bool:
        """
        Check if a position has structural support.
        A block is supported if:
        - It's at y=0 (ground level), or
        - There's a non-air block below it, or
        - There's a non-air block adjacent (for overhangs)
        """
        if y == 0:
            return True

        # Check below
        if state[x, y - 1, z] != Role.AIR:
            return True

        # Check adjacent (for walls/overhangs)
        W, H, D = state.shape
        for dx, dz in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, nz = x + dx, z + dz
            if 0 <= nx < W and 0 <= nz < D:
                if state[nx, y, nz] != Role.AIR:
                    return True

        return False


def generate_build_sequence(roles: np.ndarray, seed: int = None) -> List[BuildAction]:
    """
    Convenience function to generate a build sequence.

    Args:
        roles: 3D array of role IDs
        seed: Random seed

    Returns:
        List of BuildAction objects
    """
    generator = BuildSequenceGenerator(seed=seed)
    return generator.generate_sequence(roles)


def count_training_samples(roles: np.ndarray) -> int:
    """Count how many training samples a structure would generate."""
    non_air = np.count_nonzero(roles)
    return non_air + 1  # +1 for STOP
