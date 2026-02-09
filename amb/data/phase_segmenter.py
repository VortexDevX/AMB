"""
Phase Segmenter for AMB
-----------------------
Assigns build phases to blocks based on position and role.
Enables hierarchical, phase-by-phase construction.
"""

from enum import IntEnum
from typing import Tuple, List
import numpy as np

from .simplifier import Role


class Phase(IntEnum):
    """Build phases in construction order."""
    FOUNDATION = 0
    WALL = 1
    ROOF = 2
    WINDOW = 3
    DETAIL = 4
    
    @classmethod
    def num_phases(cls) -> int:
        return 5
    
    @classmethod
    def names(cls) -> List[str]:
        return ['foundation', 'wall', 'roof', 'window', 'detail']


class PhaseSegmenter:
    """
    Assigns build phases to each block in a structure.
    
    Phases determine the order in which blocks should be placed:
    1. FOUNDATION - Bottom layer (y=0) and floors
    2. WALL - Vertical structure elements
    3. ROOF - Top coverage (stairs, slabs at top)
    4. WINDOW - Glass and openings (after walls)
    5. DETAIL - Everything else (decorative)
    
    Usage:
        segmenter = PhaseSegmenter()
        phases = segmenter.segment(roles)  # 3D array of phase IDs
    """
    
    def __init__(self):
        pass
    
    def segment(self, roles: np.ndarray) -> np.ndarray:
        """
        Assign phases to each block.
        
        Args:
            roles: 3D array of role IDs (X, Y, Z), shape (W, H, D)
            
        Returns:
            3D array of phase IDs, same shape as input
        """
        W, H, D = roles.shape
        phases = np.full_like(roles, Phase.DETAIL, dtype=np.int32)
        
        # Find the maximum Y with non-air blocks (roof level)
        non_air_mask = roles != Role.AIR
        if not np.any(non_air_mask):
            return phases
        
        # Get Y coordinates of all non-air blocks
        y_coords = np.where(non_air_mask)[1]
        max_y = y_coords.max() if len(y_coords) > 0 else 0
        min_y = y_coords.min() if len(y_coords) > 0 else 0
        
        # FOUNDATION: y == min_y or role == FLOOR
        foundation_mask = (
            (np.arange(H)[None, :, None] == min_y) |
            (roles == Role.FLOOR)
        ) & non_air_mask
        phases[foundation_mask] = Phase.FOUNDATION
        
        # ROOF: y == max_y or role == ROOF
        roof_mask = (
            (np.arange(H)[None, :, None] == max_y) |
            (roles == Role.ROOF)
        ) & non_air_mask & ~foundation_mask
        phases[roof_mask] = Phase.ROOF
        
        # WINDOW: role == WINDOW
        window_mask = (roles == Role.WINDOW) & non_air_mask
        phases[window_mask] = Phase.WINDOW
        
        # DOOR: treat as DETAIL (placed after walls)
        door_mask = (roles == Role.DOOR) & non_air_mask
        phases[door_mask] = Phase.DETAIL
        
        # WALL: remaining structural blocks (walls)
        wall_mask = (
            (roles == Role.WALL) & 
            non_air_mask &
            ~foundation_mask &
            ~roof_mask
        )
        phases[wall_mask] = Phase.WALL
        
        # AIR remains as DETAIL (won't be placed)
        phases[roles == Role.AIR] = Phase.DETAIL
        
        return phases
    
    def get_blocks_by_phase(
        self, 
        roles: np.ndarray, 
        phases: np.ndarray
    ) -> List[List[Tuple[int, int, int, int]]]:
        """
        Get blocks grouped by phase.
        
        Args:
            roles: 3D array of role IDs
            phases: 3D array of phase IDs
            
        Returns:
            List of 5 lists, each containing (x, y, z, role) tuples
        """
        result = [[] for _ in range(Phase.num_phases())]
        
        W, H, D = roles.shape
        for x in range(W):
            for y in range(H):
                for z in range(D):
                    if roles[x, y, z] != Role.AIR:
                        phase = phases[x, y, z]
                        role = roles[x, y, z]
                        result[phase].append((x, y, z, role))
        
        return result


def segment_structure(roles: np.ndarray) -> np.ndarray:
    """
    Convenience function to segment a structure.
    
    Args:
        roles: 3D array of role IDs
        
    Returns:
        3D array of phase IDs
    """
    segmenter = PhaseSegmenter()
    return segmenter.segment(roles)
