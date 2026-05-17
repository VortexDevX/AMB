"""
PyTorch Dataset for AMB
-----------------------
Loads schematics and generates training samples for sequential building.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .schematic_loader import (
    load_schematic,
    load_schem_file,
    load_litematic_file,
    schematic_to_voxels,
    validate_schematic,
)
from .simplifier import BlockSimplifier, Role, simplify_structure
from .phase_segmenter import PhaseSegmenter, Phase, segment_structure
from .sequence_generator import (
    BuildSequenceGenerator,
    BuildAction,
    TrainingSample,
)


class AMBDataset(Dataset):
    """
    PyTorch Dataset for AMB training.
    
    Each item is a (state, phase, progress, action) tuple.
    
    Usage:
        dataset = AMBDataset(
            schematic_dir='datasets/organized',
            max_size=16
        )
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    """
    
    def __init__(
        self,
        schematic_dir: str,
        max_size: int = 16,
        max_structures: Optional[int] = None,
        augment_rotations: bool = True,
        cache_samples: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            schematic_dir: Directory containing .schematic/.schem files
            max_size: Maximum voxel grid size (structures cropped/padded)
            max_structures: Limit number of structures to load
            augment_rotations: If True, include rotated versions
            cache_samples: If True, precompute all samples in memory
            seed: Random seed for reproducibility
        """
        self.max_size = max_size
        self.augment_rotations = augment_rotations
        self.cache_samples = cache_samples
        self.seed = seed
        
        self.simplifier = BlockSimplifier()
        self.segmenter = PhaseSegmenter()
        self.sequence_gen = BuildSequenceGenerator(seed=seed)
        
        # Load structures
        self.structures = self._load_structures(schematic_dir, max_structures)
        print(f"Loaded {len(self.structures)} structures")
        
        # Generate samples
        if cache_samples:
            self.samples = self._generate_all_samples()
            print(f"Generated {len(self.samples)} training samples")
        else:
            self.samples = None
            # Calculate sample count without caching
            self._sample_indices = self._build_sample_index()
    
    def _load_structures(
        self, 
        schematic_dir: str, 
        max_structures: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Load and preprocess structures from directory."""
        structures = []
        path = Path(schematic_dir)
        
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {schematic_dir}")
        
        # Find all schematic files
        extensions = ['*.schematic', '*.schem', '*.litematic']
        files = []
        for ext in extensions:
            files.extend(path.glob(f'**/{ext}'))
        
        if max_structures:
            files = files[:max_structures]
        
        for filepath in files:
            try:
                # Load based on extension
                ext = filepath.suffix.lower()
                if ext == '.schematic':
                    schematic = load_schematic(str(filepath))
                elif ext == '.schem':
                    schematic = load_schem_file(str(filepath))
                elif ext == '.litematic':
                    schematic = load_litematic_file(str(filepath))
                else:
                    continue
                
                if schematic is None:
                    continue
                
                # Convert to voxels
                voxels = schematic_to_voxels(schematic, self.max_size)
                if voxels is None:
                    continue
                
                # Validate
                if not validate_schematic(voxels):
                    continue
                
                # Simplify to roles
                roles = simplify_structure(voxels)
                
                # Segment into phases
                phases = segment_structure(roles)
                
                structures.append({
                    'roles': roles,
                    'phases': phases,
                    'filename': filepath.name,
                    'dimensions': (
                        schematic['width'],
                        schematic['height'],
                        schematic['depth']
                    )
                })
                
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue
        
        return structures
    
    def _generate_all_samples(self) -> List[Tuple[np.ndarray, int, float, BuildAction]]:
        """Generate all training samples from structures."""
        samples = []
        
        for struct in self.structures:
            roles = struct['roles']
            phases = struct['phases']
            
            # Generate rotations if augmenting
            rotations = self._get_rotations(roles, phases) if self.augment_rotations else [(roles, phases)]
            
            for rot_roles, rot_phases in rotations:
                # Generate samples from this orientation
                for sample in self.sequence_gen.generate_samples(rot_roles, rot_phases):
                    samples.append((
                        sample.state,
                        sample.phase,
                        sample.progress,
                        sample.action
                    ))
        
        return samples
    
    def _get_rotations(
        self, 
        roles: np.ndarray, 
        phases: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get 4 rotations (90° increments) of a structure."""
        rotations = [(roles, phases)]
        
        r, p = roles, phases
        for _ in range(3):
            # Rotate 90° around Y axis
            r = np.rot90(r, axes=(0, 2))
            p = np.rot90(p, axes=(0, 2))
            rotations.append((r.copy(), p.copy()))
        
        return rotations
    
    def _build_sample_index(self) -> List[Tuple[int, int]]:
        """Build index mapping sample_idx -> (structure_idx, step_idx)."""
        indices = []
        for struct_idx, struct in enumerate(self.structures):
            roles = struct['roles']
            n_samples = np.count_nonzero(roles) + 1  # +1 for STOP
            if self.augment_rotations:
                n_samples *= 4  # 4 rotations
            for step_idx in range(n_samples):
                indices.append((struct_idx, step_idx))
        return indices
    
    def __len__(self) -> int:
        if self.cache_samples:
            return len(self.samples)
        else:
            return len(self._sample_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            Dict with keys:
                - state: (max_size, max_size, max_size) int tensor
                - phase: scalar int
                - progress: scalar float
                - action_x, action_y, action_z: scalar ints
                - action_block: scalar int (0=STOP, 1-5=roles)
        """
        if self.cache_samples:
            state, phase, progress, action = self.samples[idx]
        else:
            # On-the-fly generation (slower but memory efficient)
            struct_idx, step_idx = self._sample_indices[idx]
            struct = self.structures[struct_idx]
            roles, phases = struct['roles'], struct['phases']
            
            # Regenerate sequence and get specific sample
            samples = list(self.sequence_gen.generate_samples(roles, phases))
            sample = samples[step_idx % len(samples)]
            state, phase, progress, action = sample.state, sample.phase, sample.progress, sample.action
        
        return {
            'state': torch.from_numpy(state).long(),
            'phase': torch.tensor(phase, dtype=torch.long),
            'progress': torch.tensor(progress, dtype=torch.float32),
            'action_x': torch.tensor(action.x, dtype=torch.long),
            'action_y': torch.tensor(action.y, dtype=torch.long),
            'action_z': torch.tensor(action.z, dtype=torch.long),
            'action_block': torch.tensor(action.block_type, dtype=torch.long),
        }


def create_dataloader(
    schematic_dir: str,
    batch_size: int = 32,
    max_size: int = 16,
    max_structures: Optional[int] = None,
    num_workers: int = 0,
    shuffle: bool = True,
    **kwargs
) -> DataLoader:
    """
    Convenience function to create a DataLoader.
    
    Args:
        schematic_dir: Directory with schematics
        batch_size: Batch size
        max_size: Max voxel grid size
        max_structures: Limit structures
        num_workers: DataLoader workers
        shuffle: Shuffle samples
        
    Returns:
        PyTorch DataLoader
    """
    dataset = AMBDataset(
        schematic_dir=schematic_dir,
        max_size=max_size,
        max_structures=max_structures,
        **kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
