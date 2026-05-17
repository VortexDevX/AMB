"""
Schematic/NBT Loader for Minecraft Structures
Loads real Minecraft schematics for training the ML model
"""

import os
import gzip
import struct
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# NBT Tag types
TAG_END = 0
TAG_BYTE = 1
TAG_SHORT = 2
TAG_INT = 3
TAG_LONG = 4
TAG_FLOAT = 5
TAG_DOUBLE = 6
TAG_BYTE_ARRAY = 7
TAG_STRING = 8
TAG_LIST = 9
TAG_COMPOUND = 10
TAG_INT_ARRAY = 11
TAG_LONG_ARRAY = 12


class NBTReader:
    """Simple NBT file reader"""
    
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
    
    def read_byte(self) -> int:
        val = self.data[self.pos]
        self.pos += 1
        return val
    
    def read_short(self) -> int:
        val = struct.unpack('>h', self.data[self.pos:self.pos+2])[0]
        self.pos += 2
        return val
    
    def read_int(self) -> int:
        val = struct.unpack('>i', self.data[self.pos:self.pos+4])[0]
        self.pos += 4
        return val
    
    def read_string(self) -> str:
        length = self.read_short()
        val = self.data[self.pos:self.pos+length].decode('utf-8')
        self.pos += length
        return val
    
    def read_byte_array(self) -> bytes:
        length = self.read_int()
        val = self.data[self.pos:self.pos+length]
        self.pos += length
        return val
    
    def read_long(self) -> int:
        val = struct.unpack('>q', self.data[self.pos:self.pos+8])[0]
        self.pos += 8
        return val
    
    def read_long_array(self) -> List[int]:
        length = self.read_int()
        vals = []
        for _ in range(length):
            vals.append(self.read_long())
        return vals
    
    def skip_tag(self, tag_type: int):
        """Skip over a tag without parsing it"""
        if tag_type == TAG_END:
            pass
        elif tag_type == TAG_BYTE:
            self.pos += 1
        elif tag_type == TAG_SHORT:
            self.pos += 2
        elif tag_type == TAG_INT:
            self.pos += 4
        elif tag_type == TAG_LONG:
            self.pos += 8
        elif tag_type == TAG_FLOAT:
            self.pos += 4
        elif tag_type == TAG_DOUBLE:
            self.pos += 8
        elif tag_type == TAG_BYTE_ARRAY:
            length = self.read_int()
            self.pos += length
        elif tag_type == TAG_STRING:
            length = self.read_short()
            self.pos += length
        elif tag_type == TAG_LIST:
            elem_type = self.read_byte()
            count = self.read_int()
            for _ in range(count):
                self.skip_tag(elem_type)
        elif tag_type == TAG_COMPOUND:
            while True:
                child_type = self.read_byte()
                if child_type == TAG_END:
                    break
                self.read_string()  # name
                self.skip_tag(child_type)
        elif tag_type == TAG_INT_ARRAY:
            length = self.read_int()
            self.pos += length * 4
        elif tag_type == TAG_LONG_ARRAY:
            length = self.read_int()
            self.pos += length * 8


def load_litematic_file(filepath: str) -> Optional[Dict]:
    """
    Load a .litematic file (Litematica format)
    Litematica stores blocks in a packed long array with a palette.
    """
    try:
        # Try using nbtlib first (best approach)
        try:
            import nbtlib
            nbt = nbtlib.load(filepath)
            
            # Navigate to first region
            if 'Regions' not in nbt:
                return None
            
            regions = nbt['Regions']
            for region_name in regions:
                region = regions[region_name]
                
                # Get size
                if 'Size' not in region:
                    continue
                    
                size = region['Size']
                width = abs(int(size.get('x', size.get('X', 0))))
                height = abs(int(size.get('y', size.get('Y', 0))))
                depth = abs(int(size.get('z', size.get('Z', 0))))
                
                if width == 0 or height == 0 or depth == 0:
                    continue
                
                # Get block states
                if 'BlockStates' not in region:
                    continue
                
                block_states_raw = region['BlockStates']
                block_states = [int(v) for v in block_states_raw]
                
                # Get palette
                palette = []
                if 'BlockStatePalette' in region:
                    for entry in region['BlockStatePalette']:
                        name = str(entry.get('Name', 'minecraft:air'))
                        palette.append(block_name_to_id(name))
                
                if not palette:
                    palette = [0, 1]  # Default: air and stone
                
                # Decode packed longs to block array
                blocks = decode_packed_blocks(block_states, width, height, depth, len(palette))
                
                if blocks is not None:
                    # Map palette indices to block IDs
                    final_blocks = []
                    for b in blocks:
                        if b < len(palette):
                            final_blocks.append(palette[b])
                        else:
                            final_blocks.append(0)
                    
                    return {
                        "width": width,
                        "height": height,
                        "depth": depth,
                        "blocks": bytes(final_blocks)
                    }
            
            return None
            
        except ImportError:
            # nbtlib not available, try manual parsing
            return parse_litematic_manual(open(filepath, 'rb').read())
        
    except Exception as e:
        print(f"Failed to load litematic {filepath}: {e}")
        return None


def parse_litematic_nbt(data: bytes) -> Optional[Dict]:
    """
    Parse litematic NBT structure to extract block data.
    Returns {width, height, depth, blocks} or None.
    """
    try:
        # Try using nbtlib if available
        try:
            import nbtlib
            nbt = nbtlib.load(gzip.decompress(data) if data[:2] == b'\x1f\x8b' else data)
            
            # Navigate to first region
            if 'Regions' in nbt:
                regions = nbt['Regions']
                for region_name in regions:
                    region = regions[region_name]
                    
                    # Get size
                    if 'Size' in region:
                        size = region['Size']
                        width = abs(int(size.get('x', size.get('X', 0))))
                        height = abs(int(size.get('y', size.get('Y', 0))))
                        depth = abs(int(size.get('z', size.get('Z', 0))))
                    else:
                        continue
                    
                    if width == 0 or height == 0 or depth == 0:
                        continue
                    
                    # Get block states
                    if 'BlockStates' not in region:
                        continue
                    
                    block_states = list(region['BlockStates'])
                    
                    # Get palette
                    palette = []
                    if 'BlockStatePalette' in region:
                        for entry in region['BlockStatePalette']:
                            name = str(entry.get('Name', 'minecraft:air'))
                            # Map to simple block ID
                            palette.append(block_name_to_id(name))
                    
                    # Decode packed longs to block array
                    blocks = decode_packed_blocks(block_states, width, height, depth, len(palette))
                    
                    if blocks is not None:
                        return {
                            "width": width,
                            "height": height,
                            "depth": depth,
                            "blocks": bytes(blocks)
                        }
            
            return None
            
        except ImportError:
            # nbtlib not available, use fallback
            return parse_litematic_manual(data)
            
    except Exception as e:
        return None


def parse_litematic_manual(data: bytes) -> Optional[Dict]:
    """
    Manual parsing of litematic files without nbtlib.
    Uses pattern matching to find dimensions.
    """
    try:
        # Decompress if needed
        if data[:2] == b'\x1f\x8b':
            data = gzip.decompress(data)
        
        # Look for Size compound - it contains x, y, z ints
        # Pattern: TAG_COMPOUND followed by "Size" string
        size_marker = b'\x0a\x00\x04Size'  # TAG_COMPOUND + length 4 + "Size"
        
        pos = data.find(size_marker)
        if pos == -1:
            # Try lowercase
            size_marker = b'\x0a\x00\x04size'
            pos = data.find(size_marker)
        
        if pos == -1:
            return None
        
        # Move past the marker
        pos += len(size_marker)
        
        # Now read x, y, z (TAG_INT = 0x03)
        dims = {}
        for _ in range(3):
            if pos >= len(data) - 6:
                break
            tag_type = data[pos]
            if tag_type != TAG_INT:
                pos += 1
                continue
            pos += 1
            
            # Read name length and name
            name_len = struct.unpack('>h', data[pos:pos+2])[0]
            pos += 2
            name = data[pos:pos+name_len].decode('utf-8', errors='ignore').lower()
            pos += name_len
            
            # Read int value
            val = struct.unpack('>i', data[pos:pos+4])[0]
            pos += 4
            
            dims[name] = abs(val)
        
        width = dims.get('x', 0)
        height = dims.get('y', 0)
        depth = dims.get('z', 0)
        
        if width == 0 or height == 0 or depth == 0:
            return None
        
        # For now, create a simple filled structure
        # (Full block decoding requires more complex packed long parsing)
        total_blocks = width * height * depth
        blocks = bytes([1] * min(total_blocks, 32768))  # Fill with "wall" block
        
        return {
            "width": width,
            "height": height,
            "depth": depth,
            "blocks": blocks
        }
        
    except Exception as e:
        return None


def block_name_to_id(name: str) -> int:
    """Convert Minecraft block name to simple ID."""
    name = name.lower().replace('minecraft:', '')
    
    # Air
    if 'air' in name:
        return 0
    # Glass/Windows
    if 'glass' in name:
        return 20
    # Doors
    if 'door' in name:
        return 64
    # Stairs/Slabs (roof)
    if 'stair' in name or 'slab' in name:
        return 44
    # Wood
    if 'plank' in name or 'log' in name or 'wood' in name:
        return 5
    # Stone
    if 'stone' in name or 'cobble' in name or 'brick' in name:
        return 4
    
    # Default to stone
    return 1


def decode_packed_blocks(long_array: List[int], width: int, height: int, depth: int, palette_size: int) -> Optional[bytes]:
    """
    Decode Litematica's packed long array to block IDs.
    """
    try:
        total_blocks = width * height * depth
        
        # Calculate bits per block
        bits_per_block = max(2, (palette_size - 1).bit_length())
        
        blocks = []
        block_mask = (1 << bits_per_block) - 1
        
        bit_index = 0
        for _ in range(total_blocks):
            long_index = bit_index // 64
            bit_offset = bit_index % 64
            
            if long_index >= len(long_array):
                break
            
            val = long_array[long_index]
            
            # Handle crossing long boundary
            if bit_offset + bits_per_block <= 64:
                block_id = (val >> bit_offset) & block_mask
            else:
                # Spans two longs
                bits_from_first = 64 - bit_offset
                bits_from_second = bits_per_block - bits_from_first
                
                block_id = (val >> bit_offset) & ((1 << bits_from_first) - 1)
                if long_index + 1 < len(long_array):
                    block_id |= (long_array[long_index + 1] & ((1 << bits_from_second) - 1)) << bits_from_first
            
            blocks.append(block_id & 0xFF)
            bit_index += bits_per_block
        
        # Pad to expected size
        while len(blocks) < total_blocks:
            blocks.append(0)
        
        return blocks[:total_blocks]
        
    except:
        return None


def load_schematic(filepath: str) -> Optional[Dict]:
    """
    Load a .schematic file (MCEdit format)
    Returns: {width, height, depth, blocks, data} or None if failed
    """
    try:
        # Try gzip first
        try:
            with gzip.open(filepath, 'rb') as f:
                data = f.read()
        except:
            with open(filepath, 'rb') as f:
                data = f.read()
        
        reader = NBTReader(data)
        
        # Read root compound
        root_type = reader.read_byte()
        if root_type != TAG_COMPOUND:
            return None
        
        root_name = reader.read_string()
        
        # Parse schematic compound
        result = {}
        while True:
            tag_type = reader.read_byte()
            if tag_type == TAG_END:
                break
            
            name = reader.read_string()
            
            if name == "Width" and tag_type == TAG_SHORT:
                result["width"] = reader.read_short()
            elif name == "Height" and tag_type == TAG_SHORT:
                result["height"] = reader.read_short()
            elif name == "Length" and tag_type == TAG_SHORT:
                result["depth"] = reader.read_short()
            elif name == "Blocks" and tag_type == TAG_BYTE_ARRAY:
                result["blocks"] = reader.read_byte_array()
            elif name == "Data" and tag_type == TAG_BYTE_ARRAY:
                result["data"] = reader.read_byte_array()
            else:
                reader.skip_tag(tag_type)
        
        if all(k in result for k in ["width", "height", "depth", "blocks"]):
            return result
        return None
        
    except Exception as e:
        print(f"Failed to load {filepath}: {e}")
        return None


def schematic_to_voxels(schematic: Dict, max_size: int = 16) -> np.ndarray:
    """
    Convert schematic to voxel array (block IDs)
    Crops or pads to max_size
    """
    w, h, d = schematic["width"], schematic["height"], schematic["depth"]
    blocks = np.frombuffer(schematic["blocks"], dtype=np.uint8)
    
    # Reshape to 3D (Y, Z, X in schematic format)
    try:
        blocks_3d = blocks.reshape((h, d, w))
        # Transpose to (X, Y, Z)
        blocks_3d = blocks_3d.transpose((2, 0, 1))
    except:
        return None
    
    # Crop to max_size
    result = np.zeros((max_size, max_size, max_size), dtype=np.uint8)
    
    cw = min(w, max_size)
    ch = min(h, max_size)
    cd = min(d, max_size)
    
    result[:cw, :ch, :cd] = blocks_3d[:cw, :ch, :cd]
    
    return result


def detect_structure_type(blocks: np.ndarray, filename: str = "") -> str:
    """
    Heuristically detect structure type from block distribution and filename.
    Uses filename patterns first, then falls back to shape analysis.
    """
    filename_lower = filename.lower()
    
    # Filename-based detection (most reliable)
    type_keywords = {
        "tower": ["tower", "turret", "spire", "minaret"],
        "castle": ["castle", "fortress", "keep", "citadel", "stronghold"],
        "house": ["house", "home", "cottage", "villa", "manor"],
        "cabin": ["cabin", "log_cabin", "shack"],
        "hut": ["hut", "shack", "shelter"],
        "barn": ["barn", "stable", "farm_building"],
        "church": ["church", "chapel", "cathedral", "temple", "shrine"],
        "windmill": ["windmill", "mill"],
        "bridge": ["bridge", "viaduct", "overpass"],
        "wall": ["wall", "fence", "barrier", "rampart"],
        "shop": ["shop", "store", "market", "bakery", "smithy"],
        "inn": ["inn", "tavern", "pub", "hotel"],
        "lighthouse": ["lighthouse", "beacon"],
        "dock": ["dock", "pier", "harbor", "wharf"],
        "farm": ["farm", "field", "crop"],
        "fortress": ["fort", "bunker", "outpost"],
    }
    
    for struct_type, keywords in type_keywords.items():
        if any(kw in filename_lower for kw in keywords):
            return struct_type
    
    # Shape-based fallback
    non_air = np.sum(blocks > 0)
    h, w, d = blocks.shape[1], blocks.shape[0], blocks.shape[2]
    
    # Height ratio (tall = tower)
    if h > w * 1.5 and h > d * 1.5:
        return "tower"
    
    # Long and thin (wall or bridge)
    if w > h * 3 or d > h * 3:
        return "wall"
    
    # Large footprint, moderate height (castle)
    if w * d > 100 and h > 5:
        return "castle"
    
    # Small footprint (hut/cabin)
    if w * d < 25:
        return "hut"
    
    # Default to house
    return "house"


def map_classic_to_role(block_id: int) -> int:
    """
    Map classic Minecraft block IDs to semantic roles
    0=air, 1=wall, 2=floor, 3=roof, 4=window, 5=door
    """
    # Air
    if block_id == 0:
        return 0
    
    # Glass = window
    if block_id in [20, 102]:  # glass, glass_pane
        return 4
    
    # Doors
    if block_id in [64, 71]:  # oak_door, iron_door
        return 5
    
    # Slabs, stairs (often roof or floor)
    if block_id in [44, 53, 67, 108, 109, 114, 128, 134, 135, 136]:
        return 3  # roof
    
    # Wood planks, logs, stone, cobble (walls/floors)
    if block_id in [1, 4, 5, 17, 162]:  # stone, cobble, planks, logs
        return 1  # wall (most common)
    
    # Default to wall
    return 1


def convert_to_roles(blocks: np.ndarray) -> np.ndarray:
    """Convert block IDs to semantic roles"""
    roles = np.zeros_like(blocks, dtype=np.int64)
    
    for block_id in np.unique(blocks):
        role = map_classic_to_role(int(block_id))
        roles[blocks == block_id] = role
    
    return roles


def validate_schematic(
    voxels: np.ndarray, 
    min_blocks: int = 10, 
    max_air_ratio: float = 0.98,
    min_dimension: int = 2
) -> bool:
    """
    Validate a schematic to filter out empty/invalid structures.
    
    Args:
        voxels: 3D numpy array of block IDs
        min_blocks: Minimum number of non-air blocks
        max_air_ratio: Maximum ratio of air blocks allowed
        min_dimension: Minimum size in each dimension
    
    Returns:
        True if schematic is valid, False otherwise
    """
    # Check dimensions
    w, h, d = voxels.shape
    if w < min_dimension or h < min_dimension or d < min_dimension:
        return False
    
    # Count non-air blocks
    non_air = np.count_nonzero(voxels)
    if non_air < min_blocks:
        return False
    
    # Check air ratio
    total = voxels.size
    air_ratio = (total - non_air) / total
    if air_ratio > max_air_ratio:
        return False
    
    return True


def load_schem_file(filepath: str) -> Optional[Dict]:
    """
    Load a .schem file (WorldEdit Sponge format - NBT based)
    Returns voxel data or None
    """
    try:
        # Try gzip first (most .schem files are compressed)
        try:
            with gzip.open(filepath, 'rb') as f:
                data = f.read()
        except:
            with open(filepath, 'rb') as f:
                data = f.read()
        
        # Simple approach: look for dimensions in the NBT data
        # Schem format has Width, Height, Length as shorts
        reader = NBTReader(data)
        
        # Read root compound
        root_type = reader.read_byte()
        if root_type != TAG_COMPOUND:
            return None
        
        root_name = reader.read_string()
        
        result = {}
        block_data = None
        palette = {}
        
        while reader.pos < len(data) - 1:
            try:
                tag_type = reader.read_byte()
                if tag_type == TAG_END:
                    break
                
                name = reader.read_string()
                
                if name == "Width" and tag_type == TAG_SHORT:
                    result["width"] = reader.read_short()
                elif name == "Height" and tag_type == TAG_SHORT:
                    result["height"] = reader.read_short()
                elif name == "Length" and tag_type == TAG_SHORT:
                    result["depth"] = reader.read_short()
                elif name == "BlockData" and tag_type == TAG_BYTE_ARRAY:
                    block_data = reader.read_byte_array()
                elif name == "Blocks" and tag_type == TAG_BYTE_ARRAY:
                    block_data = reader.read_byte_array()
                else:
                    reader.skip_tag(tag_type)
            except:
                break
        
        if all(k in result for k in ["width", "height", "depth"]) and block_data:
            result["blocks"] = block_data
            return result
        
        return None
        
    except Exception as e:
        print(f"Failed to load schem {filepath}: {e}")
        return None


def load_schematics_from_dir(
    directory: str, 
    max_structures: int = 1000,
    max_size: int = 32
) -> List[Dict]:
    """
    Load all schematics from a directory (.schematic and .schem files)
    Returns list of {voxels, roles, structure_type, dimensions}
    """
    results = []
    
    path = Path(directory)
    if not path.exists():
        print(f"Directory not found: {directory}")
        return results
    
    # Find all formats
    schematic_files = list(path.glob("**/*.schematic"))
    schem_files = list(path.glob("**/*.schem"))
    litematic_files = list(path.glob("**/*.litematic"))
    
    all_files = schematic_files + schem_files + litematic_files
    print(f"Found {len(schematic_files)} .schematic files")
    print(f"Found {len(schem_files)} .schem files")
    print(f"Found {len(litematic_files)} .litematic files")
    print(f"Total: {len(all_files)} files")
    
    for i, filepath in enumerate(all_files[:max_structures]):
        # Load based on extension
        ext = filepath.suffix.lower()
        if ext == ".schematic":
            schematic = load_schematic(str(filepath))
        elif ext == ".schem":
            schematic = load_schem_file(str(filepath))
        elif ext == ".litematic":
            schematic = load_litematic_file(str(filepath))
        else:
            continue
        
        if schematic is None:
            print(f"  Skipped: {filepath.name}")
            continue
        
        voxels = schematic_to_voxels(schematic, max_size)
        if voxels is None:
            continue
        
        # Validate schematic
        if not validate_schematic(voxels):
            print(f"  Skipped (invalid): {filepath.name}")
            continue
        
        roles = convert_to_roles(voxels)
        structure_type = detect_structure_type(voxels, filepath.name)
        
        results.append({
            "voxels": voxels,
            "roles": roles,
            "structure_type": structure_type,
            "dimensions": (schematic["width"], schematic["height"], schematic["depth"]),
            "filename": filepath.name
        })
        
        print(f"  Loaded: {filepath.name} ({schematic['width']}x{schematic['height']}x{schematic['depth']})")
        
        if (i + 1) % 100 == 0:
            print(f"Loaded {i + 1} schematics")
    
    print(f"\nSuccessfully loaded {len(results)} schematics")
    return results


if __name__ == "__main__":
    # Test loading
    import sys
    
    if len(sys.argv) > 1:
        results = load_schematics_from_dir(sys.argv[1])
        for r in results[:5]:
            print(f"{r['filename']}: {r['structure_type']} {r['dimensions']}")
    else:
        print("Usage: python schematic_loader.py <schematic_directory>")
