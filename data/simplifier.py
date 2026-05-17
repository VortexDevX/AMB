"""
Block Simplifier for AMB
-------------------------
Simplifies real Minecraft blocks to structural categories.
Removes decorative blocks and collapses variants.
"""

from typing import Dict, Set
import numpy as np


# Semantic role IDs
class Role:
    AIR = 0
    WALL = 1
    FLOOR = 2
    ROOF = 3
    WINDOW = 4
    DOOR = 5

    NUM_ROLES = 6

    NAMES = ["air", "wall", "floor", "roof", "window", "door"]


# Decorative blocks to remove (these add noise, not structure)
DECOR_BLOCKS: Set[str] = {
    "torch",
    "wall_torch",
    "redstone_torch",
    "soul_torch",
    "lantern",
    "soul_lantern",
    "candle",
    "sea_pickle",
    "flower",
    "poppy",
    "dandelion",
    "rose",
    "tulip",
    "orchid",
    "allium",
    "azure_bluet",
    "oxeye_daisy",
    "cornflower",
    "lily_of_the_valley",
    "wither_rose",
    "sunflower",
    "lilac",
    "peony",
    "rose_bush",
    "banner",
    "sign",
    "wall_sign",
    "hanging_sign",
    "painting",
    "item_frame",
    "glow_item_frame",
    "carpet",
    "moss_carpet",
    "bed",
    "white_bed",
    "orange_bed",
    "magenta_bed",
    "light_blue_bed",
    "yellow_bed",
    "lime_bed",
    "pink_bed",
    "gray_bed",
    "light_gray_bed",
    "cyan_bed",
    "purple_bed",
    "blue_bed",
    "brown_bed",
    "green_bed",
    "red_bed",
    "black_bed",
    "chest",
    "trapped_chest",
    "ender_chest",
    "barrel",
    "shulker_box",
    "flower_pot",
    "potted_",
    "decorated_pot",
    "head",
    "skull",
    "player_head",
    "zombie_head",
    "skeleton_skull",
    "creeper_head",
    "dragon_head",
    "piglin_head",
    "armor_stand",
    "fence",
    "fence_gate",
    "nether_brick_fence",
    "wall",
    "cobblestone_wall",
    "mossy_cobblestone_wall",
    "brick_wall",
    "prismarine_wall",
    "sandstone_wall",
    "red_sandstone_wall",
}

# Block categories for role mapping
STRUCTURAL_BLOCKS: Dict[str, int] = {
    # Floor blocks
    "floor": Role.FLOOR,
    "ground": Role.FLOOR,
    # Glass = windows
    "glass": Role.WINDOW,
    "glass_pane": Role.WINDOW,
    "tinted_glass": Role.WINDOW,
    "stained_glass": Role.WINDOW,
    # Doors
    "door": Role.DOOR,
    "oak_door": Role.DOOR,
    "spruce_door": Role.DOOR,
    "birch_door": Role.DOOR,
    "jungle_door": Role.DOOR,
    "acacia_door": Role.DOOR,
    "dark_oak_door": Role.DOOR,
    "mangrove_door": Role.DOOR,
    "cherry_door": Role.DOOR,
    "bamboo_door": Role.DOOR,
    "crimson_door": Role.DOOR,
    "warped_door": Role.DOOR,
    "iron_door": Role.DOOR,
    # Roof blocks (stairs, slabs)
    "stair": Role.ROOF,
    "stairs": Role.ROOF,
    "slab": Role.ROOF,
}

# Classic block IDs to roles (for .schematic format)
CLASSIC_ID_TO_ROLE: Dict[int, int] = {
    0: Role.AIR,  # air
    20: Role.WINDOW,  # glass
    102: Role.WINDOW,  # glass_pane
    64: Role.DOOR,  # oak_door
    71: Role.DOOR,  # iron_door
    193: Role.DOOR,  # spruce_door
    194: Role.DOOR,  # birch_door
    195: Role.DOOR,  # jungle_door
    196: Role.DOOR,  # acacia_door
    197: Role.DOOR,  # dark_oak_door
    44: Role.ROOF,  # stone_slab
    53: Role.ROOF,  # oak_stairs
    67: Role.ROOF,  # cobblestone_stairs
    108: Role.ROOF,  # brick_stairs
    109: Role.ROOF,  # stone_brick_stairs
    114: Role.ROOF,  # nether_brick_stairs
    128: Role.ROOF,  # sandstone_stairs
    134: Role.ROOF,  # spruce_stairs
    135: Role.ROOF,  # birch_stairs
    136: Role.ROOF,  # jungle_stairs
}


class BlockSimplifier:
    """
    Simplifies Minecraft blocks to structural roles.

    Usage:
        simplifier = BlockSimplifier()
        roles = simplifier.simplify_blocks(blocks_3d)
    """

    def __init__(self, remove_decor: bool = True):
        """
        Args:
            remove_decor: If True, decorative blocks become air
        """
        self.remove_decor = remove_decor

    def block_name_to_role(self, name: str) -> int:
        """Convert modern block name to role ID."""
        name = name.lower().replace("minecraft:", "")

        # Check if decorative
        if self.remove_decor:
            for decor in DECOR_BLOCKS:
                if decor in name:
                    return Role.AIR

        # Check structural mappings
        for pattern, role in STRUCTURAL_BLOCKS.items():
            if pattern in name:
                return role

        # Default non-air blocks to wall
        if "air" in name:
            return Role.AIR

        return Role.WALL

    def classic_id_to_role(self, block_id: int) -> int:
        """Convert classic block ID to role ID."""
        if block_id in CLASSIC_ID_TO_ROLE:
            return CLASSIC_ID_TO_ROLE[block_id]

        # Air
        if block_id == 0:
            return Role.AIR

        # Default to wall
        return Role.WALL

    def simplify_blocks(
        self, blocks: np.ndarray, use_classic_ids: bool = True
    ) -> np.ndarray:
        """
        Convert 3D block array to roles.

        Args:
            blocks: 3D numpy array of block IDs or names
            use_classic_ids: If True, treat as classic numeric IDs

        Returns:
            3D numpy array of role IDs (0-5)
        """
        roles = np.zeros_like(blocks, dtype=np.int32)

        if use_classic_ids:
            for block_id in np.unique(blocks):
                role = self.classic_id_to_role(int(block_id))
                roles[blocks == block_id] = role
        else:
            # For string-based block names (future support)
            raise NotImplementedError("String block names not yet supported")

        return roles

    def assign_floor_from_position(self, roles: np.ndarray) -> np.ndarray:
        """
        Post-process to assign floor role based on position.
        Blocks at y=0 that are walls become floors.

        Args:
            roles: 3D array of roles (X, Y, Z)

        Returns:
            Modified roles array
        """
        result = roles.copy()
        # At y=0, convert walls to floors
        result[:, 0, :] = np.where(
            result[:, 0, :] == Role.WALL, Role.FLOOR, result[:, 0, :]
        )
        return result


def simplify_structure(blocks: np.ndarray) -> np.ndarray:
    """
    Convenience function to simplify a structure.

    Args:
        blocks: 3D array of classic block IDs

    Returns:
        3D array of role IDs
    """
    simplifier = BlockSimplifier()
    roles = simplifier.simplify_blocks(blocks)
    roles = simplifier.assign_floor_from_position(roles)
    return roles
