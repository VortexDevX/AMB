"""
Auto Dataset Organizer
----------------------
Automatically organizes schematic files into structure type folders.
Run this whenever you add new data to keep the dataset organized.

Usage:
    python organize.py              # Organize all subdirectories
    python organize.py --clean      # Clean and re-organize everything
"""

import os
import sys
import shutil
import argparse
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import re

# Structure type keywords for filename detection
STRUCTURE_KEYWORDS = {
    "house": [
        "house",
        "home",
        "cottage",
        "villa",
        "manor",
        "residence",
        "dwelling",
        "bungalow",
        "starter",
        "survival",
    ],
    "castle": ["castle", "fortress", "keep", "citadel", "stronghold", "palace"],
    "tower": [
        "tower",
        "turret",
        "spire",
        "minaret",
        "watchtower",
        "lighthouse",
        "beacon",
    ],
    "church": ["church", "chapel", "cathedral", "temple", "shrine", "mosque", "abbey"],
    "cabin": ["cabin", "log_cabin", "lodge", "chalet", "cozy"],
    "medieval": [
        "medieval",
        "tavern",
        "inn",
        "blacksmith",
        "smithy",
        "market",
        "village",
    ],
    "farm": ["farm", "barn", "stable", "silo", "windmill", "mill", "crop"],
    "bridge": ["bridge", "viaduct", "overpass", "pier", "dock", "harbor"],
    "wall": ["wall", "gate", "rampart", "fence", "fortification"],
    "modern": ["modern", "contemporary", "skyscraper", "apartment", "office"],
    "fantasy": ["fantasy", "elven", "dwarven", "wizard", "magic", "mushroom"],
    "ship": ["ship", "boat", "vessel", "yacht", "galleon", "pirate"],
    "statue": ["statue", "monument", "sculpture", "fountain"],
    "base": ["base", "bunker", "outpost", "camp", "storage"],
    "misc": [],  # Fallback
}

VALID_EXTENSIONS = [".schematic", ".schem", ".litematic", ".nbt"]


def detect_structure_type(filename: str) -> str:
    """Detect structure type from filename."""
    name_lower = filename.lower()

    for struct_type, keywords in STRUCTURE_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return struct_type

    return "misc"


def get_file_hash(filepath: Path) -> str:
    """Get MD5 hash of file for duplicate detection."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_valid_schematic(filepath: Path) -> bool:
    """Check if file is a valid schematic."""
    if filepath.stat().st_size < 100:
        return False

    try:
        with open(filepath, "rb") as f:
            header = f.read(10)
            if header.startswith(b"<!DOCTYPE") or header.startswith(b"<html"):
                return False
    except:
        return False

    return True


def find_all_schematics(root_dir: Path, exclude_dirs: list = None) -> list:
    """Find all schematic files recursively."""
    exclude_dirs = exclude_dirs or ["organized", "invalid", "duplicates"]
    all_files = []

    for ext in VALID_EXTENSIONS:
        for filepath in root_dir.rglob(f"*{ext}"):
            # Skip excluded directories
            if any(excl in filepath.parts for excl in exclude_dirs):
                continue
            all_files.append(filepath)

    return all_files


def organize(root_dir: Path, clean: bool = False):
    """
    Organize all schematic files into structure type folders.

    Args:
        root_dir: Root datasets directory
        clean: If True, remove existing organized folder first
    """
    organized_dir = root_dir / "organized"
    invalid_dir = root_dir / "_invalid"

    # Clean if requested
    if clean and organized_dir.exists():
        print(f"🧹 Cleaning {organized_dir}...")
        shutil.rmtree(organized_dir)

    # Create directories
    organized_dir.mkdir(exist_ok=True)
    invalid_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("📁 Dataset Auto-Organizer")
    print("=" * 60)
    print(f"Root: {root_dir}")

    # Find all schematic files
    all_files = find_all_schematics(root_dir)
    print(f"\nFound {len(all_files)} schematic files to process")

    # Track stats
    seen_hashes = {}
    stats = defaultdict(int)
    duplicates = 0
    invalid = 0
    skipped = 0

    # Load existing hashes if available
    hash_file = organized_dir / ".hashes.json"
    if hash_file.exists():
        with open(hash_file, "r") as f:
            seen_hashes = json.load(f)
        print(f"Loaded {len(seen_hashes)} existing file hashes")

    # Process files
    for filepath in all_files:
        # Skip already organized files
        if "organized" in filepath.parts:
            skipped += 1
            continue

        # Validate
        if not is_valid_schematic(filepath):
            invalid += 1
            # Move to invalid folder
            dest = invalid_dir / filepath.name
            if not dest.exists():
                shutil.copy2(filepath, dest)
            continue

        # Check for duplicates
        file_hash = get_file_hash(filepath)
        if file_hash in seen_hashes:
            duplicates += 1
            continue
        seen_hashes[file_hash] = str(filepath)

        # Detect structure type
        struct_type = detect_structure_type(filepath.name)
        stats[struct_type] += 1

        # Create type directory
        type_dir = organized_dir / struct_type
        type_dir.mkdir(exist_ok=True)

        # Copy file (handle name collisions)
        dest_path = type_dir / filepath.name
        if dest_path.exists():
            base = filepath.stem
            ext = filepath.suffix
            counter = 1
            while dest_path.exists():
                dest_path = type_dir / f"{base}_{counter}{ext}"
                counter += 1

        shutil.copy2(filepath, dest_path)

    # Save hashes for future runs
    with open(hash_file, "w") as f:
        json.dump(seen_hashes, f, indent=2)

    # Write stats
    total_organized = sum(stats.values())
    stats_file = organized_dir / "dataset_stats.json"
    with open(stats_file, "w") as f:
        json.dump(
            {
                "total": total_organized,
                "duplicates_removed": duplicates,
                "invalid_removed": invalid,
                "skipped_already_organized": skipped,
                "by_type": dict(stats),
            },
            f,
            indent=2,
        )

    # Print summary
    print("\n" + "=" * 60)
    print("✅ ORGANIZATION COMPLETE")
    print("=" * 60)
    print(f"\n📊 Statistics:")
    print(f"  Total organized: {total_organized}")
    print(f"  Duplicates skipped: {duplicates}")
    print(f"  Invalid files: {invalid}")
    print(f"  Already organized: {skipped}")
    print(f"\n📂 By structure type:")

    for struct_type, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {struct_type:15} : {count:4} files")

    print(f"\n📁 Output: {organized_dir}")
    print(f"📝 Stats: {stats_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Auto-organize Minecraft schematic dataset"
    )
    parser.add_argument(
        "--clean", "-c", action="store_true", help="Clean and re-organize everything"
    )
    parser.add_argument(
        "--dir", "-d", type=str, default=None, help="Root datasets directory"
    )
    args = parser.parse_args()

    # Determine root directory
    if args.dir:
        root_dir = Path(args.dir)
    else:
        # Default: script location (datasets folder)
        root_dir = Path(__file__).parent

    if not root_dir.exists():
        print(f"Directory not found: {root_dir}")
        sys.exit(1)

    organize(root_dir, clean=args.clean)


if __name__ == "__main__":
    main()
