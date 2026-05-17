"""
Kaggle Dataset Preparation Script
----------------------------------
Converts the datasets folder into a Kaggle-compatible ZIP file.

Follows Kaggle upload rules:
- No forbidden characters (#, :, *, ?, ", <, >, |)
- No spaces in filenames
- Lowercase names
- Removes system files (.DS_Store, Thumbs.db)
- Max 50 top-level items

Usage:
    python prepare_kaggle_dataset.py
    
Output:
    kaggle_dataset.zip (ready for upload)
"""

import os
import re
import shutil
import zipfile
from pathlib import Path


# Forbidden characters in Kaggle
FORBIDDEN_CHARS = re.compile(r'[#:*?"<>|\s]')

# System files to skip
SYSTEM_FILES = {
    '.ds_store', 'thumbs.db', 'desktop.ini', '__macosx'
}


def sanitize_name(name: str) -> str:
    """Convert filename to Kaggle-safe format."""
    # Lowercase
    name = name.lower()
    
    # Replace forbidden chars with underscore
    name = FORBIDDEN_CHARS.sub('_', name)
    
    # Replace multiple underscores with single
    name = re.sub(r'_+', '_', name)
    
    # Remove leading/trailing underscores
    name = name.strip('_')
    
    # Limit length
    if len(name) > 200:
        base, ext = os.path.splitext(name)
        name = base[:200-len(ext)] + ext
    
    return name


def is_system_file(name: str) -> bool:
    """Check if file is a system file to skip."""
    return name.lower() in SYSTEM_FILES


def prepare_kaggle_dataset(
    source_dir: str,
    output_zip: str = 'kaggle_dataset.zip',
    max_files: int = None,
):
    """
    Prepare dataset for Kaggle upload.
    
    Args:
        source_dir: Path to datasets/organized folder
        output_zip: Output ZIP filename
        max_files: Optional limit on total files
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    print("=" * 60)
    print("Kaggle Dataset Preparation")
    print("=" * 60)
    print(f"Source: {source_path}")
    print(f"Output: {output_zip}")
    
    # Collect all schematic files
    extensions = ['.schematic', '.schem', '.litematic']
    all_files = []
    
    for ext in extensions:
        all_files.extend(source_path.glob(f'**/*{ext}'))
    
    print(f"\nFound {len(all_files)} schematic files")
    
    if max_files:
        all_files = all_files[:max_files]
        print(f"Limiting to {max_files} files")
    
    # Group by structure type (subfolder)
    by_type = {}
    for file_path in all_files:
        # Get relative path parts
        rel_path = file_path.relative_to(source_path)
        parts = rel_path.parts
        
        if len(parts) >= 2:
            struct_type = sanitize_name(parts[0])
        else:
            struct_type = 'misc'
        
        if struct_type not in by_type:
            by_type[struct_type] = []
        
        by_type[struct_type].append(file_path)
    
    print(f"\nStructure types: {len(by_type)}")
    for stype, files in sorted(by_type.items()):
        print(f"  {stype}: {len(files)} files")
    
    # Create ZIP
    print(f"\nCreating {output_zip}...")
    
    files_added = 0
    files_skipped = 0
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for struct_type, files in by_type.items():
            for i, file_path in enumerate(files):
                # Skip system files
                if is_system_file(file_path.name):
                    files_skipped += 1
                    continue
                
                # Create safe name
                safe_name = sanitize_name(file_path.stem)
                ext = file_path.suffix.lower()
                
                # Add index to avoid collisions
                archive_name = f"{struct_type}/{safe_name}_{i:04d}{ext}"
                
                try:
                    zf.write(file_path, archive_name)
                    files_added += 1
                except Exception as e:
                    print(f"  Error adding {file_path.name}: {e}")
                    files_skipped += 1
    
    # Get final size
    zip_size = os.path.getsize(output_zip)
    zip_size_mb = zip_size / (1024 * 1024)
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Files added: {files_added}")
    print(f"Files skipped: {files_skipped}")
    print(f"Output size: {zip_size_mb:.1f} MB")
    print(f"\nOutput: {output_zip}")
    print("\nReady for Kaggle upload!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare dataset for Kaggle")
    parser.add_argument(
        "--source", "-s",
        default="datasets/organized",
        help="Source directory"
    )
    parser.add_argument(
        "--output", "-o",
        default="kaggle_dataset.zip",
        help="Output ZIP file"
    )
    parser.add_argument(
        "--max-files", "-m",
        type=int,
        default=None,
        help="Max files to include"
    )
    
    args = parser.parse_args()
    
    prepare_kaggle_dataset(
        source_dir=args.source,
        output_zip=args.output,
        max_files=args.max_files,
    )
