import os
import glob
import argparse
import sys

try:
    from colorama import init, Fore, Style
    init()
except ImportError:
    print("Installing colorama for colored output...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "colorama"])
    from colorama import init, Fore, Style
    init()


def create_template(input_path):
    """Create a template input file if it doesn't exist."""
    template = """# Add file paths below, one per line
# Lines starting with '#' are comments and will be ignored
# Blank lines are ignored
# Glob patterns like src/*.py are supported

# Example:
# src/main.py
# src/utils.py
# lib/*.js
"""
    with open(input_path, "w", encoding="utf-8") as f:
        f.write(template)
    print(f"{Fore.YELLOW}Created template '{input_path}'. Add your filepaths and run again.{Style.RESET_ALL}")


def is_binary(filepath):
    """Check if a file is binary."""
    try:
        with open(filepath, "rb") as f:
            chunk = f.read(8192)
            return b"\x00" in chunk
    except Exception:
        return False


def escape_backticks(content):
    """Escape triple backticks inside file content."""
    return content.replace("```", "` ` `")


def resolve_filepaths(raw_paths):
    """Resolve glob patterns and remove duplicates."""
    resolved = []
    seen = set()
    for path in raw_paths:
        expanded = glob.glob(path, recursive=True)
        if not expanded:
            if path not in seen:
                resolved.append(path)
                seen.add(path)
        else:
            for p in sorted(expanded):
                normalized = os.path.normpath(p)
                if normalized not in seen and os.path.isfile(normalized):
                    resolved.append(normalized)
                    seen.add(normalized)
    return resolved


def count_lines(content):
    """Count lines in content."""
    if not content:
        return 0
    return content.count("\n") + (0 if content.endswith("\n") else 1)


def get_file_size(filepath):
    """Get file size in human readable format."""
    size = os.path.getsize(filepath)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def write_entry(file_handle, filepath, content, is_first):
    """Write a single file entry to output."""
    if not is_first:
        file_handle.write("\n")

    file_handle.write(f"{filepath}:\n")
    file_handle.write("```\n")
    file_handle.write(escape_backticks(content))
    if not content.endswith("\n"):
        file_handle.write("\n")
    file_handle.write("```\n")


def main():
    parser = argparse.ArgumentParser(description="Combine multiple files into formatted output.")
    parser.add_argument("-i", "--input", default="input.txt", help="Input file path (default: input.txt)")
    parser.add_argument("-o", "--output", default="output.txt", help="Output file path or directory (default: output.txt)")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    # Handle output as directory
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, "output.txt")

    # Check input file
    if not os.path.exists(input_path):
        create_template(input_path)
        return

    # Read and parse input file
    with open(input_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    raw_paths = [line.strip() for line in all_lines if line.strip() and not line.strip().startswith("#")]
    comment_lines = sum(1 for line in all_lines if line.strip().startswith("#"))
    blank_lines = sum(1 for line in all_lines if not line.strip())

    if not raw_paths:
        print(f"{Fore.RED}Error: '{input_path}' has no filepaths.{Style.RESET_ALL}")
        return

    filepaths = resolve_filepaths(raw_paths)

    if not filepaths:
        print(f"{Fore.RED}Error: No valid filepaths found.{Style.RESET_ALL}")
        return

    # Stats tracking
    total_mentioned = len(raw_paths)
    total_resolved = len(filepaths)
    success = 0
    skipped = 0
    total_content_lines = 0
    total_output_lines = 0
    total_content_size = 0

    skip_reasons = {
        "not_found": 0,
        "directory": 0,
        "binary": 0,
        "unreadable": 0,
    }

    file_stats = []  # [(filepath, lines, size)]
    extensions = {}  # {ext: count}

    # Header
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  📂 File Combiner{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"  Input:  {Fore.YELLOW}{input_path}{Style.RESET_ALL}")
    print(f"  Output: {Fore.YELLOW}{output_path}{Style.RESET_ALL}")
    print(f"  Paths mentioned: {Fore.CYAN}{total_mentioned}{Style.RESET_ALL}")
    print(f"  Paths resolved:  {Fore.CYAN}{total_resolved}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")

    print(f"{Fore.CYAN}Processing files...{Style.RESET_ALL}\n")

    # Find longest filename for alignment
    max_name_len = max(len(os.path.basename(fp)) for fp in filepaths)
    max_name_len = min(max_name_len, 40)  # Cap at 40

    with open(output_path, "w", encoding="utf-8") as out:
        is_first = True

        for filepath in filepaths:
            display_name = os.path.basename(filepath)
            padded_name = filepath.ljust(max_name_len + 10)

            # Validation checks
            if not os.path.exists(filepath):
                print(f"  {Fore.RED}[SKIP]{Style.RESET_ALL} {padded_name} {Fore.RED}Not found{Style.RESET_ALL}")
                skipped += 1
                skip_reasons["not_found"] += 1
                continue

            if os.path.isdir(filepath):
                print(f"  {Fore.RED}[SKIP]{Style.RESET_ALL} {padded_name} {Fore.RED}Is a directory{Style.RESET_ALL}")
                skipped += 1
                skip_reasons["directory"] += 1
                continue

            if is_binary(filepath):
                print(f"  {Fore.RED}[SKIP]{Style.RESET_ALL} {padded_name} {Fore.RED}Binary file{Style.RESET_ALL}")
                skipped += 1
                skip_reasons["binary"] += 1
                continue

            # Read file
            content = None
            for encoding in ("utf-8", "latin-1"):
                try:
                    with open(filepath, "r", encoding=encoding) as f:
                        content = f.read()
                    break
                except (UnicodeDecodeError, Exception):
                    continue

            if content is None:
                print(f"  {Fore.RED}[SKIP]{Style.RESET_ALL} {padded_name} {Fore.RED}Could not read{Style.RESET_ALL}")
                skipped += 1
                skip_reasons["unreadable"] += 1
                continue

            file_lines = count_lines(content)
            file_size = get_file_size(filepath)
            file_size_bytes = os.path.getsize(filepath)
            total_content_lines += file_lines
            total_content_size += file_size_bytes

            # Track extension
            _, ext = os.path.splitext(filepath)
            ext = ext if ext else "(no ext)"
            extensions[ext] = extensions.get(ext, 0) + 1

            # Track per-file stats
            file_stats.append((filepath, file_lines, file_size))

            write_entry(out, filepath, content, is_first)
            is_first = False

            print(f"  {Fore.GREEN}[OK]{Style.RESET_ALL}   {padded_name} {Fore.CYAN}{file_lines:>6} lines{Style.RESET_ALL}  {Fore.MAGENTA}{file_size:>10}{Style.RESET_ALL}")
            success += 1

    # Count output file lines
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            total_output_lines = sum(1 for _ in f)
        output_size = get_file_size(output_path)
    else:
        total_output_lines = 0
        output_size = "0 B"

    # ============================================
    # DETAILED SUMMARY
    # ============================================

    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  📊 Summary{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")

    # Input stats
    print(f"\n  {Fore.YELLOW}📋 Input ({input_path}):{Style.RESET_ALL}")
    print(f"     Total lines in input:    {len(all_lines)}")
    print(f"     File paths mentioned:    {total_mentioned}")
    print(f"     Paths after resolving:   {total_resolved}")
    print(f"     Comment lines:           {comment_lines}")
    print(f"     Blank lines:             {blank_lines}")

    # Processing stats
    print(f"\n  {Fore.GREEN}✅ Processing:{Style.RESET_ALL}")
    print(f"     Successfully combined:   {Fore.GREEN}{success}{Style.RESET_ALL}")
    print(f"     Skipped:                 {Fore.RED}{skipped}{Style.RESET_ALL}")
    if skipped > 0:
        if skip_reasons["not_found"]:
            print(f"       └─ Not found:          {skip_reasons['not_found']}")
        if skip_reasons["directory"]:
            print(f"       └─ Directories:        {skip_reasons['directory']}")
        if skip_reasons["binary"]:
            print(f"       └─ Binary files:       {skip_reasons['binary']}")
        if skip_reasons["unreadable"]:
            print(f"       └─ Unreadable:         {skip_reasons['unreadable']}")

    # Output stats
    print(f"\n  {Fore.YELLOW}📄 Output ({output_path}):{Style.RESET_ALL}")
    print(f"     Total lines in output:   {Fore.CYAN}{total_output_lines}{Style.RESET_ALL}")
    print(f"     Content lines:           {total_content_lines}")
    print(f"     Formatting lines:        {total_output_lines - total_content_lines}")
    print(f"     Output file size:        {Fore.MAGENTA}{output_size}{Style.RESET_ALL}")

    # File type breakdown
    if extensions:
        print(f"\n  {Fore.YELLOW}📁 File types:{Style.RESET_ALL}")
        for ext, count in sorted(extensions.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * count
            print(f"     {ext:<12} {count:>3} file(s)  {Fore.CYAN}{bar}{Style.RESET_ALL}")

    # Largest / Smallest files
    if file_stats:
        sorted_by_lines = sorted(file_stats, key=lambda x: x[1], reverse=True)

        print(f"\n  {Fore.YELLOW}📏 File sizes:{Style.RESET_ALL}")
        print(f"     Largest:   {Fore.CYAN}{sorted_by_lines[0][1]:>6} lines{Style.RESET_ALL}  {sorted_by_lines[0][0]}")
        print(f"     Smallest:  {Fore.CYAN}{sorted_by_lines[-1][1]:>6} lines{Style.RESET_ALL}  {sorted_by_lines[-1][0]}")
        avg_lines = total_content_lines // success if success > 0 else 0
        print(f"     Average:   {Fore.CYAN}{avg_lines:>6} lines{Style.RESET_ALL}")

    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}  ✨ Done!{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()