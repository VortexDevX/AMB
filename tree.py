import os
import sys
import fnmatch
import json
from io import StringIO

try:
    from colorama import init, Fore, Style
    init()
except ImportError:
    print("Installing colorama for colored output...")
    import subprocess
    subprocess.check_call(["pip", "install", "colorama"])
    from colorama import init, Fore, Style
    init()

# ============================================
# IGNORE CONFIGURATION
# ============================================

IGNORE = {
    # Version Control
    ".git", ".svn", ".hg",
    
    # Python
    "__pycache__", "venv", "env", ".venv", ".env",
    ".pytest_cache", ".mypy_cache", ".tox", ".coverage",
    ".eggs", "htmlcov",
    
    # JavaScript/Node/React/Next
    "node_modules", ".next", ".nuxt", "out", ".output",
    "coverage", ".cache", ".parcel-cache", ".turbo",
    "dist", "build", ".svelte-kit",
    
    # Flutter/Dart
    ".dart_tool", ".flutter-plugins", ".flutter-plugins-dependencies",
    ".packages",
    
    # Java/Gradle/Maven/Android
    ".gradle", "gradle", "target", "bin",
    ".settings", ".classpath", ".project",
    
    # IDEs
    ".idea", ".vscode", ".eclipse", ".fleet",
    
    # OS
    ".DS_Store", "Thumbs.db", "desktop.ini",
    
    # Other
    "tmp", "temp", ".tmp", "logs",
    
    # Your custom ignores
    "chapters", ".generate_structure.py", "usinglater",
    "chapter_extractor.py"
}

IGNORE_SUBTREE = {
    ".dart_tool", "ephemeral", ".plugin_symlinks", ".symlinks",
    "example", "xcshareddata", "xcuserdata", "project.xcworkspace",
    "Pods", ".kotlin", "generated_plugin_registrant", "cpp_client_wrapper",
}

IGNORE_EXTENSIONS = {
    ".pyc", ".pyo", ".log", ".tmp", ".swp", ".swo",
    ".class", ".o", ".so", ".dylib", ".dll", ".exe",
    ".jar", ".war", ".iml", ".pdb", ".exp", ".lib",
    ".stamp", ".filecache", ".d",
}

# Wildcard patterns (separate from set for proper matching)
IGNORE_WILDCARDS = [
    "*.egg-info",
    "*.swp",
    "*.swo",
    "*~",
]

IGNORE_PATTERNS = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "poetry.lock", "Pipfile.lock", "pubspec.lock",
    ".env.local", ".env.production", ".env.development",
    "generated_plugin_registrant", "GeneratedPluginRegistrant",
    ".flutter-plugins-dependencies",
}

PLATFORM_FOLDERS = {"ios", "android", "macos", "linux", "windows", "web"}

# ============================================
# EMOJIS & COLORS
# ============================================

FOLDER_EMOJI = "📁"
FILE_EMOJI = "📄"
WARNING_EMOJI = "⚠️"

# ============================================
# STATISTICS TRACKER
# ============================================

class Stats:
    def __init__(self):
        self.folders = 0
        self.files = 0
        self.ignored = 0
    
    def reset(self):
        self.folders = 0
        self.files = 0
        self.ignored = 0

stats = Stats()

# ============================================
# GITIGNORE LOADER
# ============================================

def load_gitignore(root_dir):
    """Load patterns from .gitignore file."""
    gitignore_path = os.path.join(root_dir, ".gitignore")
    patterns = []
    
    if os.path.exists(gitignore_path):
        try:
            with open(gitignore_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith("#"):
                        # Remove trailing slashes for directories
                        patterns.append(line.rstrip("/"))
        except Exception:
            pass
    
    return patterns

# ============================================
# IGNORE LOGIC
# ============================================

def should_ignore(name, path, gitignore_patterns=None):
    """Check if file/folder should be ignored."""
    
    # Check subtree ignores
    if name in IGNORE_SUBTREE:
        stats.ignored += 1
        return True
    
    # Check exact name match
    if name in IGNORE:
        stats.ignored += 1
        return True
    
    # Check extension
    _, ext = os.path.splitext(name)
    if ext in IGNORE_EXTENSIONS:
        stats.ignored += 1
        return True
    
    # Check patterns
    if name in IGNORE_PATTERNS:
        stats.ignored += 1
        return True
    
    # Ignore generated files
    if name.startswith("generated_"):
        stats.ignored += 1
        return True
    
    # Check wildcard patterns
    for pattern in IGNORE_WILDCARDS:
        if fnmatch.fnmatch(name, pattern):
            stats.ignored += 1
            return True
    
    # Check gitignore patterns
    if gitignore_patterns:
        for pattern in gitignore_patterns:
            if fnmatch.fnmatch(name, pattern):
                stats.ignored += 1
                return True
            # Also check if pattern matches with wildcard
            if fnmatch.fnmatch(name, f"*{pattern}*"):
                # Only for specific patterns, not too greedy
                pass
    
    return False

# ============================================
# PLATFORM FOLDER DETECTION
# ============================================

def detect_platform_folders(root_dir):
    """Detect which platform folders exist in the project."""
    found = []
    try:
        entries = os.listdir(root_dir)
        for entry in entries:
            if entry in PLATFORM_FOLDERS:
                path = os.path.join(root_dir, entry)
                if os.path.isdir(path):
                    # Count items inside
                    try:
                        count = len(os.listdir(path))
                        found.append((entry, count))
                    except PermissionError:
                        found.append((entry, "?"))
    except PermissionError:
        pass
    return found

# ============================================
# TREE GENERATOR
# ============================================

def generate_tree(root_dir, prefix="", max_depth=None, current_depth=0, 
                  gitignore_patterns=None, collapse_platforms=False, 
                  use_color=True):
    """Generate tree structure."""
    
    if max_depth is not None and current_depth >= max_depth:
        return
    
    try:
        entries = os.listdir(root_dir)
    except PermissionError:
        if use_color:
            print(f"{prefix}    {Fore.RED}{WARNING_EMOJI} [Permission Denied]{Style.RESET_ALL}")
        else:
            print(f"{prefix}    {WARNING_EMOJI} [Permission Denied]")
        return
    
    # Filter ignored entries
    filtered = [e for e in entries if not should_ignore(e, os.path.join(root_dir, e), gitignore_patterns)]
    
    # Separate folders and files
    folders = sorted([e for e in filtered if os.path.isdir(os.path.join(root_dir, e))])
    files = sorted([e for e in filtered if os.path.isfile(os.path.join(root_dir, e))])
    
    # Sort: non-platform folders first, then platform folders, then files
    platform_folders = [f for f in folders if f in PLATFORM_FOLDERS]
    other_folders = [f for f in folders if f not in PLATFORM_FOLDERS]
    
    all_entries = other_folders + platform_folders + files
    
    for idx, entry in enumerate(all_entries):
        path = os.path.join(root_dir, entry)
        is_last = idx == len(all_entries) - 1
        connector = "└── " if is_last else "├── "
        extension_prefix = "    " if is_last else "│   "
        
        if os.path.isdir(path):
            stats.folders += 1
            is_platform = entry in PLATFORM_FOLDERS
            
            # Handle platform folder collapsing
            if is_platform and collapse_platforms:
                try:
                    item_count = len(os.listdir(path))
                except PermissionError:
                    item_count = "?"
                
                if use_color:
                    print(f"{prefix}{connector}{Fore.YELLOW}{FOLDER_EMOJI} {entry}/{Style.RESET_ALL}")
                    print(f"{prefix}{extension_prefix}    {Fore.CYAN}... ({item_count} items){Style.RESET_ALL}")
                else:
                    print(f"{prefix}{connector}{FOLDER_EMOJI} {entry}/")
                    print(f"{prefix}{extension_prefix}    ... ({item_count} items)")
                continue
            
            # Normal folder - print and recurse
            if use_color:
                print(f"{prefix}{connector}{Fore.BLUE}{FOLDER_EMOJI} {entry}/{Style.RESET_ALL}")
            else:
                print(f"{prefix}{connector}{FOLDER_EMOJI} {entry}/")
            
            generate_tree(
                path, 
                prefix + extension_prefix, 
                max_depth, 
                current_depth + 1,
                gitignore_patterns,
                collapse_platforms,
                use_color
            )
        else:
            stats.files += 1
            if use_color:
                print(f"{prefix}{connector}{Fore.GREEN}{FILE_EMOJI} {entry}{Style.RESET_ALL}")
            else:
                print(f"{prefix}{connector}{FILE_EMOJI} {entry}")

# ============================================
# OUTPUT FORMATTERS
# ============================================

def generate_tree_dict(root_dir, max_depth=None, current_depth=0, 
                       gitignore_patterns=None, collapse_platforms=False):
    """Generate tree as dictionary for JSON output."""
    
    result = {
        "name": os.path.basename(root_dir) or root_dir,
        "type": "folder",
        "children": []
    }
    
    if max_depth is not None and current_depth >= max_depth:
        return result
    
    try:
        entries = os.listdir(root_dir)
    except PermissionError:
        result["error"] = "Permission Denied"
        return result
    
    filtered = [e for e in entries if not should_ignore(e, os.path.join(root_dir, e), gitignore_patterns)]
    
    folders = sorted([e for e in filtered if os.path.isdir(os.path.join(root_dir, e))])
    files = sorted([e for e in filtered if os.path.isfile(os.path.join(root_dir, e))])
    
    for folder in folders:
        path = os.path.join(root_dir, folder)
        is_platform = folder in PLATFORM_FOLDERS
        
        if is_platform and collapse_platforms:
            try:
                item_count = len(os.listdir(path))
            except PermissionError:
                item_count = "?"
            result["children"].append({
                "name": folder,
                "type": "folder",
                "collapsed": True,
                "item_count": item_count
            })
        else:
            child = generate_tree_dict(
                path, max_depth, current_depth + 1,
                gitignore_patterns, collapse_platforms
            )
            result["children"].append(child)
    
    for file in files:
        result["children"].append({
            "name": file,
            "type": "file"
        })
    
    return result

def generate_flat_list(root_dir, max_depth=None, current_depth=0,
                       gitignore_patterns=None, collapse_platforms=False,
                       prefix_path=""):
    """Generate flat list of paths."""
    
    lines = []
    
    if max_depth is not None and current_depth >= max_depth:
        return lines
    
    try:
        entries = os.listdir(root_dir)
    except PermissionError:
        return lines
    
    filtered = [e for e in entries if not should_ignore(e, os.path.join(root_dir, e), gitignore_patterns)]
    
    folders = sorted([e for e in filtered if os.path.isdir(os.path.join(root_dir, e))])
    files = sorted([e for e in filtered if os.path.isfile(os.path.join(root_dir, e))])
    
    for folder in folders:
        path = os.path.join(root_dir, folder)
        relative = os.path.join(prefix_path, folder) if prefix_path else folder
        is_platform = folder in PLATFORM_FOLDERS
        
        lines.append(f"{relative}/")
        
        if not (is_platform and collapse_platforms):
            lines.extend(generate_flat_list(
                path, max_depth, current_depth + 1,
                gitignore_patterns, collapse_platforms, relative
            ))
    
    for file in files:
        relative = os.path.join(prefix_path, file) if prefix_path else file
        lines.append(relative)
    
    return lines

# ============================================
# SAVE TO FILE
# ============================================

def save_to_file(root_dir, output_file, max_depth=None, 
                 gitignore_patterns=None, collapse_platforms=False,
                 output_format="tree"):
    """Save structure to file in specified format."""
    
    root_name = os.path.basename(os.path.abspath(root_dir)) or root_dir
    
    if output_format == "json":
        stats.reset()
        tree_dict = generate_tree_dict(
            root_dir, max_depth, 0,
            gitignore_patterns, collapse_platforms
        )
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(tree_dict, f, indent=2)
    
    elif output_format == "list":
        stats.reset()
        lines = generate_flat_list(
            root_dir, max_depth, 0,
            gitignore_patterns, collapse_platforms
        )
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"{root_name}/\n")
            for line in lines:
                f.write(f"{line}\n")
    
    else:  # tree format
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        stats.reset()
        print(f"{FOLDER_EMOJI} {root_name}/")
        generate_tree(
            root_dir, "", max_depth, 0,
            gitignore_patterns, collapse_platforms, 
            use_color=False  # No color in file output
        )
        
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output)
    
    return True

# ============================================
# PROJECT TYPE DETECTION
# ============================================

def detect_project_type(project_path):
    """Detect project type and return info."""
    
    if os.path.exists(os.path.join(project_path, "pubspec.yaml")):
        return "flutter", "Flutter/Dart", 4
    elif os.path.exists(os.path.join(project_path, "package.json")):
        return "node", "Node.js/JavaScript", None
    elif os.path.exists(os.path.join(project_path, "requirements.txt")):
        return "python", "Python", None
    elif os.path.exists(os.path.join(project_path, "setup.py")):
        return "python", "Python", None
    elif os.path.exists(os.path.join(project_path, "Cargo.toml")):
        return "rust", "Rust", None
    elif os.path.exists(os.path.join(project_path, "go.mod")):
        return "go", "Go", None
    elif os.path.exists(os.path.join(project_path, "pom.xml")):
        return "java", "Java/Maven", None
    elif os.path.exists(os.path.join(project_path, "build.gradle")):
        return "java", "Java/Gradle", None
    
    return None, None, None

# ============================================
# PRINT SUMMARY
# ============================================

def print_summary():
    """Print statistics summary."""
    print(f"\n{Fore.CYAN}{'─' * 40}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}📊 Summary:{Style.RESET_ALL}")
    print(f"   {Fore.BLUE}Folders:{Style.RESET_ALL} {stats.folders}")
    print(f"   {Fore.GREEN}Files:{Style.RESET_ALL}   {stats.files}")
    print(f"   {Fore.YELLOW}Ignored:{Style.RESET_ALL} {stats.ignored}")
    print(f"{Fore.CYAN}{'─' * 40}{Style.RESET_ALL}")

# ============================================
# MAIN
# ============================================

def main():
    print(f"\n{Fore.CYAN}🌳 Project Structure Generator{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
    
    # Get project path
    project_path = input(f"{Fore.YELLOW}📂 Enter project path (default: current directory):{Style.RESET_ALL} ").strip() or "."
    
    if not os.path.exists(project_path):
        print(f"{Fore.RED}❌ Error: Path '{project_path}' does not exist!{Style.RESET_ALL}")
        return
    
    # Detect project type
    project_type, project_name, suggested_depth = detect_project_type(project_path)
    
    if project_name:
        print(f"{Fore.GREEN}🎯 Detected: {project_name} project{Style.RESET_ALL}")
    
    # Load gitignore
    gitignore_patterns = load_gitignore(project_path)
    if gitignore_patterns:
        print(f"{Fore.GREEN}📋 Loaded {len(gitignore_patterns)} patterns from .gitignore{Style.RESET_ALL}")
    
    # Get max depth
    depth_prompt = f"{Fore.YELLOW}📏 Max depth (default: "
    if suggested_depth:
        depth_prompt += f"{suggested_depth} for {project_name}"
    else:
        depth_prompt += "unlimited"
    depth_prompt += f"):{Style.RESET_ALL} "
    
    depth_input = input(depth_prompt).strip()
    
    if depth_input.isdigit():
        max_depth = int(depth_input)
    elif suggested_depth:
        max_depth = suggested_depth
        print(f"{Fore.CYAN}   Using suggested depth: {max_depth}{Style.RESET_ALL}")
    else:
        max_depth = None
    
    # Check for platform folders
    collapse_platforms = False
    platform_folders = detect_platform_folders(project_path)
    
    if platform_folders:
        print(f"\n{Fore.YELLOW}🔍 Found platform folders:{Style.RESET_ALL}")
        for folder, count in platform_folders:
            print(f"   {Fore.BLUE}{folder}/{Style.RESET_ALL} ({count} items)")
        
        collapse_input = input(f"{Fore.YELLOW}📦 Collapse platform folders? (Y/n):{Style.RESET_ALL} ").strip().lower()
        collapse_platforms = collapse_input != "n"
    
    # Output format
    print(f"\n{Fore.YELLOW}📄 Output format:{Style.RESET_ALL}")
    print(f"   {Fore.CYAN}1.{Style.RESET_ALL} tree (default)")
    print(f"   {Fore.CYAN}2.{Style.RESET_ALL} json")
    print(f"   {Fore.CYAN}3.{Style.RESET_ALL} list")
    
    format_input = input(f"{Fore.YELLOW}   Choose (1/2/3):{Style.RESET_ALL} ").strip()
    
    format_map = {"1": "tree", "2": "json", "3": "list", "": "tree"}
    output_format = format_map.get(format_input, "tree")
    
    # Save option
    save_option = input(f"\n{Fore.YELLOW}💾 Save to file? (y/N):{Style.RESET_ALL} ").strip().lower()
    
    print(f"\n{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}\n")
    
    root_name = os.path.basename(os.path.abspath(project_path)) or project_path
    stats.reset()
    
    if save_option == "y":
        # Get output filename
        ext_map = {"tree": "md", "json": "json", "list": "txt"}
        default_ext = ext_map.get(output_format, "md")
        default_name = f"structure.{default_ext}"
        
        output_file = input(f"{Fore.YELLOW}📝 Output filename (default: {default_name}):{Style.RESET_ALL} ").strip() or default_name
        
        save_to_file(
            project_path, output_file, max_depth,
            gitignore_patterns, collapse_platforms, output_format
        )
        
        print(f"{Fore.GREEN}✅ Structure saved to: {output_file}{Style.RESET_ALL}")
        
        # Also print to console
        if output_format == "tree":
            print(f"\n{FOLDER_EMOJI} {root_name}/")
            generate_tree(
                project_path, "", max_depth, 0,
                gitignore_patterns, collapse_platforms, use_color=True
            )
    else:
        if output_format == "json":
            tree_dict = generate_tree_dict(
                project_path, max_depth, 0,
                gitignore_patterns, collapse_platforms
            )
            print(json.dumps(tree_dict, indent=2))
        
        elif output_format == "list":
            print(f"{root_name}/")
            lines = generate_flat_list(
                project_path, max_depth, 0,
                gitignore_patterns, collapse_platforms
            )
            for line in lines:
                print(line)
        
        else:  # tree
            print(f"{FOLDER_EMOJI} {root_name}/")
            generate_tree(
                project_path, "", max_depth, 0,
                gitignore_patterns, collapse_platforms, use_color=True
            )
    
    # Print summary
    print_summary()
    
    print(f"\n{Fore.GREEN}✨ Done!{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()