#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

BACKTICK_RE = re.compile(r"`([^`]+)`")
OPEN_FENCE_RE = re.compile(r"^\s*```")          # ``` or ```lang
CLOSE_FENCE_RE = re.compile(r"^\s*```\s*$")     # closing ```


def clean_path(s: str) -> str:
    s = s.strip()

    # strip quotes/backticks if they exist
    s = s.strip("`").strip('"').strip("'")

    # remove trailing ":" like "file.py:" (but keep drive letters like "C:")
    if s.endswith(":") and not (len(s) == 2 and s[1] == ":"):
        s = s[:-1].rstrip()

    return s


def extract_path_from_line(line: str) -> str | None:
    line = line.strip()
    if not line or line in {"---", "***", "___"}:
        return None

    # Prefer content inside backticks: ### `path/to/file.ext`
    m = BACKTICK_RE.search(line)
    if m:
        candidate = m.group(1)
    else:
        # Fallback: maybe the line IS the path
        candidate = line

        # remove markdown header markers like "### "
        candidate = re.sub(r"^\s*#+\s*", "", candidate).strip()

    candidate = clean_path(candidate)

    # Heuristic: accept only "path-like" things
    name = Path(candidate).name
    if not candidate:
        return None
    if ("/" not in candidate and "\\" not in candidate) and ("." not in name):
        return None

    return candidate


def parse_markdown_blocks(text: str):
    lines = text.splitlines()
    i, n = 0, len(lines)

    while i < n:
        path = extract_path_from_line(lines[i])
        if not path:
            i += 1
            continue

        # Find the next opening fence after the path line
        j = i + 1
        while j < n and not OPEN_FENCE_RE.match(lines[j]):
            j += 1

        if j >= n:
            raise ValueError(f"Expected ``` after filepath '{path}' (near line {i+1})")

        # Skip opening fence line (may be ```csharp etc.)
        j += 1

        code_lines = []
        while j < n and not CLOSE_FENCE_RE.match(lines[j]):
            code_lines.append(lines[j])
            j += 1

        if j >= n:
            raise ValueError(f"Unclosed ``` block for filepath '{path}' (started near line {i+1})")

        # Consume closing fence
        j += 1

        yield path, "\n".join(code_lines)
        i = j


def main():
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("input.txt")
    text = input_path.read_text(encoding="utf-8")

    for filepath, code in parse_markdown_blocks(text):
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if code and not code.endswith("\n"):
            code += "\n"

        out_path.write_text(code, encoding="utf-8")
        print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())