#!/usr/bin/env python3
"""Generate a Markdown report of CUDA kernel resource usage from build artifacts."""

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class KernelInfo:
    name: str
    arch: str
    reg: int = 0
    shared: int = 0
    local: int = 0
    constant: int = 0
    stack: int = 0
    texture: int = 0
    surface: int = 0
    sampler: int = 0


def find_object_files(build_dir: Path, target: Optional[str] = None) -> List[Path]:
    """Find object files in the build directory.

    If target is provided, only scan CMakeFiles/<target>.dir/ and exclude
    cmake_device_link.o to avoid duplicate kernel entries.
    """
    objs: List[Path] = []
    if target:
        target_dir = build_dir / "CMakeFiles" / f"{target}.dir"
        if target_dir.exists():
            for ext in ("*.o", "*.obj"):
                objs.extend(target_dir.rglob(ext))
            # Exclude device-link object to avoid duplicates
            objs = [o for o in objs if o.name != "cmake_device_link.o"]
        else:
            # Fallback to full recursive scan
            for ext in ("*.o", "*.obj"):
                objs.extend(build_dir.rglob(ext))
    else:
        for ext in ("*.o", "*.obj"):
            objs.extend(build_dir.rglob(ext))
    return objs


def run_cuobjdump(obj: Path) -> Optional[str]:
    """Run cuobjdump -res-usage on an object file and return stdout."""
    try:
        result = subprocess.run(
            ["cuobjdump", "-res-usage", str(obj)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        return result.stdout + result.stderr
    except FileNotFoundError:
        print("Error: cuobjdump not found. Make sure CUDA toolkit is in PATH.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error running cuobjdump on {obj}: {e}", file=sys.stderr)
        return None


def demangle_symbol(name: str) -> str:
    """Demangle a C++ symbol using c++filt."""
    try:
        result = subprocess.run(
            ["c++filt", name],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        pass
    return name


def parse_cuobjdump_output(text: str) -> List[KernelInfo]:
    """Parse cuobjdump -res-usage output into KernelInfo objects."""
    kernels = []
    lines = text.splitlines()
    current_arch = "unknown"
    i = 0

    while i < len(lines):
        line = lines[i]

        # Detect architecture
        arch_match = re.search(r"arch\s*=\s*(sm_\d+)", line)
        if arch_match:
            current_arch = arch_match.group(1)

        # Detect function entry
        func_match = re.match(r"^\s*Function\s+(.+):\s*$", line)
        if func_match:
            raw_name = func_match.group(1).strip()
            # Skip "Common:" pseudo-function
            if raw_name.lower() == "common":
                i += 1
                continue

            kernel = KernelInfo(name=demangle_symbol(raw_name), arch=current_arch)
            i += 1
            # Read indented resource lines until next Function or non-indented line
            while i < len(lines):
                next_line = lines[i]
                if re.match(r"^\s*Function\s+(.+):\s*$", next_line):
                    break
                if not next_line.startswith(" ") and not next_line.startswith("\t"):
                    break
                # Parse key:value pairs
                kv_match = re.findall(r"(\w+(?:\[\d+\])?):(\d+)", next_line)
                for key, val in kv_match:
                    v = int(val)
                    key_lower = key.lower()
                    if key_lower == "reg":
                        kernel.reg = v
                    elif key_lower == "shared":
                        kernel.shared = v
                    elif key_lower == "local":
                        kernel.local = v
                    elif key_lower.startswith("constant"):
                        kernel.constant = v
                    elif key_lower == "stack":
                        kernel.stack = v
                    elif key_lower == "texture":
                        kernel.texture = v
                    elif key_lower == "surface":
                        kernel.surface = v
                    elif key_lower == "sampler":
                        kernel.sampler = v
                i += 1
            kernels.append(kernel)
            continue

        i += 1

    return kernels


def generate_markdown(kernels: List[KernelInfo], output: Path) -> None:
    """Write kernel resource usage as a Markdown table."""
    if not kernels:
        content = "# CUDA Kernel Resource Usage Report\n\nNo kernel resource information found.\n"
        output.write_text(content, encoding="utf-8")
        return

    lines = [
        "# CUDA Kernel Resource Usage Report",
        "",
        "Generated automatically after each build.",
        "",
        "| Kernel | Arch | REG | Shared Mem | Local Mem | Constant Mem | Stack |",
        "|--------|------|-----|------------|-----------|--------------|-------|",
    ]

    for k in kernels:
        lines.append(
            f"| `{k.name}` | {k.arch} | {k.reg} | {k.shared} | {k.local} | {k.constant} | {k.stack} |"
        )

    lines.append("")
    lines.append("## Legend")
    lines.append("")
    lines.append("- **REG**: Number of registers used per thread")
    lines.append("- **Shared Mem**: Static shared memory usage in bytes")
    lines.append("- **Local Mem**: Local memory usage in bytes")
    lines.append("- **Constant Mem**: Constant memory usage in bytes")
    lines.append("- **Stack**: Stack memory usage in bytes")
    lines.append("")

    output.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate CUDA kernel resource report")
    parser.add_argument("--build-dir", required=True, help="CMake build directory")
    parser.add_argument("--output", required=True, help="Output Markdown file path")
    parser.add_argument("--target", default=None, help="CMake target name to filter object files")
    args = parser.parse_args()

    build_dir = Path(args.build_dir)
    output = Path(args.output)

    if not build_dir.exists():
        print(f"Build directory {build_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    objs = find_object_files(build_dir, args.target)
    all_kernels: List[KernelInfo] = []

    for obj in objs:
        text = run_cuobjdump(obj)
        if text:
            kernels = parse_cuobjdump_output(text)
            all_kernels.extend(kernels)

    # Sort by name for stable output
    all_kernels.sort(key=lambda k: k.name)
    generate_markdown(all_kernels, output)
    print(f"Kernel resource report written to: {output}")


if __name__ == "__main__":
    main()
