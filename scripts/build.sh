#!/bin/bash
# Auto-detect GPU architecture and build the project.
# Usage: ./scripts/build.sh
# Supported: SM80 (A100), SM89 (RTX 4060), SM90 (H100)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"

# Detect GPU compute capability
detect_arch() {
    if ! command -v nvidia-smi &>/dev/null; then
        echo "Error: nvidia-smi not found. Are you on a machine with NVIDIA GPU?" >&2
        exit 1
    fi
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.'
}

cap=$(detect_arch)
echo "Detected GPU compute capability: $cap"

case "$cap" in
    89)
        CMAKE_ARCH="89"
        MACHINE="RTX 4060 (Ada, SM89)"
        ;;
    80)
        CMAKE_ARCH="80"
        MACHINE="A100 (Ampere, SM80)"
        ;;
    90)
        CMAKE_ARCH="90a"
        MACHINE="H100 (Hopper, SM90)"
        ;;
    *)
        echo "Error: Unknown compute capability '$cap'." >&2
        echo "Supported: 80 (A100), 89 (RTX 4060), 90 (H100)" >&2
        exit 1
        ;;
esac

echo "Machine:    $MACHINE"
echo "CMake arch: $CMAKE_ARCH"
echo ""

# Clean old build to avoid mixing architectures
if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning old build directory ..."
    rm -rf "$BUILD_DIR"
fi

echo "Configuring ..."
cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" -DCMAKE_CUDA_ARCHITECTURES="$CMAKE_ARCH"

echo ""
echo "Building ..."
cmake --build "$BUILD_DIR" -j"$(nproc)"

echo ""
echo "Build complete: $BUILD_DIR/indigo_star"
echo ""
echo "Run benchmark:  ./scripts/bench.sh"
echo "Run verify:     ./scripts/verify.sh"
echo "Run help:       $BUILD_DIR/indigo_star --help"
