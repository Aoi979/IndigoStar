#!/bin/bash
# Auto-detect GPU architecture and run verification for all locally-available kernels.
# Usage: ./scripts/verify.sh [SIZE]
#   SIZE: matrix size M=N=K (default: 512)
# Example:
#   ./scripts/verify.sh
#   ./scripts/verify.sh 1024

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
INDIGO_STAR="$PROJECT_DIR/build/indigo_star"

if [ ! -f "$INDIGO_STAR" ]; then
    echo "Error: $INDIGO_STAR not found. Run ./scripts/build.sh first." >&2
    exit 1
fi

# Detect GPU compute capability
detect_arch() {
    if ! command -v nvidia-smi &>/dev/null; then
        echo "Error: nvidia-smi not found." >&2
        exit 1
    fi
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.'
}

cap=$(detect_arch)
echo "Detected GPU compute capability: $cap"
echo ""

SIZE="${1:-512}"
echo "Verify parameters: size=$SIZE"
echo ""

case "$cap" in
    89|80)
        # RTX 4060 / A100: verify all SM80 SGEMM + HGEMM kernels
        echo "Verifying all SM80 kernels (SGEMM + HGEMM) ..."
        echo ""
        "$INDIGO_STAR" --verify --size "$SIZE" \
            --kernel sgemm-custom \
            --kernel sgemm-naive \
            --kernel sgemm-external-db \
            --kernel sgemm-external-nodb \
            --kernel sgemm-cutlass-like-s5 \
            --kernel sgemm-cutlass-like-s5-1cta \
            --kernel sgemm-cutlass-like-s5-warporder \
            --kernel sgemm-cutlass-like-s5-schedule \
            --kernel sgemm-cutlass-like-s5-copyorder \
            --kernel sgemm-cutlass-like-s5-mmaorder \
            --kernel sgemm-cutlass-ref-s5 \
            --kernel hgemm-cute \
            --kernel hgemm-cute-noreg \
            --kernel hgemm-cutlass-sm80
        ;;

    90)
        # H100: verify all SM90 HGEMM kernels
        echo "Verifying all SM90 kernels (HGEMM) ..."
        echo ""
        "$INDIGO_STAR" --verify --size "$SIZE" \
            --kernel hgemm-sm90-pingpong \
            --kernel hgemm-cutlass-sm90-pp \
            --kernel hgemm-cutlass-sm90-coop
        ;;

    *)
        echo "Error: Unknown compute capability '$cap'." >&2
        exit 1
        ;;
esac
