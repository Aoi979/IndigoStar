#!/bin/bash
# Auto-detect GPU architecture and run benchmark for all locally-available kernels.
# Usage: ./scripts/bench.sh [SIZE] [ITERS] [WARMUP]
#   SIZE   : matrix size M=N=K (default: 1024)
#   ITERS  : benchmark iterations (default: 100)
#   WARMUP : warmup iterations (default: 10)
# Examples:
#   ./scripts/bench.sh
#   ./scripts/bench.sh 2048
#   ./scripts/bench.sh 2048 200 20

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

# Parse arguments
SIZE="${1:-1024}"
ITERS="${2:-100}"
WARMUP="${3:-10}"

if [ "$WARMUP" -lt 1 ]; then
    WARMUP=1
fi

echo "Benchmark parameters: size=$SIZE, iters=$ITERS, warmup=$WARMUP"
echo ""

case "$cap" in
    89|80)
        # RTX 4060 / A100: all SM80 SGEMM + HGEMM kernels
        # Auto-baseline: sgemm-cublas, hgemm-cublas-fp16acc
        # Explicit fp32acc baseline for comparison
        echo "Running all SM80 kernels (SGEMM + HGEMM) ..."
        echo ""
        "$INDIGO_STAR" bench \
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
            --kernel hgemm-cutlass-sm80 \
            --kernel hgemm-cublas-fp32acc \
            --size "$SIZE" --iters "$ITERS" --warmup "$WARMUP"
        ;;

    90)
        # H100: all SM90 HGEMM kernels
        # Auto-baseline: hgemm-cublas-fp16acc
        # Explicit fp32acc baseline for comparison
        echo "Running all SM90 kernels (HGEMM) ..."
        echo ""
        "$INDIGO_STAR" bench \
            --kernel hgemm-sm90-pingpong \
            --kernel hgemm-cutlass-sm90-pp \
            --kernel hgemm-cutlass-sm90-coop \
            --kernel hgemm-cublas-fp32acc \
            --size "$SIZE" --iters "$ITERS" --warmup "$WARMUP"
        ;;

    *)
        echo "Error: Unknown compute capability '$cap'." >&2
        exit 1
        ;;
esac
