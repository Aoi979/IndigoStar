#pragma once

#include <cuda_runtime.h>

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/half.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

namespace cutlass_hgemm {

using Element = cutlass::half_t;
using Layout = cutlass::layout::RowMajor;
using Accumulator = cutlass::half_t;

using HgemmSm80TensorOpA100 = cutlass::gemm::device::Gemm<
    Element, Layout, Element, Layout, Element, Layout, Accumulator,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<Element, 8, Accumulator,
                                                  Accumulator>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 3>;

using HgemmSm80TensorOpLocal = cutlass::gemm::device::Gemm<
    Element, Layout, Element, Layout, Element, Layout, Accumulator,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<Element, 8, Accumulator,
                                                  Accumulator>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 3>;

template <typename GemmOp>
constexpr int shared_storage_bytes() {
  return int(sizeof(typename GemmOp::GemmKernel::SharedStorage));
}

template <typename GemmOp>
inline cutlass::Status launch_hgemm_with_op(Element *A, Element *B,
                                            Element *C, int M, int N, int K,
                                            cudaStream_t stream) {
  GemmOp gemm_op;

  typename GemmOp::Arguments args({M, N, K}, {A, K}, {B, N}, {C, N}, {C, N},
                                  {Accumulator(1.0f), Accumulator(0.0f)});

  cutlass::Status status = gemm_op.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  return gemm_op(args, nullptr, stream);
}

inline cutlass::Status launch_hgemm_sm80_tensorop(Element *A, Element *B,
                                                  Element *C, int M, int N,
                                                  int K,
                                                  cudaStream_t stream = 0) {
  int device = 0;
  int max_dynamic_smem = 0;
  if (cudaGetDevice(&device) == cudaSuccess &&
      cudaDeviceGetAttribute(&max_dynamic_smem,
                             cudaDevAttrMaxSharedMemoryPerBlockOptin,
                             device) == cudaSuccess &&
      max_dynamic_smem >= shared_storage_bytes<HgemmSm80TensorOpA100>()) {
    return launch_hgemm_with_op<HgemmSm80TensorOpA100>(A, B, C, M, N, K,
                                                       stream);
  }

  return launch_hgemm_with_op<HgemmSm80TensorOpLocal>(A, B, C, M, N, K,
                                                      stream);
}

}  // namespace cutlass_hgemm
