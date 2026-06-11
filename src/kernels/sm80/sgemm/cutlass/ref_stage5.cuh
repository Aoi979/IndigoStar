#pragma once

#include <cuda_runtime.h>

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"

namespace cutlass_ref {

using Element = float;
using Layout = cutlass::layout::RowMajor;

using Sgemm128x128x8Stage5 = cutlass::gemm::device::Gemm<
    Element, Layout, Element, Layout, Element, Layout, Element,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<Element, 1, Element, Element>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 5, 1, 1,
    false, cutlass::arch::OpMultiplyAdd>;

inline cutlass::Status launch_sgemm_128x128x8stage5(float *A, float *B,
                                                    float *C, int M, int N,
                                                    int K,
                                                    cudaStream_t stream = 0) {
  Sgemm128x128x8Stage5 gemm;

  typename Sgemm128x128x8Stage5::Arguments const args(
      {M, N, K}, {A, K}, {B, N}, {C, N}, {C, N}, {1.0f, 0.0f});

  cutlass::Status status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  return gemm(args, nullptr, stream);
}

} // namespace cutlass_ref
