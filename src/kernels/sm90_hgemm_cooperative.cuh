#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cutlass_sm90_hgemm.cuh"

namespace sm90_hgemm_cooperative {

inline cudaError_t launch_hgemm_128x128x64_cooperative(
    half const *A, half const *B, half *C, int M, int N, int K,
    cudaStream_t stream = 0) {
  static_assert(sizeof(half) == sizeof(cutlass::half_t));
  auto const *cutlass_A = reinterpret_cast<cutlass::half_t const *>(A);
  auto const *cutlass_B = reinterpret_cast<cutlass::half_t const *>(B);
  auto *cutlass_C = reinterpret_cast<cutlass::half_t *>(C);

  cutlass::Status status = cutlass_sm90_hgemm::launch_hgemm_cooperative(
      cutlass_A, cutlass_B, cutlass_C, M, N, K, stream);
  return status == cutlass::Status::kSuccess ? cudaSuccess
                                             : cudaErrorInvalidValue;
}

} // namespace sm90_hgemm_cooperative
