#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace sm90_hgemm_cooperative {

inline cudaError_t launch_hgemm_128x128x64_cooperative(
    half const *A, half const *B, half *C, int M, int N, int K,
    cudaStream_t stream = 0) {
  // TODO: implement handwritten SM90 cooperative kernel here.
  // Currently returns not-supported so the build stays clean.
  (void)A;
  (void)B;
  (void)C;
  (void)M;
  (void)N;
  (void)K;
  (void)stream;
  return cudaErrorNotSupported;
}

} // namespace sm90_hgemm_cooperative
