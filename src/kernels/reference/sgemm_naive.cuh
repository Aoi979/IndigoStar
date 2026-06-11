#include "../common.hpp"
#include <cstdint>

__global__ void sgemm_naive(float *A, float *B, float *C, int M, int N, int K) {
  int cx = blockIdx.x * blockDim.x + threadIdx.x;
  int cy = blockIdx.y * blockDim.y + threadIdx.y;

  if (cx >= N || cy >= M) {
    return;
  }

  float accumulator = 0.0F;
  for (int i = 0; i < K; i++) {
    accumulator += A[cy * K + i] * B[i * N + cx];
  }

  __syncthreads();
  C[cx + cy * N] = accumulator;
}
