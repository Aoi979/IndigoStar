#include "cuda_fp16.h"

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n)                                                 \
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
// ca(cache all, L1 + L2): support 4, 8, 16 bytes, cg(cache global, L2): only
// support 16 bytes.
#undef CP_ASYNC_CA
#undef CP_ASYNC_CG
#define CP_ASYNC_CA(dst, src, bytes)                                           \
  asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(   \
                   ::smem_addr(dst)),                                          \
               "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes)                                           \
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(   \
                   ::smem_addr(dst)),                                          \
               "l"(src), "n"(bytes))

#define CP_ASYNC_CA_4B(dst, src)                                               \
  asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 4;\n" ::"r"(    \
                   ::smem_addr(dst)),                                          \
               "l"(src))

__device__ __forceinline__ unsigned smem_addr(const void *ptr) {
  return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

#include <cuda_runtime.h>
#include <type_traits>

template <typename Shape_MNK> struct Buffer {
  half A[Shape_MNK::M * Shape_MNK::K];
  half B[Shape_MNK::K * Shape_MNK::N];
};
template <typename Shape_MNK, int Stages> struct HgemmSharedStorage {
  Buffer<Shape_MNK> buffer[Stages];
};

template <typename Shape_MNK, int kStages>
__global__ void hgemm_f16f16f16_kernel(half *A, half *B, half *C, int M, int N,
                                       int K) {
  constexpr int kCtaM = Shape_MNK::M;
  constexpr int kCtaN = Shape_MNK::N;
  constexpr int kCtaK = Shape_MNK::K;

  constexpr int kWarpsM = 2;
  constexpr int kWarpsN = 2;
  constexpr int kWarpsK = 1;
  constexpr int kWarps = kWarpsM * kWarpsN * kWarpsK;
  constexpr int kWarpSize = 32;
  constexpr int kThreads = kWarps * kWarpSize;

  constexpr int kSmemStrideA = kCtaK;
  constexpr int kSmemStrideB = kCtaK;

  constexpr int Tiled_MMA_M = 32;
  constexpr int Tiled_MMA_N = 32;
  constexpr int Tiled_MMA_K = 16;

  extern __shared__ char shared_memory[];
  using MainLoopSharedStorage = HgemmSharedStorage<Shape_MNK, kStages>;
  MainLoopSharedStorage *smem =
      reinterpret_cast<MainLoopSharedStorage *>(shared_memory);

  int StrideA = K;
  int StrideB = N;
  int StrideC = N;

  const half *gA_base = A + blockIdx.x * kCtaM * StrideA;
  const half *gB_base = B + blockIdx.y * kCtaK * StrideB;

  half *gC = C + blockIdx.x * kCtaM * StrideC + blockIdx.y * kCtaN;

  int tid = threadIdx.x;
  int warp_id = tid / kWarpSize;
  int warp_row = warp_id / kWarpsN;
  int warp_col = warp_id % kWarpsN;

  constexpr int K_TILE_MAX = K / kCtaK;
  constexpr int K_BLOCK_MAX = kCtaK / Tiled_MMA_K;
  constexpr int K_PIPE_MAX = kStages;

  half tCrC[2][2][2];
  half tCrA[4][2];
  half tCrB[2][2][2];

  int lane_id = tid % kWarpSize;
  int tA_row = tid / 8;
  int tA_col = tid % 8;

  int tB_row = tid / 16;
  int tB_col = tid % 16;

  int k_tiles_to_issue = K_TILE_MAX;
  int k_tiles_to_compute = K_TILE_MAX;
  int k_tile_next = 0;

#pragma unroll
  for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe) {
    const half *gA = gA_base + k_tile_next * kCtaK;
    const half *gB = gB_base + k_tile_next * kCtaK * StrideB;
    CP_ASYNC_CA(&smem->buffer[k_pipe].A[tA_row * kSmemStrideA + tA_col],
                &gA[tA_row * StrideA + tA_col], 16);
  }
}