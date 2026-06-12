#include "cuda_fp16.h"
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ uint32_t smem_addr(const void *ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

#define MMA_1_ROW(m)                                                           \
  mma::m16n8k16_f16f16f16_accum(                                               \
      as_u32(tCrC[m][0][0][0][0]), as_u32(tCrC[m][0][0][1][0]),                \
      as_u32(tCrA[m][k_block][0][0][0]), as_u32(tCrA[m][k_block][0][1][0]),    \
      as_u32(tCrA[m][k_block][1][0][0]), as_u32(tCrA[m][k_block][1][1][0]),    \
      as_u32(tCrB[0][k_block][0][0][0]), as_u32(tCrB[0][k_block][0][1][0]));   \
  mma::m16n8k16_f16f16f16_accum(                                               \
      as_u32(tCrC[m][0][1][0][0]), as_u32(tCrC[m][0][1][1][0]),                \
      as_u32(tCrA[m][k_block][0][0][0]), as_u32(tCrA[m][k_block][0][1][0]),    \
      as_u32(tCrA[m][k_block][1][0][0]), as_u32(tCrA[m][k_block][1][1][0]),    \
      as_u32(tCrB[0][k_block][1][0][0]), as_u32(tCrB[0][k_block][1][1][0]));   \
  mma::m16n8k16_f16f16f16_accum(                                               \
      as_u32(tCrC[m][1][0][0][0]), as_u32(tCrC[m][1][0][1][0]),                \
      as_u32(tCrA[m][k_block][0][0][0]), as_u32(tCrA[m][k_block][0][1][0]),    \
      as_u32(tCrA[m][k_block][1][0][0]), as_u32(tCrA[m][k_block][1][1][0]),    \
      as_u32(tCrB[1][k_block][0][0][0]), as_u32(tCrB[1][k_block][0][1][0]));   \
  mma::m16n8k16_f16f16f16_accum(                                               \
      as_u32(tCrC[m][1][1][0][0]), as_u32(tCrC[m][1][1][1][0]),                \
      as_u32(tCrA[m][k_block][0][0][0]), as_u32(tCrA[m][k_block][0][1][0]),    \
      as_u32(tCrA[m][k_block][1][0][0]), as_u32(tCrA[m][k_block][1][1][0]),    \
      as_u32(tCrB[1][k_block][1][0][0]), as_u32(tCrB[1][k_block][1][1][0]));   \
  mma::m16n8k16_f16f16f16_accum(                                               \
      as_u32(tCrC[m][2][0][0][0]), as_u32(tCrC[m][2][0][1][0]),                \
      as_u32(tCrA[m][k_block][0][0][0]), as_u32(tCrA[m][k_block][0][1][0]),    \
      as_u32(tCrA[m][k_block][1][0][0]), as_u32(tCrA[m][k_block][1][1][0]),    \
      as_u32(tCrB[2][k_block][0][0][0]), as_u32(tCrB[2][k_block][0][1][0]));   \
  mma::m16n8k16_f16f16f16_accum(                                               \
      as_u32(tCrC[m][2][1][0][0]), as_u32(tCrC[m][2][1][1][0]),                \
      as_u32(tCrA[m][k_block][0][0][0]), as_u32(tCrA[m][k_block][0][1][0]),    \
      as_u32(tCrA[m][k_block][1][0][0]), as_u32(tCrA[m][k_block][1][1][0]),    \
      as_u32(tCrB[2][k_block][1][0][0]), as_u32(tCrB[2][k_block][1][1][0]));   \
  mma::m16n8k16_f16f16f16_accum(                                               \
      as_u32(tCrC[m][3][0][0][0]), as_u32(tCrC[m][3][0][1][0]),                \
      as_u32(tCrA[m][k_block][0][0][0]), as_u32(tCrA[m][k_block][0][1][0]),    \
      as_u32(tCrA[m][k_block][1][0][0]), as_u32(tCrA[m][k_block][1][1][0]),    \
      as_u32(tCrB[3][k_block][0][0][0]), as_u32(tCrB[3][k_block][0][1][0]));   \
  mma::m16n8k16_f16f16f16_accum(                                               \
      as_u32(tCrC[m][3][1][0][0]), as_u32(tCrC[m][3][1][1][0]),                \
      as_u32(tCrA[m][k_block][0][0][0]), as_u32(tCrA[m][k_block][0][1][0]),    \
      as_u32(tCrA[m][k_block][1][0][0]), as_u32(tCrA[m][k_block][1][1][0]),    \
      as_u32(tCrB[3][k_block][1][0][0]), as_u32(tCrB[3][k_block][1][1][0]))

#define MMA_1_ROW_FP32(m)                                                      \
  mma::m16n8k16_f32f16f16f32_accum(                                            \
      tCrC[m][0][0][0][0], tCrC[m][0][0][0][1], tCrC[m][0][0][1][0],           \
      tCrC[m][0][0][1][1], as_u32(tCrA[m][k_block][0][0][0]),                  \
      as_u32(tCrA[m][k_block][0][1][0]), as_u32(tCrA[m][k_block][1][0][0]),    \
      as_u32(tCrA[m][k_block][1][1][0]), as_u32(tCrB[0][k_block][0][0][0]),    \
      as_u32(tCrB[0][k_block][0][1][0]));                                      \
  mma::m16n8k16_f32f16f16f32_accum(                                            \
      tCrC[m][0][1][0][0], tCrC[m][0][1][0][1], tCrC[m][0][1][1][0],           \
      tCrC[m][0][1][1][1], as_u32(tCrA[m][k_block][0][0][0]),                  \
      as_u32(tCrA[m][k_block][0][1][0]), as_u32(tCrA[m][k_block][1][0][0]),    \
      as_u32(tCrA[m][k_block][1][1][0]), as_u32(tCrB[0][k_block][1][0][0]),    \
      as_u32(tCrB[0][k_block][1][1][0]));                                      \
  mma::m16n8k16_f32f16f16f32_accum(                                            \
      tCrC[m][1][0][0][0], tCrC[m][1][0][0][1], tCrC[m][1][0][1][0],           \
      tCrC[m][1][0][1][1], as_u32(tCrA[m][k_block][0][0][0]),                  \
      as_u32(tCrA[m][k_block][0][1][0]), as_u32(tCrA[m][k_block][1][0][0]),    \
      as_u32(tCrA[m][k_block][1][1][0]), as_u32(tCrB[1][k_block][0][0][0]),    \
      as_u32(tCrB[1][k_block][0][1][0]));                                      \
  mma::m16n8k16_f32f16f16f32_accum(                                            \
      tCrC[m][1][1][0][0], tCrC[m][1][1][0][1], tCrC[m][1][1][1][0],           \
      tCrC[m][1][1][1][1], as_u32(tCrA[m][k_block][0][0][0]),                  \
      as_u32(tCrA[m][k_block][0][1][0]), as_u32(tCrA[m][k_block][1][0][0]),    \
      as_u32(tCrA[m][k_block][1][1][0]), as_u32(tCrB[1][k_block][1][0][0]),    \
      as_u32(tCrB[1][k_block][1][1][0]));                                      \
  mma::m16n8k16_f32f16f16f32_accum(                                            \
      tCrC[m][2][0][0][0], tCrC[m][2][0][0][1], tCrC[m][2][0][1][0],           \
      tCrC[m][2][0][1][1], as_u32(tCrA[m][k_block][0][0][0]),                  \
      as_u32(tCrA[m][k_block][0][1][0]), as_u32(tCrA[m][k_block][1][0][0]),    \
      as_u32(tCrA[m][k_block][1][1][0]), as_u32(tCrB[2][k_block][0][0][0]),    \
      as_u32(tCrB[2][k_block][0][1][0]));                                      \
  mma::m16n8k16_f32f16f16f32_accum(                                            \
      tCrC[m][2][1][0][0], tCrC[m][2][1][0][1], tCrC[m][2][1][1][0],           \
      tCrC[m][2][1][1][1], as_u32(tCrA[m][k_block][0][0][0]),                  \
      as_u32(tCrA[m][k_block][0][1][0]), as_u32(tCrA[m][k_block][1][0][0]),    \
      as_u32(tCrA[m][k_block][1][1][0]), as_u32(tCrB[2][k_block][1][0][0]),    \
      as_u32(tCrB[2][k_block][1][1][0]));                                      \
  mma::m16n8k16_f32f16f16f32_accum(                                            \
      tCrC[m][3][0][0][0], tCrC[m][3][0][0][1], tCrC[m][3][0][1][0],           \
      tCrC[m][3][0][1][1], as_u32(tCrA[m][k_block][0][0][0]),                  \
      as_u32(tCrA[m][k_block][0][1][0]), as_u32(tCrA[m][k_block][1][0][0]),    \
      as_u32(tCrA[m][k_block][1][1][0]), as_u32(tCrB[3][k_block][0][0][0]),    \
      as_u32(tCrB[3][k_block][0][1][0]));                                      \
  mma::m16n8k16_f32f16f16f32_accum(                                            \
      tCrC[m][3][1][0][0], tCrC[m][3][1][0][1], tCrC[m][3][1][1][0],           \
      tCrC[m][3][1][1][1], as_u32(tCrA[m][k_block][0][0][0]),                  \
      as_u32(tCrA[m][k_block][0][1][0]), as_u32(tCrA[m][k_block][1][0][0]),    \
      as_u32(tCrA[m][k_block][1][1][0]), as_u32(tCrB[3][k_block][1][0][0]),    \
      as_u32(tCrB[3][k_block][1][1][0]))

namespace cp_async {

enum class CacheMode {
  CA, // cache all: L1 + L2
  CG  // cache global: L2 only
};

__device__ __forceinline__ void commit_group() {
  asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void wait_all() {
  asm volatile("cp.async.wait_all;\n" ::);
}

template <int N> __device__ __forceinline__ void wait_group() {
  static_assert(N >= 0 && N <= 7, "cp.async.wait_group N must be in [0, 7]");
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

template <CacheMode Mode, int Bytes>
__device__ __forceinline__ void copy(void *smem_ptr, const void *gmem_ptr) {
  static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                "cp.async.ca supports 4, 8, 16 bytes; cp.async.cg supports "
                "only 16 bytes");

  if constexpr (Mode == CacheMode::CA) {
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
                 :
                 : "r"(smem_addr(smem_ptr)), "l"(gmem_ptr), "n"(Bytes));
  } else {
    static_assert(Bytes == 16, "cp.async.cg only supports 16 bytes");

    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                 :
                 : "r"(smem_addr(smem_ptr)), "l"(gmem_ptr));
  }
}

template <int Bytes>
__device__ __forceinline__ void ca(void *smem_ptr, const void *gmem_ptr) {
  copy<CacheMode::CA, Bytes>(smem_ptr, gmem_ptr);
}

template <int Bytes>
__device__ __forceinline__ void cg(void *smem_ptr, const void *gmem_ptr) {
  copy<CacheMode::CG, Bytes>(smem_ptr, gmem_ptr);
}

} // namespace cp_async

namespace ldsm {

enum class Trans { No, Yes };

constexpr Trans T = Trans::Yes;
constexpr Trans N = Trans::No;

template <Trans kTrans = Trans::No>
__device__ __forceinline__ void x1(uint32_t &d0, const void *smem_ptr) {
  uint32_t addr = smem_addr(smem_ptr);

  if constexpr (kTrans == Trans::No) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 "
                 "{%0}, [%1];\n"
                 : "=r"(d0)
                 : "r"(addr));
  } else {
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 "
                 "{%0}, [%1];\n"
                 : "=r"(d0)
                 : "r"(addr));
  }
}

template <Trans kTrans = Trans::No>
__device__ __forceinline__ void x2(uint32_t &d0, uint32_t &d1,
                                   const void *smem_ptr) {
  uint32_t addr = smem_addr(smem_ptr);

  if constexpr (kTrans == Trans::No) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
                 "{%0, %1}, [%2];\n"
                 : "=r"(d0), "=r"(d1)
                 : "r"(addr));
  } else {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                 "{%0, %1}, [%2];\n"
                 : "=r"(d0), "=r"(d1)
                 : "r"(addr));
  }
}

template <Trans kTrans = Trans::No>
__device__ __forceinline__ void x4(uint32_t &v0v1, uint32_t &v2v3,
                                   uint32_t &v4v5, uint32_t &v6v7,
                                   const void *smem_ptr) {
  uint32_t addr = smem_addr(smem_ptr);

  if constexpr (kTrans == Trans::No) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                 "{%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v0v1), "=r"(v2v3), "=r"(v4v5), "=r"(v6v7)
                 : "r"(addr));
  } else {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                 "{%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v0v1), "=r"(v2v3), "=r"(v4v5), "=r"(v6v7)
                 : "r"(addr));
  }
}

} // namespace ldsm

namespace mma {

__device__ __forceinline__ void
m16n8k16_f16f16f16(uint32_t &d0, uint32_t &d1,

                   uint32_t const &a0, uint32_t const &a1, uint32_t const &a2,
                   uint32_t const &a3,

                   uint32_t const &b0, uint32_t const &b1,

                   uint32_t const &c0, uint32_t const &c1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
               "{%0, %1}, "
               "{%2, %3, %4, %5}, "
               "{%6, %7}, "
               "{%8, %9};\n"
               : "=r"(d0), "=r"(d1)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0),
                 "r"(c1));
}

__device__ __forceinline__ void
m16n8k16_f16f16f16_accum(uint32_t &c0, uint32_t &c1,

                         uint32_t const &a0, uint32_t const &a1,
                         uint32_t const &a2, uint32_t const &a3,

                         uint32_t const &b0, uint32_t const &b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
               "{%0, %1}, "
               "{%2, %3, %4, %5}, "
               "{%6, %7}, "
               "{%0, %1};\n"
               : "+r"(c0), "+r"(c1)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void
m16n8k16_f32f16f16f32_accum(float &c0, float &c1, float &c2, float &c3,

                            uint32_t const &a0, uint32_t const &a1,
                            uint32_t const &a2, uint32_t const &a3,

                            uint32_t const &b0, uint32_t const &b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0, %1, %2, %3}, "
               "{%4, %5, %6, %7}, "
               "{%8, %9}, "
               "{%0, %1, %2, %3};\n"
               : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

} // namespace mma

__device__ __forceinline__ uint32_t &as_u32(half &x) {
  return *reinterpret_cast<uint32_t *>(&x);
}

__device__ __forceinline__ uint32_t pack_f32x2_to_f16x2(float x, float y) {
  __half2 xy = __floats2half2_rn(x, y);
  return *reinterpret_cast<uint32_t *>(&xy);
}

namespace hgemm_smem {

__device__ __forceinline__ int offset_A(int m, int k) {
  int k_vec = (k >> 3) ^ (m & 7);
  return (m << 6) + (k_vec << 3) + (k & 7);
}

__device__ __forceinline__ int offset_B(int n, int k) {
  int n_vec = (n >> 3) ^ (k & 7);
  return (k << 7) + (n_vec << 3) + (n & 7);
}

} // namespace hgemm_smem

template <int RowBlock>
__device__ __forceinline__ void issue_cp_async_A(half *smem_A, const half *gA,
                                                 int tA_row, int tA_col,
                                                 int strideA) {
  constexpr int kElementsPerAccess = 8;
  int row = tA_row + RowBlock * 16;
  int col = tA_col * kElementsPerAccess;
  cp_async::cg<16>(&smem_A[hgemm_smem::offset_A(row, col)],
                   &gA[row * strideA + col]);
}

template <int RowBlock>
__device__ __forceinline__ void issue_cp_async_B(half *smem_B, const half *gB,
                                                 int tB_row, int tB_col,
                                                 int strideB) {
  constexpr int kElementsPerAccess = 8;
  int row = tB_row + RowBlock * 8;
  int col = tB_col * kElementsPerAccess;
  cp_async::cg<16>(&smem_B[hgemm_smem::offset_B(col, row)],
                   &gB[row * strideB + col]);
}

template <typename Shape_MNK> struct Buffer {
  half A[Shape_MNK::M * Shape_MNK::K];
  half B[Shape_MNK::K * Shape_MNK::N];
};
template <typename Shape_MNK, int Stages> struct HgemmSharedStorage {
  Buffer<Shape_MNK> buffer[Stages];
};

struct shape_mnk {
  static constexpr int M = 128;
  static constexpr int N = 128;
  static constexpr int K = 64;
};

// only supports M/N/K that are multiples of 128/128/64 respectively
// only supports CtaM == 128, CtaN == 128, CtaK == 64

template <typename Shape_MNK = shape_mnk, int kStages, int kBlockSwizzle>
__global__ void hgemm_f16f16f32_kernel(half *A, half *B, half *C, int M, int N,
                                       int K) {
  constexpr int kCtaM = Shape_MNK::M; // 128
  constexpr int kCtaN = Shape_MNK::N; // 128
  constexpr int kCtaK = Shape_MNK::K; // 64
  static_assert(kCtaM == 128 && kCtaN == 128 && kCtaK == 64,
                "swizzled shared-memory layout assumes a 128x128x64 CTA");

  constexpr int kWarpsM = 2;
  constexpr int kWarpSize = 32;

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

  int const tile_m_max = (M + kCtaM - 1) / kCtaM;
  int const tile_n_max = (N + kCtaN - 1) / kCtaN;

  int tile_m = blockIdx.x / kBlockSwizzle;
  int tile_n = blockIdx.y * kBlockSwizzle + blockIdx.x % kBlockSwizzle;
  if (tile_m >= tile_m_max || tile_n >= tile_n_max) {
    return;
  }

  const half *gA_base = A + tile_m * kCtaM * StrideA;
  const half *gB_base = B + tile_n * kCtaN;

  half *gC = C + tile_m * kCtaM * StrideC + tile_n * kCtaN;

  int tid = threadIdx.x;
  int warp_id = tid / kWarpSize;

  int const K_TILE_MAX = K / kCtaK;
  constexpr int K_BLOCK_MAX = kCtaK / Tiled_MMA_K;
  constexpr int K_PIPE_MAX = kStages;
  static_assert(K_BLOCK_MAX == 4,
                "mainloop cp.async schedule assumes four MMA K-blocks");

  constexpr int MMA_M = kCtaM / Tiled_MMA_M;
  constexpr int MMA_N = kCtaN / Tiled_MMA_N;
  constexpr int MMA_K = kCtaK / Tiled_MMA_K;

  constexpr int Fragment = 2;
  constexpr int CoreMatrix_M = 2;
  constexpr int CoreMatrix_N = 2;
  constexpr int CoreMatrix_K = 2;

  constexpr int kElementsPerAccess = 8; // half, 16B

  // (MMA_M, MMA_N, CoreMatrix_N, CoreMatrix_M, Fragment)
  // :
  // (8 * MMA_N, 8, 4, 2, 1)

  float tCrC[MMA_M][MMA_N][CoreMatrix_N][CoreMatrix_M][Fragment];
  half tCrA[MMA_M][MMA_K][CoreMatrix_K][CoreMatrix_M][Fragment];
  half tCrB[MMA_N][MMA_K][CoreMatrix_N][CoreMatrix_K][Fragment];

#pragma unroll
  for (int m = 0; m < MMA_M; ++m) {
#pragma unroll
    for (int n = 0; n < MMA_N; ++n) {
#pragma unroll
      for (int cm_n = 0; cm_n < CoreMatrix_N; ++cm_n) {
#pragma unroll
        for (int cm_m = 0; cm_m < CoreMatrix_M; ++cm_m) {
          tCrC[m][n][cm_n][cm_m][0] = 0.0f;
          tCrC[m][n][cm_n][cm_m][1] = 0.0f;
        }
      }
    }
  }

  int lane_id = tid % kWarpSize;

  int tA_row = tid / (kCtaK / kElementsPerAccess); // 8
  int tA_col = tid % (kCtaK / kElementsPerAccess);

  int tB_row = tid / (kCtaN / kElementsPerAccess); // 16
  int tB_col = tid % (kCtaN / kElementsPerAccess);

  int k_tiles_to_issue = K_TILE_MAX;
  int k_tiles_to_compute = K_TILE_MAX;
  int k_tile_next = 0;

#pragma unroll
  for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe) {
    const half *gA = gA_base + k_tile_next * kCtaK;
    const half *gB = gB_base + k_tile_next * kCtaK * StrideB;
    cp_async::cg<16>(
        &smem->buffer[k_pipe].A[hgemm_smem::offset_A(
            tA_row + 0 * 16, tA_col * kElementsPerAccess)],
        &gA[(tA_row + 0 * 16) * StrideA + tA_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].A[hgemm_smem::offset_A(
            tA_row + 1 * 16, tA_col * kElementsPerAccess)],
        &gA[(tA_row + 1 * 16) * StrideA + tA_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].A[hgemm_smem::offset_A(
            tA_row + 2 * 16, tA_col * kElementsPerAccess)],
        &gA[(tA_row + 2 * 16) * StrideA + tA_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].A[hgemm_smem::offset_A(
            tA_row + 3 * 16, tA_col * kElementsPerAccess)],
        &gA[(tA_row + 3 * 16) * StrideA + tA_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].A[hgemm_smem::offset_A(
            tA_row + 4 * 16, tA_col * kElementsPerAccess)],
        &gA[(tA_row + 4 * 16) * StrideA + tA_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].A[hgemm_smem::offset_A(
            tA_row + 5 * 16, tA_col * kElementsPerAccess)],
        &gA[(tA_row + 5 * 16) * StrideA + tA_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].A[hgemm_smem::offset_A(
            tA_row + 6 * 16, tA_col * kElementsPerAccess)],
        &gA[(tA_row + 6 * 16) * StrideA + tA_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].A[hgemm_smem::offset_A(
            tA_row + 7 * 16, tA_col * kElementsPerAccess)],
        &gA[(tA_row + 7 * 16) * StrideA + tA_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].B[hgemm_smem::offset_B(
            tB_col * kElementsPerAccess, tB_row + 0 * 8)],
        &gB[(tB_row + 0 * 8) * StrideB + tB_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].B[hgemm_smem::offset_B(
            tB_col * kElementsPerAccess, tB_row + 1 * 8)],
        &gB[(tB_row + 1 * 8) * StrideB + tB_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].B[hgemm_smem::offset_B(
            tB_col * kElementsPerAccess, tB_row + 2 * 8)],
        &gB[(tB_row + 2 * 8) * StrideB + tB_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].B[hgemm_smem::offset_B(
            tB_col * kElementsPerAccess, tB_row + 3 * 8)],
        &gB[(tB_row + 3 * 8) * StrideB + tB_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].B[hgemm_smem::offset_B(
            tB_col * kElementsPerAccess, tB_row + 4 * 8)],
        &gB[(tB_row + 4 * 8) * StrideB + tB_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].B[hgemm_smem::offset_B(
            tB_col * kElementsPerAccess, tB_row + 5 * 8)],
        &gB[(tB_row + 5 * 8) * StrideB + tB_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].B[hgemm_smem::offset_B(
            tB_col * kElementsPerAccess, tB_row + 6 * 8)],
        &gB[(tB_row + 6 * 8) * StrideB + tB_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].B[hgemm_smem::offset_B(
            tB_col * kElementsPerAccess, tB_row + 7 * 8)],
        &gB[(tB_row + 7 * 8) * StrideB + tB_col * kElementsPerAccess]);

    cp_async::commit_group();
    --k_tiles_to_issue;
    ++k_tile_next;
  }

  int smem_pipe_read = 0;
  int smem_pipe_write = K_PIPE_MAX - 1;

  int warp_m_id = warp_id % kWarpsM;
  int warp_n_id = warp_id / kWarpsM;

  int ldsmx4_row = lane_id % 16;
  int ldsmx4_col = lane_id / 16;

  int ldsmx4T_col = lane_id % 16;
  int ldsmx4T_row = lane_id / 16;

  if constexpr (K_BLOCK_MAX > 1) {
    cp_async::wait_group<K_PIPE_MAX - 2>();
    __syncthreads();

    ldsm::x4<ldsm::N>(as_u32(tCrA[0][0][0][0][0]), as_u32(tCrA[0][0][0][1][0]),
                      as_u32(tCrA[0][0][1][0][0]), as_u32(tCrA[0][0][1][1][0]),
                      &smem->buffer[smem_pipe_read].A[hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 0 * Tiled_MMA_M,
                          0 * Tiled_MMA_K + ldsmx4_col * 8)]);
    ldsm::x4<ldsm::N>(as_u32(tCrA[1][0][0][0][0]), as_u32(tCrA[1][0][0][1][0]),
                      as_u32(tCrA[1][0][1][0][0]), as_u32(tCrA[1][0][1][1][0]),
                      &smem->buffer[smem_pipe_read].A[hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 1 * Tiled_MMA_M,
                          0 * Tiled_MMA_K + ldsmx4_col * 8)]);
    ldsm::x4<ldsm::N>(as_u32(tCrA[2][0][0][0][0]), as_u32(tCrA[2][0][0][1][0]),
                      as_u32(tCrA[2][0][1][0][0]), as_u32(tCrA[2][0][1][1][0]),
                      &smem->buffer[smem_pipe_read].A[hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 2 * Tiled_MMA_M,
                          0 * Tiled_MMA_K + ldsmx4_col * 8)]);
    ldsm::x4<ldsm::N>(as_u32(tCrA[3][0][0][0][0]), as_u32(tCrA[3][0][0][1][0]),
                      as_u32(tCrA[3][0][1][0][0]), as_u32(tCrA[3][0][1][1][0]),
                      &smem->buffer[smem_pipe_read].A[hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 3 * Tiled_MMA_M,
                          0 * Tiled_MMA_K + ldsmx4_col * 8)]);
    ldsm::x4<ldsm::T>(as_u32(tCrB[0][0][0][0][0]), as_u32(tCrB[0][0][0][1][0]),
                      as_u32(tCrB[0][0][1][0][0]), as_u32(tCrB[0][0][1][1][0]),
                      &smem->buffer[smem_pipe_read].B[hgemm_smem::offset_B(
                          warp_n_id * 8 + Tiled_MMA_N * 0 + ldsmx4T_row * 16,
                          ldsmx4T_col + 0 * Tiled_MMA_K)]);
    ldsm::x4<ldsm::T>(as_u32(tCrB[1][0][0][0][0]), as_u32(tCrB[1][0][0][1][0]),
                      as_u32(tCrB[1][0][1][0][0]), as_u32(tCrB[1][0][1][1][0]),
                      &smem->buffer[smem_pipe_read].B[hgemm_smem::offset_B(
                          warp_n_id * 8 + Tiled_MMA_N * 1 + ldsmx4T_row * 16,
                          ldsmx4T_col + 0 * Tiled_MMA_K)]);
    ldsm::x4<ldsm::T>(as_u32(tCrB[2][0][0][0][0]), as_u32(tCrB[2][0][0][1][0]),
                      as_u32(tCrB[2][0][1][0][0]), as_u32(tCrB[2][0][1][1][0]),
                      &smem->buffer[smem_pipe_read].B[hgemm_smem::offset_B(
                          warp_n_id * 8 + Tiled_MMA_N * 2 + ldsmx4T_row * 16,
                          ldsmx4T_col + 0 * Tiled_MMA_K)]);
    ldsm::x4<ldsm::T>(as_u32(tCrB[3][0][0][0][0]), as_u32(tCrB[3][0][0][1][0]),
                      as_u32(tCrB[3][0][1][0][0]), as_u32(tCrB[3][0][1][1][0]),
                      &smem->buffer[smem_pipe_read].B[hgemm_smem::offset_B(
                          warp_n_id * 8 + Tiled_MMA_N * 3 + ldsmx4T_row * 16,
                          ldsmx4T_col + 0 * Tiled_MMA_K)]);
  }
  auto stage_A_p = smem->buffer[smem_pipe_read].A;
  auto stage_B_p = smem->buffer[smem_pipe_read].B;

  while (k_tiles_to_compute > 0) {
#pragma unroll
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
      if (k_tiles_to_issue > 0) {
        const half *gA = gA_base + k_tile_next * kCtaK;
        const half *gB = gB_base + k_tile_next * kCtaK * StrideB;
        half *sA = smem->buffer[smem_pipe_write].A;
        half *sB = smem->buffer[smem_pipe_write].B;

        if (k_block == 0) {
          issue_cp_async_A<0>(sA, gA, tA_row, tA_col, StrideA);
          issue_cp_async_A<1>(sA, gA, tA_row, tA_col, StrideA);
          issue_cp_async_B<0>(sB, gB, tB_row, tB_col, StrideB);
          issue_cp_async_B<1>(sB, gB, tB_row, tB_col, StrideB);
        } else if (k_block == 1) {
          issue_cp_async_A<2>(sA, gA, tA_row, tA_col, StrideA);
          issue_cp_async_A<3>(sA, gA, tA_row, tA_col, StrideA);
          issue_cp_async_B<2>(sB, gB, tB_row, tB_col, StrideB);
          issue_cp_async_B<3>(sB, gB, tB_row, tB_col, StrideB);
        } else if (k_block == 2) {
          issue_cp_async_A<4>(sA, gA, tA_row, tA_col, StrideA);
          issue_cp_async_A<5>(sA, gA, tA_row, tA_col, StrideA);
          issue_cp_async_B<4>(sB, gB, tB_row, tB_col, StrideB);
          issue_cp_async_B<5>(sB, gB, tB_row, tB_col, StrideB);
        } else {
          issue_cp_async_A<6>(sA, gA, tA_row, tA_col, StrideA);
          issue_cp_async_A<7>(sA, gA, tA_row, tA_col, StrideA);
          issue_cp_async_B<6>(sB, gB, tB_row, tB_col, StrideB);
          issue_cp_async_B<7>(sB, gB, tB_row, tB_col, StrideB);
        }
      }

      if (k_block == K_BLOCK_MAX - 1) {
        cp_async::commit_group();
        if (k_tiles_to_issue > 0) {
          --k_tiles_to_issue;
          ++k_tile_next;
        }
        smem_pipe_write = smem_pipe_read;
        smem_pipe_read =
            (smem_pipe_read == K_PIPE_MAX - 1) ? 0 : smem_pipe_read + 1;

        stage_A_p = smem->buffer[smem_pipe_read].A;
        stage_B_p = smem->buffer[smem_pipe_read].B;
        if (k_tiles_to_compute <= K_PIPE_MAX - 1) {
          cp_async::wait_group<0>();
        } else {
          cp_async::wait_group<K_PIPE_MAX - 2>();
        }
        __syncthreads();
      }

      int k_block_next = (k_block + 1) % K_BLOCK_MAX;
      ldsm::x4<ldsm::N>(as_u32(tCrA[0][k_block_next][0][0][0]),
                        as_u32(tCrA[0][k_block_next][0][1][0]),
                        as_u32(tCrA[0][k_block_next][1][0][0]),
                        as_u32(tCrA[0][k_block_next][1][1][0]),
                        &smem->buffer[smem_pipe_read].A[hgemm_smem::offset_A(
                            warp_m_id * 16 + ldsmx4_row + 0 * Tiled_MMA_M,
                            k_block_next * Tiled_MMA_K + ldsmx4_col * 8)]);
      ldsm::x4<ldsm::N>(as_u32(tCrA[1][k_block_next][0][0][0]),
                        as_u32(tCrA[1][k_block_next][0][1][0]),
                        as_u32(tCrA[1][k_block_next][1][0][0]),
                        as_u32(tCrA[1][k_block_next][1][1][0]),
                        &smem->buffer[smem_pipe_read].A[hgemm_smem::offset_A(
                            warp_m_id * 16 + ldsmx4_row + 1 * Tiled_MMA_M,
                            k_block_next * Tiled_MMA_K + ldsmx4_col * 8)]);
      ldsm::x4<ldsm::N>(as_u32(tCrA[2][k_block_next][0][0][0]),
                        as_u32(tCrA[2][k_block_next][0][1][0]),
                        as_u32(tCrA[2][k_block_next][1][0][0]),
                        as_u32(tCrA[2][k_block_next][1][1][0]),
                        &smem->buffer[smem_pipe_read].A[hgemm_smem::offset_A(
                            warp_m_id * 16 + ldsmx4_row + 2 * Tiled_MMA_M,
                            k_block_next * Tiled_MMA_K + ldsmx4_col * 8)]);
      ldsm::x4<ldsm::N>(as_u32(tCrA[3][k_block_next][0][0][0]),
                        as_u32(tCrA[3][k_block_next][0][1][0]),
                        as_u32(tCrA[3][k_block_next][1][0][0]),
                        as_u32(tCrA[3][k_block_next][1][1][0]),
                        &smem->buffer[smem_pipe_read].A[hgemm_smem::offset_A(
                            warp_m_id * 16 + ldsmx4_row + 3 * Tiled_MMA_M,
                            k_block_next * Tiled_MMA_K + ldsmx4_col * 8)]);
      ldsm::x4<ldsm::T>(as_u32(tCrB[0][k_block_next][0][0][0]),
                        as_u32(tCrB[0][k_block_next][0][1][0]),
                        as_u32(tCrB[0][k_block_next][1][0][0]),
                        as_u32(tCrB[0][k_block_next][1][1][0]),
                        &smem->buffer[smem_pipe_read].B[hgemm_smem::offset_B(
                            warp_n_id * 8 + Tiled_MMA_N * 0 + ldsmx4T_row * 16,
                            ldsmx4T_col + k_block_next * Tiled_MMA_K)]);
      ldsm::x4<ldsm::T>(as_u32(tCrB[1][k_block_next][0][0][0]),
                        as_u32(tCrB[1][k_block_next][0][1][0]),
                        as_u32(tCrB[1][k_block_next][1][0][0]),
                        as_u32(tCrB[1][k_block_next][1][1][0]),
                        &smem->buffer[smem_pipe_read].B[hgemm_smem::offset_B(
                            warp_n_id * 8 + Tiled_MMA_N * 1 + ldsmx4T_row * 16,
                            ldsmx4T_col + k_block_next * Tiled_MMA_K)]);
      ldsm::x4<ldsm::T>(as_u32(tCrB[2][k_block_next][0][0][0]),
                        as_u32(tCrB[2][k_block_next][0][1][0]),
                        as_u32(tCrB[2][k_block_next][1][0][0]),
                        as_u32(tCrB[2][k_block_next][1][1][0]),
                        &smem->buffer[smem_pipe_read].B[hgemm_smem::offset_B(
                            warp_n_id * 8 + Tiled_MMA_N * 2 + ldsmx4T_row * 16,
                            ldsmx4T_col + k_block_next * Tiled_MMA_K)]);
      ldsm::x4<ldsm::T>(as_u32(tCrB[3][k_block_next][0][0][0]),
                        as_u32(tCrB[3][k_block_next][0][1][0]),
                        as_u32(tCrB[3][k_block_next][1][0][0]),
                        as_u32(tCrB[3][k_block_next][1][1][0]),
                        &smem->buffer[smem_pipe_read].B[hgemm_smem::offset_B(
                            warp_n_id * 8 + Tiled_MMA_N * 3 + ldsmx4T_row * 16,
                            ldsmx4T_col + k_block_next * Tiled_MMA_K)]);

      MMA_1_ROW_FP32(0);
      MMA_1_ROW_FP32(1);
      MMA_1_ROW_FP32(2);
      MMA_1_ROW_FP32(3);
    }
    --k_tiles_to_compute;
  }

  //
  // Epilogue
  //

  cp_async::wait_all();
  __syncthreads();
  half *sC = reinterpret_cast<half *>(shared_memory);

  int core_matrix_row = lane_id / 4;
  int core_matrix_col = lane_id % 4;

  constexpr int kSmemStrideC = 136;
#pragma unroll
  for (int m = 0; m < MMA_M; ++m) {
    for (int n = 0; n < MMA_N; ++n) {
      *reinterpret_cast<uint32_t *>(
          &sC[(m * Tiled_MMA_M + warp_m_id * 16 + 0 * 8 + core_matrix_row) *
                  kSmemStrideC +
              n * Tiled_MMA_N + warp_n_id * 8 + 0 * 16 + core_matrix_col * 2]) =
          pack_f32x2_to_f16x2(tCrC[m][n][0][0][0], tCrC[m][n][0][0][1]);
      *reinterpret_cast<uint32_t *>(
          &sC[(m * Tiled_MMA_M + warp_m_id * 16 + 1 * 8 + core_matrix_row) *
                  kSmemStrideC +
              n * Tiled_MMA_N + warp_n_id * 8 + 0 * 16 + core_matrix_col * 2]) =
          pack_f32x2_to_f16x2(tCrC[m][n][0][1][0], tCrC[m][n][0][1][1]);

      *reinterpret_cast<uint32_t *>(
          &sC[(m * Tiled_MMA_M + warp_m_id * 16 + 0 * 8 + core_matrix_row) *
                  kSmemStrideC +
              n * Tiled_MMA_N + warp_n_id * 8 + 1 * 16 + core_matrix_col * 2]) =
          pack_f32x2_to_f16x2(tCrC[m][n][1][0][0], tCrC[m][n][1][0][1]);
      *reinterpret_cast<uint32_t *>(
          &sC[(m * Tiled_MMA_M + warp_m_id * 16 + 1 * 8 + core_matrix_row) *
                  kSmemStrideC +
              n * Tiled_MMA_N + warp_n_id * 8 + 1 * 16 + core_matrix_col * 2]) =
          pack_f32x2_to_f16x2(tCrC[m][n][1][1][0], tCrC[m][n][1][1][1]);
    }
  }

  __syncthreads();

  for (int vec = threadIdx.x; vec < kCtaM * kCtaN / kElementsPerAccess;
       vec += blockDim.x) {
    int vec_row = vec / (kCtaN / kElementsPerAccess);
    int vec_col = vec % (kCtaN / kElementsPerAccess);
    uint4 *d_ptr = reinterpret_cast<uint4 *>(gC + vec_row * StrideC +
                                             vec_col * kElementsPerAccess);
    uint4 *s_ptr = reinterpret_cast<uint4 *>(sC + vec_row * kSmemStrideC +
                                             vec_col * kElementsPerAccess);
    *d_ptr = *s_ptr;
  }
}

template <typename Shape_MNK = shape_mnk, int kStages, int kBlockSwizzle>
__global__ void hgemm_f16f16f16_kernel(half *A, half *B, half *C, int M, int N,
                                       int K) {
  constexpr int kCtaM = Shape_MNK::M; // 128
  constexpr int kCtaN = Shape_MNK::N; // 128
  constexpr int kCtaK = Shape_MNK::K; // 64
  static_assert(kCtaM == 128 && kCtaN == 128 && kCtaK == 64,
                "swizzled shared-memory layout assumes a 128x128x64 CTA");

  constexpr int kWarpsM = 2;
  constexpr int kWarpSize = 32;

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

  int const tile_m_max = (M + kCtaM - 1) / kCtaM;
  int const tile_n_max = (N + kCtaN - 1) / kCtaN;

  int tile_m = blockIdx.x / kBlockSwizzle;
  int tile_n = blockIdx.y * kBlockSwizzle + blockIdx.x % kBlockSwizzle;
  if (tile_m >= tile_m_max || tile_n >= tile_n_max) {
    return;
  }

  const half *gA_base = A + tile_m * kCtaM * StrideA;
  const half *gB_base = B + tile_n * kCtaN;

  half *gC = C + tile_m * kCtaM * StrideC + tile_n * kCtaN;

  int tid = threadIdx.x;
  int warp_id = tid / kWarpSize;

  int const K_TILE_MAX = K / kCtaK;
  constexpr int K_BLOCK_MAX = kCtaK / Tiled_MMA_K;
  constexpr int K_PIPE_MAX = kStages;
  static_assert(K_BLOCK_MAX == 4,
                "mainloop cp.async schedule assumes four MMA K-blocks");

  constexpr int MMA_M = kCtaM / Tiled_MMA_M;
  constexpr int MMA_N = kCtaN / Tiled_MMA_N;
  constexpr int MMA_K = kCtaK / Tiled_MMA_K;

  constexpr int Fragment = 2;
  constexpr int CoreMatrix_M = 2;
  constexpr int CoreMatrix_N = 2;
  constexpr int CoreMatrix_K = 2;

  constexpr int kElementsPerAccess = 8; // half, 16B

  // (MMA_M, MMA_N, CoreMatrix_N, CoreMatrix_M, Fragment)
  // :
  // (8 * MMA_N, 8, 4, 2, 1)

  half tCrC[MMA_M][MMA_N][CoreMatrix_N][CoreMatrix_M][Fragment];
  half tCrA[MMA_M][MMA_K][CoreMatrix_K][CoreMatrix_M][Fragment];
  half tCrB[MMA_N][MMA_K][CoreMatrix_N][CoreMatrix_K][Fragment];

#pragma unroll
  for (int m = 0; m < MMA_M; ++m) {
#pragma unroll
    for (int n = 0; n < MMA_N; ++n) {
#pragma unroll
      for (int cm_n = 0; cm_n < CoreMatrix_N; ++cm_n) {
#pragma unroll
        for (int cm_m = 0; cm_m < CoreMatrix_M; ++cm_m) {
          as_u32(tCrC[m][n][cm_n][cm_m][0]) = 0;
        }
      }
    }
  }

  int lane_id = tid % kWarpSize;

  int tA_row = tid / (kCtaK / kElementsPerAccess); // 8
  int tA_col = tid % (kCtaK / kElementsPerAccess);

  int tB_row = tid / (kCtaN / kElementsPerAccess); // 16
  int tB_col = tid % (kCtaN / kElementsPerAccess);

  int k_tiles_to_issue = K_TILE_MAX;
  int k_tiles_to_compute = K_TILE_MAX;
  int k_tile_next = 0;

#pragma unroll
  for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe) {
    const half *gA = gA_base + k_tile_next * kCtaK;
    const half *gB = gB_base + k_tile_next * kCtaK * StrideB;
    cp_async::cg<16>(
        &smem->buffer[k_pipe].A[hgemm_smem::offset_A(
            tA_row + 0 * 16, tA_col * kElementsPerAccess)],
        &gA[(tA_row + 0 * 16) * StrideA + tA_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].A[hgemm_smem::offset_A(
            tA_row + 1 * 16, tA_col * kElementsPerAccess)],
        &gA[(tA_row + 1 * 16) * StrideA + tA_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].A[hgemm_smem::offset_A(
            tA_row + 2 * 16, tA_col * kElementsPerAccess)],
        &gA[(tA_row + 2 * 16) * StrideA + tA_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].A[hgemm_smem::offset_A(
            tA_row + 3 * 16, tA_col * kElementsPerAccess)],
        &gA[(tA_row + 3 * 16) * StrideA + tA_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].A[hgemm_smem::offset_A(
            tA_row + 4 * 16, tA_col * kElementsPerAccess)],
        &gA[(tA_row + 4 * 16) * StrideA + tA_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].A[hgemm_smem::offset_A(
            tA_row + 5 * 16, tA_col * kElementsPerAccess)],
        &gA[(tA_row + 5 * 16) * StrideA + tA_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].A[hgemm_smem::offset_A(
            tA_row + 6 * 16, tA_col * kElementsPerAccess)],
        &gA[(tA_row + 6 * 16) * StrideA + tA_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].A[hgemm_smem::offset_A(
            tA_row + 7 * 16, tA_col * kElementsPerAccess)],
        &gA[(tA_row + 7 * 16) * StrideA + tA_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].B[hgemm_smem::offset_B(
            tB_col * kElementsPerAccess, tB_row + 0 * 8)],
        &gB[(tB_row + 0 * 8) * StrideB + tB_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].B[hgemm_smem::offset_B(
            tB_col * kElementsPerAccess, tB_row + 1 * 8)],
        &gB[(tB_row + 1 * 8) * StrideB + tB_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].B[hgemm_smem::offset_B(
            tB_col * kElementsPerAccess, tB_row + 2 * 8)],
        &gB[(tB_row + 2 * 8) * StrideB + tB_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].B[hgemm_smem::offset_B(
            tB_col * kElementsPerAccess, tB_row + 3 * 8)],
        &gB[(tB_row + 3 * 8) * StrideB + tB_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].B[hgemm_smem::offset_B(
            tB_col * kElementsPerAccess, tB_row + 4 * 8)],
        &gB[(tB_row + 4 * 8) * StrideB + tB_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].B[hgemm_smem::offset_B(
            tB_col * kElementsPerAccess, tB_row + 5 * 8)],
        &gB[(tB_row + 5 * 8) * StrideB + tB_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].B[hgemm_smem::offset_B(
            tB_col * kElementsPerAccess, tB_row + 6 * 8)],
        &gB[(tB_row + 6 * 8) * StrideB + tB_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe].B[hgemm_smem::offset_B(
            tB_col * kElementsPerAccess, tB_row + 7 * 8)],
        &gB[(tB_row + 7 * 8) * StrideB + tB_col * kElementsPerAccess]);

    cp_async::commit_group();
    --k_tiles_to_issue;
    ++k_tile_next;
  }

  int smem_pipe_read = 0;
  int smem_pipe_write = K_PIPE_MAX - 1;

  int warp_m_id = warp_id % kWarpsM;
  int warp_n_id = warp_id / kWarpsM;

  int ldsmx4_row = lane_id % 16;
  int ldsmx4_col = lane_id / 16;

  int ldsmx4T_col = lane_id % 16;
  int ldsmx4T_row = lane_id / 16;

  if constexpr (K_BLOCK_MAX > 1) {
    cp_async::wait_group<K_PIPE_MAX - 2>();
    __syncthreads();

    ldsm::x4<ldsm::N>(as_u32(tCrA[0][0][0][0][0]), as_u32(tCrA[0][0][0][1][0]),
                      as_u32(tCrA[0][0][1][0][0]), as_u32(tCrA[0][0][1][1][0]),
                      &smem->buffer[smem_pipe_read].A[hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 0 * Tiled_MMA_M,
                          0 * Tiled_MMA_K + ldsmx4_col * 8)]);
    ldsm::x4<ldsm::N>(as_u32(tCrA[1][0][0][0][0]), as_u32(tCrA[1][0][0][1][0]),
                      as_u32(tCrA[1][0][1][0][0]), as_u32(tCrA[1][0][1][1][0]),
                      &smem->buffer[smem_pipe_read].A[hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 1 * Tiled_MMA_M,
                          0 * Tiled_MMA_K + ldsmx4_col * 8)]);
    ldsm::x4<ldsm::N>(as_u32(tCrA[2][0][0][0][0]), as_u32(tCrA[2][0][0][1][0]),
                      as_u32(tCrA[2][0][1][0][0]), as_u32(tCrA[2][0][1][1][0]),
                      &smem->buffer[smem_pipe_read].A[hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 2 * Tiled_MMA_M,
                          0 * Tiled_MMA_K + ldsmx4_col * 8)]);
    ldsm::x4<ldsm::N>(as_u32(tCrA[3][0][0][0][0]), as_u32(tCrA[3][0][0][1][0]),
                      as_u32(tCrA[3][0][1][0][0]), as_u32(tCrA[3][0][1][1][0]),
                      &smem->buffer[smem_pipe_read].A[hgemm_smem::offset_A(
                          warp_m_id * 16 + ldsmx4_row + 3 * Tiled_MMA_M,
                          0 * Tiled_MMA_K + ldsmx4_col * 8)]);
    ldsm::x4<ldsm::T>(as_u32(tCrB[0][0][0][0][0]), as_u32(tCrB[0][0][0][1][0]),
                      as_u32(tCrB[0][0][1][0][0]), as_u32(tCrB[0][0][1][1][0]),
                      &smem->buffer[smem_pipe_read].B[hgemm_smem::offset_B(
                          warp_n_id * 8 + Tiled_MMA_N * 0 + ldsmx4T_row * 16,
                          ldsmx4T_col + 0 * Tiled_MMA_K)]);
    ldsm::x4<ldsm::T>(as_u32(tCrB[1][0][0][0][0]), as_u32(tCrB[1][0][0][1][0]),
                      as_u32(tCrB[1][0][1][0][0]), as_u32(tCrB[1][0][1][1][0]),
                      &smem->buffer[smem_pipe_read].B[hgemm_smem::offset_B(
                          warp_n_id * 8 + Tiled_MMA_N * 1 + ldsmx4T_row * 16,
                          ldsmx4T_col + 0 * Tiled_MMA_K)]);
    ldsm::x4<ldsm::T>(as_u32(tCrB[2][0][0][0][0]), as_u32(tCrB[2][0][0][1][0]),
                      as_u32(tCrB[2][0][1][0][0]), as_u32(tCrB[2][0][1][1][0]),
                      &smem->buffer[smem_pipe_read].B[hgemm_smem::offset_B(
                          warp_n_id * 8 + Tiled_MMA_N * 2 + ldsmx4T_row * 16,
                          ldsmx4T_col + 0 * Tiled_MMA_K)]);
    ldsm::x4<ldsm::T>(as_u32(tCrB[3][0][0][0][0]), as_u32(tCrB[3][0][0][1][0]),
                      as_u32(tCrB[3][0][1][0][0]), as_u32(tCrB[3][0][1][1][0]),
                      &smem->buffer[smem_pipe_read].B[hgemm_smem::offset_B(
                          warp_n_id * 8 + Tiled_MMA_N * 3 + ldsmx4T_row * 16,
                          ldsmx4T_col + 0 * Tiled_MMA_K)]);
  }
  auto stage_A_p = smem->buffer[smem_pipe_read].A;
  auto stage_B_p = smem->buffer[smem_pipe_read].B;

  while (k_tiles_to_compute > 0) {
#pragma unroll
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
      if (k_tiles_to_issue > 0) {
        const half *gA = gA_base + k_tile_next * kCtaK;
        const half *gB = gB_base + k_tile_next * kCtaK * StrideB;
        half *sA = smem->buffer[smem_pipe_write].A;
        half *sB = smem->buffer[smem_pipe_write].B;

        if (k_block == 0) {
          issue_cp_async_A<0>(sA, gA, tA_row, tA_col, StrideA);
          issue_cp_async_A<1>(sA, gA, tA_row, tA_col, StrideA);
          issue_cp_async_B<0>(sB, gB, tB_row, tB_col, StrideB);
          issue_cp_async_B<1>(sB, gB, tB_row, tB_col, StrideB);
        } else if (k_block == 1) {
          issue_cp_async_A<2>(sA, gA, tA_row, tA_col, StrideA);
          issue_cp_async_A<3>(sA, gA, tA_row, tA_col, StrideA);
          issue_cp_async_B<2>(sB, gB, tB_row, tB_col, StrideB);
          issue_cp_async_B<3>(sB, gB, tB_row, tB_col, StrideB);
        } else if (k_block == 2) {
          issue_cp_async_A<4>(sA, gA, tA_row, tA_col, StrideA);
          issue_cp_async_A<5>(sA, gA, tA_row, tA_col, StrideA);
          issue_cp_async_B<4>(sB, gB, tB_row, tB_col, StrideB);
          issue_cp_async_B<5>(sB, gB, tB_row, tB_col, StrideB);
        } else {
          issue_cp_async_A<6>(sA, gA, tA_row, tA_col, StrideA);
          issue_cp_async_A<7>(sA, gA, tA_row, tA_col, StrideA);
          issue_cp_async_B<6>(sB, gB, tB_row, tB_col, StrideB);
          issue_cp_async_B<7>(sB, gB, tB_row, tB_col, StrideB);
        }
      }

      if (k_block == K_BLOCK_MAX - 1) {
        cp_async::commit_group();
        if (k_tiles_to_issue > 0) {
          --k_tiles_to_issue;
          ++k_tile_next;
        }
        smem_pipe_write = smem_pipe_read;
        smem_pipe_read =
            (smem_pipe_read == K_PIPE_MAX - 1) ? 0 : smem_pipe_read + 1;

        stage_A_p = smem->buffer[smem_pipe_read].A;
        stage_B_p = smem->buffer[smem_pipe_read].B;
        if (k_tiles_to_compute <= K_PIPE_MAX - 1) {
          cp_async::wait_group<0>();
        } else {
          cp_async::wait_group<K_PIPE_MAX - 2>();
        }
        __syncthreads();
      }

      int k_block_next = (k_block + 1) % K_BLOCK_MAX;
      ldsm::x4<ldsm::N>(as_u32(tCrA[0][k_block_next][0][0][0]),
                        as_u32(tCrA[0][k_block_next][0][1][0]),
                        as_u32(tCrA[0][k_block_next][1][0][0]),
                        as_u32(tCrA[0][k_block_next][1][1][0]),
                        &smem->buffer[smem_pipe_read].A[hgemm_smem::offset_A(
                            warp_m_id * 16 + ldsmx4_row + 0 * Tiled_MMA_M,
                            k_block_next * Tiled_MMA_K + ldsmx4_col * 8)]);
      ldsm::x4<ldsm::N>(as_u32(tCrA[1][k_block_next][0][0][0]),
                        as_u32(tCrA[1][k_block_next][0][1][0]),
                        as_u32(tCrA[1][k_block_next][1][0][0]),
                        as_u32(tCrA[1][k_block_next][1][1][0]),
                        &smem->buffer[smem_pipe_read].A[hgemm_smem::offset_A(
                            warp_m_id * 16 + ldsmx4_row + 1 * Tiled_MMA_M,
                            k_block_next * Tiled_MMA_K + ldsmx4_col * 8)]);
      ldsm::x4<ldsm::N>(as_u32(tCrA[2][k_block_next][0][0][0]),
                        as_u32(tCrA[2][k_block_next][0][1][0]),
                        as_u32(tCrA[2][k_block_next][1][0][0]),
                        as_u32(tCrA[2][k_block_next][1][1][0]),
                        &smem->buffer[smem_pipe_read].A[hgemm_smem::offset_A(
                            warp_m_id * 16 + ldsmx4_row + 2 * Tiled_MMA_M,
                            k_block_next * Tiled_MMA_K + ldsmx4_col * 8)]);
      ldsm::x4<ldsm::N>(as_u32(tCrA[3][k_block_next][0][0][0]),
                        as_u32(tCrA[3][k_block_next][0][1][0]),
                        as_u32(tCrA[3][k_block_next][1][0][0]),
                        as_u32(tCrA[3][k_block_next][1][1][0]),
                        &smem->buffer[smem_pipe_read].A[hgemm_smem::offset_A(
                            warp_m_id * 16 + ldsmx4_row + 3 * Tiled_MMA_M,
                            k_block_next * Tiled_MMA_K + ldsmx4_col * 8)]);
      ldsm::x4<ldsm::T>(as_u32(tCrB[0][k_block_next][0][0][0]),
                        as_u32(tCrB[0][k_block_next][0][1][0]),
                        as_u32(tCrB[0][k_block_next][1][0][0]),
                        as_u32(tCrB[0][k_block_next][1][1][0]),
                        &smem->buffer[smem_pipe_read].B[hgemm_smem::offset_B(
                            warp_n_id * 8 + Tiled_MMA_N * 0 + ldsmx4T_row * 16,
                            ldsmx4T_col + k_block_next * Tiled_MMA_K)]);
      ldsm::x4<ldsm::T>(as_u32(tCrB[1][k_block_next][0][0][0]),
                        as_u32(tCrB[1][k_block_next][0][1][0]),
                        as_u32(tCrB[1][k_block_next][1][0][0]),
                        as_u32(tCrB[1][k_block_next][1][1][0]),
                        &smem->buffer[smem_pipe_read].B[hgemm_smem::offset_B(
                            warp_n_id * 8 + Tiled_MMA_N * 1 + ldsmx4T_row * 16,
                            ldsmx4T_col + k_block_next * Tiled_MMA_K)]);
      ldsm::x4<ldsm::T>(as_u32(tCrB[2][k_block_next][0][0][0]),
                        as_u32(tCrB[2][k_block_next][0][1][0]),
                        as_u32(tCrB[2][k_block_next][1][0][0]),
                        as_u32(tCrB[2][k_block_next][1][1][0]),
                        &smem->buffer[smem_pipe_read].B[hgemm_smem::offset_B(
                            warp_n_id * 8 + Tiled_MMA_N * 2 + ldsmx4T_row * 16,
                            ldsmx4T_col + k_block_next * Tiled_MMA_K)]);
      ldsm::x4<ldsm::T>(as_u32(tCrB[3][k_block_next][0][0][0]),
                        as_u32(tCrB[3][k_block_next][0][1][0]),
                        as_u32(tCrB[3][k_block_next][1][0][0]),
                        as_u32(tCrB[3][k_block_next][1][1][0]),
                        &smem->buffer[smem_pipe_read].B[hgemm_smem::offset_B(
                            warp_n_id * 8 + Tiled_MMA_N * 3 + ldsmx4T_row * 16,
                            ldsmx4T_col + k_block_next * Tiled_MMA_K)]);

      MMA_1_ROW(0);
      MMA_1_ROW(1);
      MMA_1_ROW(2);
      MMA_1_ROW(3);
    }
    --k_tiles_to_compute;
  }

  //
  // Epilogue
  //

  cp_async::wait_all();
  __syncthreads();
  half *sC = reinterpret_cast<half *>(shared_memory);

  int core_matrix_row = lane_id / 4;
  int core_matrix_col = lane_id % 4;

  constexpr int kSmemStrideC = 136;
#pragma unroll
  for (int m = 0; m < MMA_M; ++m) {
    for (int n = 0; n < MMA_N; ++n) {
      *reinterpret_cast<uint32_t *>(
          &sC[(m * Tiled_MMA_M + warp_m_id * 16 + 0 * 8 + core_matrix_row) *
                  kSmemStrideC +
              n * Tiled_MMA_N + warp_n_id * 8 + 0 * 16 + core_matrix_col * 2]) =
          as_u32(tCrC[m][n][0][0][0]);
      *reinterpret_cast<uint32_t *>(
          &sC[(m * Tiled_MMA_M + warp_m_id * 16 + 1 * 8 + core_matrix_row) *
                  kSmemStrideC +
              n * Tiled_MMA_N + warp_n_id * 8 + 0 * 16 + core_matrix_col * 2]) =
          as_u32(tCrC[m][n][0][1][0]);

      *reinterpret_cast<uint32_t *>(
          &sC[(m * Tiled_MMA_M + warp_m_id * 16 + 0 * 8 + core_matrix_row) *
                  kSmemStrideC +
              n * Tiled_MMA_N + warp_n_id * 8 + 1 * 16 + core_matrix_col * 2]) =
          as_u32(tCrC[m][n][1][0][0]);
      *reinterpret_cast<uint32_t *>(
          &sC[(m * Tiled_MMA_M + warp_m_id * 16 + 1 * 8 + core_matrix_row) *
                  kSmemStrideC +
              n * Tiled_MMA_N + warp_n_id * 8 + 1 * 16 + core_matrix_col * 2]) =
          as_u32(tCrC[m][n][1][1][0]);
    }
  }

  __syncthreads();

  for (int vec = threadIdx.x; vec < kCtaM * kCtaN / kElementsPerAccess;
       vec += blockDim.x) {
    int vec_row = vec / (kCtaN / kElementsPerAccess);
    int vec_col = vec % (kCtaN / kElementsPerAccess);
    uint4 *d_ptr = reinterpret_cast<uint4 *>(gC + vec_row * StrideC +
                                             vec_col * kElementsPerAccess);
    uint4 *s_ptr = reinterpret_cast<uint4 *>(sC + vec_row * kSmemStrideC +
                                             vec_col * kElementsPerAccess);
    *d_ptr = *s_ptr;
  }
}

namespace sm80_hgemm {

constexpr int kStages = 3;
constexpr int kBlockSwizzle = 8;
constexpr int kThreads = 128;
constexpr int kSharedStorageBytes =
    sizeof(HgemmSharedStorage<shape_mnk, kStages>);

inline cudaError_t launch_hgemm_128x128x64_fp16acc(half *A, half *B, half *C,
                                                   int M, int N, int K,
                                                   cudaStream_t stream = 0) {
  auto kernel_fptr = hgemm_f16f16f16_kernel<shape_mnk, kStages, kBlockSwizzle>;

  cudaError_t err = cudaFuncSetAttribute(
      kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
      kSharedStorageBytes);
  if (err != cudaSuccess)
    return err;

  err = cudaFuncSetAttribute(
      kernel_fptr, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  if (err != cudaSuccess)
    return err;

  int tile_m_count = M / shape_mnk::M;
  int tile_n_count = N / shape_mnk::N;
  dim3 block(kThreads);
  dim3 grid(tile_m_count * kBlockSwizzle,
            (tile_n_count + kBlockSwizzle - 1) / kBlockSwizzle);

  kernel_fptr<<<grid, block, kSharedStorageBytes, stream>>>(A, B, C, M, N, K);
  return cudaGetLastError();
}

inline cudaError_t launch_hgemm_128x128x64_fp32acc(half *A, half *B, half *C,
                                                   int M, int N, int K,
                                                   cudaStream_t stream = 0) {
  auto kernel_fptr = hgemm_f16f16f32_kernel<shape_mnk, kStages, kBlockSwizzle>;

  cudaError_t err = cudaFuncSetAttribute(
      kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
      kSharedStorageBytes);
  if (err != cudaSuccess)
    return err;

  err = cudaFuncSetAttribute(
      kernel_fptr, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  if (err != cudaSuccess)
    return err;

  int tile_m_count = M / shape_mnk::M;
  int tile_n_count = N / shape_mnk::N;
  dim3 block(kThreads);
  dim3 grid(tile_m_count * kBlockSwizzle,
            (tile_n_count + kBlockSwizzle - 1) / kBlockSwizzle);

  kernel_fptr<<<grid, block, kSharedStorageBytes, stream>>>(A, B, C, M, N, K);
  return cudaGetLastError();
}

} // namespace sm80_hgemm
