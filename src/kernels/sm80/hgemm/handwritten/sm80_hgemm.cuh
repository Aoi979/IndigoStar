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

} // namespace mma

__device__ __forceinline__ uint32_t &as_u32(half &x) {
  return *reinterpret_cast<uint32_t *>(&x);
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
  constexpr int kSmemStrideB = kCtaN;

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
  const half *gA_base = A + tile_m * kCtaM * StrideA;
  const half *gB_base = B + tile_n * kCtaN;

  half *gC = C + tile_m * kCtaM * StrideC + tile_n * kCtaN;

  int tid = threadIdx.x;
  int warp_id = tid / kWarpSize;
  int warp_row = warp_id / kWarpsN;
  int warp_col = warp_id % kWarpsN;

  constexpr int K_TILE_MAX = K / kCtaK;
  constexpr int K_BLOCK_MAX = kCtaK / Tiled_MMA_K;
  constexpr int K_PIPE_MAX = kStages;

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
        &smem->buffer[k_pipe]
             .A[(tA_row + 0 * 16) * kSmemStrideA + tA_col * kElementsPerAccess],
        &gA[(tA_row + 0 * 16) * StrideA + tA_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe]
             .A[(tA_row + 1 * 16) * kSmemStrideA + tA_col * kElementsPerAccess],
        &gA[(tA_row + 1 * 16) * StrideA + tA_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe]
             .A[(tA_row + 2 * 16) * kSmemStrideA + tA_col * kElementsPerAccess],
        &gA[(tA_row + 2 * 16) * StrideA + tA_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe]
             .A[(tA_row + 3 * 16) * kSmemStrideA + tA_col * kElementsPerAccess],
        &gA[(tA_row + 3 * 16) * StrideA + tA_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe]
             .A[(tA_row + 4 * 16) * kSmemStrideA + tA_col * kElementsPerAccess],
        &gA[(tA_row + 4 * 16) * StrideA + tA_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe]
             .A[(tA_row + 5 * 16) * kSmemStrideA + tA_col * kElementsPerAccess],
        &gA[(tA_row + 5 * 16) * StrideA + tA_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe]
             .A[(tA_row + 6 * 16) * kSmemStrideA + tA_col * kElementsPerAccess],
        &gA[(tA_row + 6 * 16) * StrideA + tA_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe]
             .A[(tA_row + 7 * 16) * kSmemStrideA + tA_col * kElementsPerAccess],
        &gA[(tA_row + 7 * 16) * StrideA + tA_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe]
             .B[(tB_row + 0 * 8) * kSmemStrideB + tB_col * kElementsPerAccess],
        &gB[(tB_row + 0 * 8) * StrideB + tB_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe]
             .B[(tB_row + 1 * 8) * kSmemStrideB + tB_col * kElementsPerAccess],
        &gB[(tB_row + 1 * 8) * StrideB + tB_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe]
             .B[(tB_row + 2 * 8) * kSmemStrideB + tB_col * kElementsPerAccess],
        &gB[(tB_row + 2 * 8) * StrideB + tB_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe]
             .B[(tB_row + 3 * 8) * kSmemStrideB + tB_col * kElementsPerAccess],
        &gB[(tB_row + 3 * 8) * StrideB + tB_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe]
             .B[(tB_row + 4 * 8) * kSmemStrideB + tB_col * kElementsPerAccess],
        &gB[(tB_row + 4 * 8) * StrideB + tB_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe]
             .B[(tB_row + 5 * 8) * kSmemStrideB + tB_col * kElementsPerAccess],
        &gB[(tB_row + 5 * 8) * StrideB + tB_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe]
             .B[(tB_row + 6 * 8) * kSmemStrideB + tB_col * kElementsPerAccess],
        &gB[(tB_row + 6 * 8) * StrideB + tB_col * kElementsPerAccess]);
    cp_async::cg<16>(
        &smem->buffer[k_pipe]
             .B[(tB_row + 7 * 8) * kSmemStrideB + tB_col * kElementsPerAccess],
        &gB[(tB_row + 7 * 8) * StrideB + tB_col * kElementsPerAccess]);

    cp_async::commit_group();
    --k_tiles_to_issue;
    ++k_tile_next;
  }

  int smem_pipe_read = 0;
  int smem_pipe_write = K_PIPE_MAX - 1;

  int warp_m_id = warp_id % kWarpsM;
  int warp_n_id = warp_id / kWarpsM;
  int warp_base_offset_A = warp_m_id * 16 * kSmemStrideA;
  int warp_base_offset_B = warp_n_id * 8;

  int ldsmx4_row = lane_id % 16;
  int ldsmx4_col = lane_id / 16;

  int ldsmx4T_col = lane_id % 16;
  int ldsmx4T_row = lane_id / 16;

  if constexpr (K_BLOCK_MAX > 1) {
    cp_async::wait_group<K_PIPE_MAX - 2>();
    __syncthreads();

    ldsm::x4<ldsm::N>(as_u32(tCrA[0][0][0][0][0]), as_u32(tCrA[0][0][0][1][0]),
                      as_u32(tCrA[0][0][1][0][0]), as_u32(tCrA[0][0][1][1][0]),
                      &smem->buffer[smem_pipe_read]
                           .A[warp_base_offset_A +
                              (ldsmx4_row + 0 * Tiled_MMA_M) * kSmemStrideA +
                              0 * Tiled_MMA_K + ldsmx4_col * 8]);
    ldsm::x4<ldsm::N>(as_u32(tCrA[1][0][0][0][0]), as_u32(tCrA[1][0][0][1][0]),
                      as_u32(tCrA[1][0][1][0][0]), as_u32(tCrA[1][0][1][1][0]),
                      &smem->buffer[smem_pipe_read]
                           .A[warp_base_offset_A +
                              (ldsmx4_row + 1 * Tiled_MMA_M) * kSmemStrideA +
                              0 * Tiled_MMA_K + ldsmx4_col * 8]);
    ldsm::x4<ldsm::N>(as_u32(tCrA[2][0][0][0][0]), as_u32(tCrA[2][0][0][1][0]),
                      as_u32(tCrA[2][0][1][0][0]), as_u32(tCrA[2][0][1][1][0]),
                      &smem->buffer[smem_pipe_read]
                           .A[warp_base_offset_A +
                              (ldsmx4_row + 2 * Tiled_MMA_M) * kSmemStrideA +
                              0 * Tiled_MMA_K + ldsmx4_col * 8]);
    ldsm::x4<ldsm::N>(as_u32(tCrA[3][0][0][0][0]), as_u32(tCrA[3][0][0][1][0]),
                      as_u32(tCrA[3][0][1][0][0]), as_u32(tCrA[3][0][1][1][0]),
                      &smem->buffer[smem_pipe_read]
                           .A[warp_base_offset_A +
                              (ldsmx4_row + 3 * Tiled_MMA_M) * kSmemStrideA +
                              0 * Tiled_MMA_K + ldsmx4_col * 8]);
    ldsm::x4<ldsm::T>(as_u32(tCrB[0][0][0][0][0]), as_u32(tCrB[0][0][0][1][0]),
                      as_u32(tCrB[0][0][1][0][0]), as_u32(tCrB[0][0][1][1][0]),
                      &smem->buffer[smem_pipe_read]
                           .B[warp_base_offset_B +
                              (ldsmx4T_col + 0 * Tiled_MMA_K) * kSmemStrideB +
                              Tiled_MMA_N * 0 + ldsmx4T_row * 16]);
    ldsm::x4<ldsm::T>(as_u32(tCrB[1][0][0][0][0]), as_u32(tCrB[1][0][0][1][0]),
                      as_u32(tCrB[1][0][1][0][0]), as_u32(tCrB[1][0][1][1][0]),
                      &smem->buffer[smem_pipe_read]
                           .B[warp_base_offset_B +
                              (ldsmx4T_col + 0 * Tiled_MMA_K) * kSmemStrideB +
                              Tiled_MMA_N * 1 + ldsmx4T_row * 16]);
    ldsm::x4<ldsm::T>(as_u32(tCrB[2][0][0][0][0]), as_u32(tCrB[2][0][0][1][0]),
                      as_u32(tCrB[2][0][1][0][0]), as_u32(tCrB[2][0][1][1][0]),
                      &smem->buffer[smem_pipe_read]
                           .B[warp_base_offset_B +
                              (ldsmx4T_col + 0 * Tiled_MMA_K) * kSmemStrideB +
                              Tiled_MMA_N * 2 + ldsmx4T_row * 16]);
    ldsm::x4<ldsm::T>(as_u32(tCrB[3][0][0][0][0]), as_u32(tCrB[3][0][0][1][0]),
                      as_u32(tCrB[3][0][1][0][0]), as_u32(tCrB[3][0][1][1][0]),
                      &smem->buffer[smem_pipe_read]
                           .B[warp_base_offset_B +
                              (ldsmx4T_col + 0 * Tiled_MMA_K) * kSmemStrideB +
                              Tiled_MMA_N * 3 + ldsmx4T_row * 16]);
  }
  auto stage_A_p = smem->buffer[smem_pipe_read].A;
  auto stage_B_p = smem->buffer[smem_pipe_read].B;

  while (k_tiles_to_compute > 0) {
#pragma unroll
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
      if (k_block == K_BLOCK_MAX - 1) {
        stage_A_p = smem->buffer[smem_pipe_read].A;
        stage_B_p = smem->buffer[smem_pipe_read].B;
        cp_async::wait_group<K_PIPE_MAX - 2>();
        __syncthreads();
      }

      int k_block_next = (k_block + 1) % K_BLOCK_MAX;
      ldsm::x4<ldsm::N>(as_u32(tCrA[0][k_block_next][0][0][0]),
                        as_u32(tCrA[0][k_block_next][0][1][0]),
                        as_u32(tCrA[0][k_block_next][1][0][0]),
                        as_u32(tCrA[0][k_block_next][1][1][0]),
                        &smem->buffer[smem_pipe_read]
                             .A[warp_base_offset_A +
                                (ldsmx4_row + 0 * Tiled_MMA_M) * kSmemStrideA +
                                k_block_next * Tiled_MMA_K + ldsmx4_col * 8]);
      ldsm::x4<ldsm::N>(as_u32(tCrA[1][k_block_next][0][0][0]),
                        as_u32(tCrA[1][k_block_next][0][1][0]),
                        as_u32(tCrA[1][k_block_next][1][0][0]),
                        as_u32(tCrA[1][k_block_next][1][1][0]),
                        &smem->buffer[smem_pipe_read]
                             .A[warp_base_offset_A +
                                (ldsmx4_row + 1 * Tiled_MMA_M) * kSmemStrideA +
                                k_block_next * Tiled_MMA_K + ldsmx4_col * 8]);
      ldsm::x4<ldsm::N>(as_u32(tCrA[2][k_block_next][0][0][0]),
                        as_u32(tCrA[2][k_block_next][0][1][0]),
                        as_u32(tCrA[2][k_block_next][1][0][0]),
                        as_u32(tCrA[2][k_block_next][1][1][0]),
                        &smem->buffer[smem_pipe_read]
                             .A[warp_base_offset_A +
                                (ldsmx4_row + 2 * Tiled_MMA_M) * kSmemStrideA +
                                k_block_next * Tiled_MMA_K + ldsmx4_col * 8]);
      ldsm::x4<ldsm::N>(as_u32(tCrA[3][k_block_next][0][0][0]),
                        as_u32(tCrA[3][k_block_next][0][1][0]),
                        as_u32(tCrA[3][k_block_next][1][0][0]),
                        as_u32(tCrA[3][k_block_next][1][1][0]),
                        &smem->buffer[smem_pipe_read]
                             .A[warp_base_offset_A +
                                (ldsmx4_row + 3 * Tiled_MMA_M) * kSmemStrideA +
                                k_block_next * Tiled_MMA_K + ldsmx4_col * 8]);
      ldsm::x4<ldsm::T>(
          as_u32(tCrB[0][k_block_next][0][0][0]),
          as_u32(tCrB[0][k_block_next][0][1][0]),
          as_u32(tCrB[0][k_block_next][1][0][0]),
          as_u32(tCrB[0][k_block_next][1][1][0]),
          &smem->buffer[smem_pipe_read]
               .B[warp_base_offset_B +
                  (ldsmx4T_col + k_block_next * Tiled_MMA_K) * kSmemStrideB +
                  Tiled_MMA_N * 0 + ldsmx4T_row * 16]);
      ldsm::x4<ldsm::T>(
          as_u32(tCrB[1][k_block_next][0][0][0]),
          as_u32(tCrB[1][k_block_next][0][1][0]),
          as_u32(tCrB[1][k_block_next][1][0][0]),
          as_u32(tCrB[1][k_block_next][1][1][0]),
          &smem->buffer[smem_pipe_read]
               .B[warp_base_offset_B +
                  (ldsmx4T_col + k_block_next * Tiled_MMA_K) * kSmemStrideB +
                  Tiled_MMA_N * 1 + ldsmx4T_row * 16]);
      ldsm::x4<ldsm::T>(
          as_u32(tCrB[2][k_block_next][0][0][0]),
          as_u32(tCrB[2][k_block_next][0][1][0]),
          as_u32(tCrB[2][k_block_next][1][0][0]),
          as_u32(tCrB[2][k_block_next][1][1][0]),
          &smem->buffer[smem_pipe_read]
               .B[warp_base_offset_B +
                  (ldsmx4T_col + k_block_next * Tiled_MMA_K) * kSmemStrideB +
                  Tiled_MMA_N * 2 + ldsmx4T_row * 16]);
      ldsm::x4<ldsm::T>(
          as_u32(tCrB[3][k_block_next][0][0][0]),
          as_u32(tCrB[3][k_block_next][0][1][0]),
          as_u32(tCrB[3][k_block_next][1][0][0]),
          as_u32(tCrB[3][k_block_next][1][1][0]),
          &smem->buffer[smem_pipe_read]
               .B[warp_base_offset_B +
                  (ldsmx4T_col + k_block_next * Tiled_MMA_K) * kSmemStrideB +
                  Tiled_MMA_N * 3 + ldsmx4T_row * 16]);

      if (k_block == 0) {
        if (k_tiles_to_issue > 0) {
          const half *gA = gA_base + k_tile_next * kCtaK;

          cp_async::cg<16>(
              &smem->buffer[smem_pipe_write]
                   .A[(tA_row + 0 * 16) * kSmemStrideA +
                      tA_col * kElementsPerAccess],
              &gA[(tA_row + 0 * 16) * StrideA + tA_col * kElementsPerAccess]);
          cp_async::cg<16>(
              &smem->buffer[smem_pipe_write]
                   .A[(tA_row + 1 * 16) * kSmemStrideA +
                      tA_col * kElementsPerAccess],
              &gA[(tA_row + 1 * 16) * StrideA + tA_col * kElementsPerAccess]);
          cp_async::cg<16>(
              &smem->buffer[smem_pipe_write]
                   .A[(tA_row + 2 * 16) * kSmemStrideA +
                      tA_col * kElementsPerAccess],
              &gA[(tA_row + 2 * 16) * StrideA + tA_col * kElementsPerAccess]);
          cp_async::cg<16>(
              &smem->buffer[smem_pipe_write]
                   .A[(tA_row + 3 * 16) * kSmemStrideA +
                      tA_col * kElementsPerAccess],
              &gA[(tA_row + 3 * 16) * StrideA + tA_col * kElementsPerAccess]);
          cp_async::cg<16>(
              &smem->buffer[smem_pipe_write]
                   .A[(tA_row + 4 * 16) * kSmemStrideA +
                      tA_col * kElementsPerAccess],
              &gA[(tA_row + 4 * 16) * StrideA + tA_col * kElementsPerAccess]);
          cp_async::cg<16>(
              &smem->buffer[smem_pipe_write]
                   .A[(tA_row + 5 * 16) * kSmemStrideA +
                      tA_col * kElementsPerAccess],
              &gA[(tA_row + 5 * 16) * StrideA + tA_col * kElementsPerAccess]);
          cp_async::cg<16>(
              &smem->buffer[smem_pipe_write]
                   .A[(tA_row + 6 * 16) * kSmemStrideA +
                      tA_col * kElementsPerAccess],
              &gA[(tA_row + 6 * 16) * StrideA + tA_col * kElementsPerAccess]);
          cp_async::cg<16>(
              &smem->buffer[smem_pipe_write]
                   .A[(tA_row + 7 * 16) * kSmemStrideA +
                      tA_col * kElementsPerAccess],
              &gA[(tA_row + 7 * 16) * StrideA + tA_col * kElementsPerAccess]);
        }
      }
      if (k_block == 1) {
        if (k_tiles_to_issue > 0) {
          const half *gB = gB_base + k_tile_next * kCtaK * StrideB;
          cp_async::cg<16>(
              &smem->buffer[smem_pipe_write].B[(tB_row + 0 * 8) * kSmemStrideB +
                                               tB_col * kElementsPerAccess],
              &gB[(tB_row + 0 * 8) * StrideB + tB_col * kElementsPerAccess]);
          cp_async::cg<16>(
              &smem->buffer[smem_pipe_write].B[(tB_row + 1 * 8) * kSmemStrideB +
                                               tB_col * kElementsPerAccess],
              &gB[(tB_row + 1 * 8) * StrideB + tB_col * kElementsPerAccess]);
          cp_async::cg<16>(
              &smem->buffer[smem_pipe_write].B[(tB_row + 2 * 8) * kSmemStrideB +
                                               tB_col * kElementsPerAccess],
              &gB[(tB_row + 2 * 8) * StrideB + tB_col * kElementsPerAccess]);
          cp_async::cg<16>(
              &smem->buffer[smem_pipe_write].B[(tB_row + 3 * 8) * kSmemStrideB +
                                               tB_col * kElementsPerAccess],
              &gB[(tB_row + 3 * 8) * StrideB + tB_col * kElementsPerAccess]);
          cp_async::cg<16>(
              &smem->buffer[smem_pipe_write].B[(tB_row + 4 * 8) * kSmemStrideB +
                                               tB_col * kElementsPerAccess],
              &gB[(tB_row + 4 * 8) * StrideB + tB_col * kElementsPerAccess]);
          cp_async::cg<16>(
              &smem->buffer[smem_pipe_write].B[(tB_row + 5 * 8) * kSmemStrideB +
                                               tB_col * kElementsPerAccess],
              &gB[(tB_row + 5 * 8) * StrideB + tB_col * kElementsPerAccess]);
          cp_async::cg<16>(
              &smem->buffer[smem_pipe_write].B[(tB_row + 6 * 8) * kSmemStrideB +
                                               tB_col * kElementsPerAccess],
              &gB[(tB_row + 6 * 8) * StrideB + tB_col * kElementsPerAccess]);
          cp_async::cg<16>(
              &smem->buffer[smem_pipe_write].B[(tB_row + 7 * 8) * kSmemStrideB +
                                               tB_col * kElementsPerAccess],
              &gB[(tB_row + 7 * 8) * StrideB + tB_col * kElementsPerAccess]);
        }
        cp_async::commit_group();
        if (k_tiles_to_issue > 0) {
          --k_tiles_to_issue;
          ++k_tile_next;
        }
      }

      if (k_block == K_BLOCK_MAX - 2) {
        smem_pipe_write = smem_pipe_read;
        smem_pipe_read =
            (smem_pipe_read == K_PIPE_MAX - 1) ? 0 : smem_pipe_read + 1;
      }

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
  auto sC = smem->buffer[0].A;

  int core_matrix_row = lane_id / 4;
  int core_matrix_col = lane_id % 4;

  constexpr int kSmemStrideC = 136;
#pragma unroll
  for (int m = 0; m < MMA_M; ++m) {
    for (int n = 0; n < MMA_N; ++n) {
      sC[(m * Tiled_MMA_M + warp_m_id * 16 + 0 * 8 + core_matrix_row) *
             kSmemStrideC +
         n * Tiled_MMA_N + warp_n_id * 8 + 0 * 16 + core_matrix_col] =
          tCrC[m][n][0][0][0];
      sC[(m * Tiled_MMA_M + warp_m_id * 16 + 1 * 8 + core_matrix_row) *
             kSmemStrideC +
         n * Tiled_MMA_N + warp_n_id * 8 + 0 * 16 + core_matrix_col] =
          tCrC[m][n][0][1][0];

      sC[(m * Tiled_MMA_M + warp_m_id * 16 + 0 * 8 + core_matrix_row) *
             kSmemStrideC +
         n * Tiled_MMA_N + warp_n_id * 8 + 1 * 16 + core_matrix_col] =
          tCrC[m][n][1][0][0];
      sC[(m * Tiled_MMA_M + warp_m_id * 16 + 1 * 8 + core_matrix_row) *
             kSmemStrideC +
         n * Tiled_MMA_N + warp_n_id * 8 + 1 * 16 + core_matrix_col] =
          tCrC[m][n][1][1][0];
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