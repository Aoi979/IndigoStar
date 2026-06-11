#pragma once

#include "../../../common.hpp"
#include <vector_types.h>
namespace cutlass_like {

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define FETCH_CONST_FLOAT4(pointer)                                            \
  (reinterpret_cast<const float4 *>(&(pointer))[0])

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
                   ::cutlass_like::smem_addr(dst)),                            \
               "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes)                                           \
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(   \
                   ::cutlass_like::smem_addr(dst)),                            \
               "l"(src), "n"(bytes))

#define CP_ASYNC_CA_4B(dst, src)                                               \
  asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 4;\n" ::"r"(    \
                   ::cutlass_like::smem_addr(dst)),                            \
               "l"(src))

// CG (cache global, L2 only) 不支持 4B，最小 16B
// 若你硬要 4B 且只走 L2，仍然只能用 CA

// ---------- 宏定义 ----------
#define FFMA_ROW(C_row, A_comp, B0, B1)                                        \
  tCrC[C_row][0] += (A_comp) * (B0).x;                                         \
  tCrC[C_row][1] += (A_comp) * (B0).y;                                         \
  tCrC[C_row][2] += (A_comp) * (B0).z;                                         \
  tCrC[C_row][3] += (A_comp) * (B0).w;                                         \
  tCrC[C_row][4] += (A_comp) * (B1).x;                                         \
  tCrC[C_row][5] += (A_comp) * (B1).y;                                         \
  tCrC[C_row][6] += (A_comp) * (B1).z;                                         \
  tCrC[C_row][7] += (A_comp) * (B1).w

#define FFMA_8x8(kb)                                                           \
  FFMA_ROW(0, tCrA[kb][0].x, tCrB[kb][0], tCrB[kb][1]);                        \
  FFMA_ROW(1, tCrA[kb][0].y, tCrB[kb][0], tCrB[kb][1]);                        \
  FFMA_ROW(2, tCrA[kb][0].z, tCrB[kb][0], tCrB[kb][1]);                        \
  FFMA_ROW(3, tCrA[kb][0].w, tCrB[kb][0], tCrB[kb][1]);                        \
  FFMA_ROW(4, tCrA[kb][1].x, tCrB[kb][0], tCrB[kb][1]);                        \
  FFMA_ROW(5, tCrA[kb][1].y, tCrB[kb][0], tCrB[kb][1]);                        \
  FFMA_ROW(6, tCrA[kb][1].z, tCrB[kb][0], tCrB[kb][1]);                        \
  FFMA_ROW(7, tCrA[kb][1].w, tCrB[kb][0], tCrB[kb][1])

#define FFMA_COL_ASC(kb, C_col, B_comp)                                        \
  tCrC[C_col][0] += tCrA[kb][0].x * (B_comp);                                  \
  tCrC[C_col][1] += tCrA[kb][0].y * (B_comp);                                  \
  tCrC[C_col][2] += tCrA[kb][0].z * (B_comp);                                  \
  tCrC[C_col][3] += tCrA[kb][0].w * (B_comp);                                  \
  tCrC[C_col][4] += tCrA[kb][1].x * (B_comp);                                  \
  tCrC[C_col][5] += tCrA[kb][1].y * (B_comp);                                  \
  tCrC[C_col][6] += tCrA[kb][1].z * (B_comp);                                  \
  tCrC[C_col][7] += tCrA[kb][1].w * (B_comp)

#define FFMA_COL_DESC(kb, C_col, B_comp)                                       \
  tCrC[C_col][7] += tCrA[kb][1].w * (B_comp);                                  \
  tCrC[C_col][6] += tCrA[kb][1].z * (B_comp);                                  \
  tCrC[C_col][5] += tCrA[kb][1].y * (B_comp);                                  \
  tCrC[C_col][4] += tCrA[kb][1].x * (B_comp);                                  \
  tCrC[C_col][3] += tCrA[kb][0].w * (B_comp);                                  \
  tCrC[C_col][2] += tCrA[kb][0].z * (B_comp);                                  \
  tCrC[C_col][1] += tCrA[kb][0].y * (B_comp);                                  \
  tCrC[C_col][0] += tCrA[kb][0].x * (B_comp)

#define FFMA_8x8_CUTLASS_SM80(kb)                                              \
  FFMA_COL_ASC(kb, 0, tCrB[kb][0].x);                                          \
  FFMA_COL_DESC(kb, 1, tCrB[kb][0].y);                                         \
  FFMA_COL_ASC(kb, 2, tCrB[kb][0].z);                                          \
  FFMA_COL_DESC(kb, 3, tCrB[kb][0].w);                                         \
  FFMA_COL_ASC(kb, 4, tCrB[kb][1].x);                                          \
  FFMA_COL_DESC(kb, 5, tCrB[kb][1].y);                                         \
  FFMA_COL_ASC(kb, 6, tCrB[kb][1].z);                                          \
  FFMA_COL_DESC(kb, 7, tCrB[kb][1].w)

#define STORE_4_ROWS(B_half)                                                   \
  FETCH_FLOAT4(                                                                \
      gC[(warp_row * kWarpM + 0 * (kWarpM / 2) + lC_row * 4 + 0) * kStrideC +  \
         warp_col * kWarpN + (B_half) * (kWarpN / 2) + lC_col * 4]) =          \
      FETCH_FLOAT4(tCrC[0][(B_half) * 4]);                                     \
  FETCH_FLOAT4(                                                                \
      gC[(warp_row * kWarpM + 0 * (kWarpM / 2) + lC_row * 4 + 1) * kStrideC +  \
         warp_col * kWarpN + (B_half) * (kWarpN / 2) + lC_col * 4]) =          \
      FETCH_FLOAT4(tCrC[1][(B_half) * 4]);                                     \
  FETCH_FLOAT4(                                                                \
      gC[(warp_row * kWarpM + 0 * (kWarpM / 2) + lC_row * 4 + 2) * kStrideC +  \
         warp_col * kWarpN + (B_half) * (kWarpN / 2) + lC_col * 4]) =          \
      FETCH_FLOAT4(tCrC[2][(B_half) * 4]);                                     \
  FETCH_FLOAT4(                                                                \
      gC[(warp_row * kWarpM + 0 * (kWarpM / 2) + lC_row * 4 + 3) * kStrideC +  \
         warp_col * kWarpN + (B_half) * (kWarpN / 2) + lC_col * 4]) =          \
      FETCH_FLOAT4(tCrC[3][(B_half) * 4]);                                     \
  FETCH_FLOAT4(                                                                \
      gC[(warp_row * kWarpM + 1 * (kWarpM / 2) + lC_row * 4 + 0) * kStrideC +  \
         warp_col * kWarpN + (B_half) * (kWarpN / 2) + lC_col * 4]) =          \
      FETCH_FLOAT4(tCrC[4][(B_half) * 4]);                                     \
  FETCH_FLOAT4(                                                                \
      gC[(warp_row * kWarpM + 1 * (kWarpM / 2) + lC_row * 4 + 1) * kStrideC +  \
         warp_col * kWarpN + (B_half) * (kWarpN / 2) + lC_col * 4]) =          \
      FETCH_FLOAT4(tCrC[5][(B_half) * 4]);                                     \
  FETCH_FLOAT4(                                                                \
      gC[(warp_row * kWarpM + 1 * (kWarpM / 2) + lC_row * 4 + 2) * kStrideC +  \
         warp_col * kWarpN + (B_half) * (kWarpN / 2) + lC_col * 4]) =          \
      FETCH_FLOAT4(tCrC[6][(B_half) * 4]);                                     \
  FETCH_FLOAT4(                                                                \
      gC[(warp_row * kWarpM + 1 * (kWarpM / 2) + lC_row * 4 + 3) * kStrideC +  \
         warp_col * kWarpN + (B_half) * (kWarpN / 2) + lC_col * 4]) =          \
      FETCH_FLOAT4(tCrC[7][(B_half) * 4])

#define STORE_4_ROWS_COLACC(B_half)                                            \
  {                                                                            \
    constexpr int kColBase = (B_half) * 4;                                      \
    float4 c0 = {tCrC[kColBase + 0][0], tCrC[kColBase + 1][0],                 \
                 tCrC[kColBase + 2][0], tCrC[kColBase + 3][0]};                \
    float4 c1 = {tCrC[kColBase + 0][1], tCrC[kColBase + 1][1],                 \
                 tCrC[kColBase + 2][1], tCrC[kColBase + 3][1]};                \
    float4 c2 = {tCrC[kColBase + 0][2], tCrC[kColBase + 1][2],                 \
                 tCrC[kColBase + 2][2], tCrC[kColBase + 3][2]};                \
    float4 c3 = {tCrC[kColBase + 0][3], tCrC[kColBase + 1][3],                 \
                 tCrC[kColBase + 2][3], tCrC[kColBase + 3][3]};                \
    float4 c4 = {tCrC[kColBase + 0][4], tCrC[kColBase + 1][4],                 \
                 tCrC[kColBase + 2][4], tCrC[kColBase + 3][4]};                \
    float4 c5 = {tCrC[kColBase + 0][5], tCrC[kColBase + 1][5],                 \
                 tCrC[kColBase + 2][5], tCrC[kColBase + 3][5]};                \
    float4 c6 = {tCrC[kColBase + 0][6], tCrC[kColBase + 1][6],                 \
                 tCrC[kColBase + 2][6], tCrC[kColBase + 3][6]};                \
    float4 c7 = {tCrC[kColBase + 0][7], tCrC[kColBase + 1][7],                 \
                 tCrC[kColBase + 2][7], tCrC[kColBase + 3][7]};                \
    FETCH_FLOAT4(                                                              \
        gC[(warp_row * kWarpM + 0 * (kWarpM / 2) + lC_row * 4 + 0) *           \
               kStrideC +                                                      \
           warp_col * kWarpN + (B_half) * (kWarpN / 2) + lC_col * 4]) = c0;    \
    FETCH_FLOAT4(                                                              \
        gC[(warp_row * kWarpM + 0 * (kWarpM / 2) + lC_row * 4 + 1) *           \
               kStrideC +                                                      \
           warp_col * kWarpN + (B_half) * (kWarpN / 2) + lC_col * 4]) = c1;    \
    FETCH_FLOAT4(                                                              \
        gC[(warp_row * kWarpM + 0 * (kWarpM / 2) + lC_row * 4 + 2) *           \
               kStrideC +                                                      \
           warp_col * kWarpN + (B_half) * (kWarpN / 2) + lC_col * 4]) = c2;    \
    FETCH_FLOAT4(                                                              \
        gC[(warp_row * kWarpM + 0 * (kWarpM / 2) + lC_row * 4 + 3) *           \
               kStrideC +                                                      \
           warp_col * kWarpN + (B_half) * (kWarpN / 2) + lC_col * 4]) = c3;    \
    FETCH_FLOAT4(                                                              \
        gC[(warp_row * kWarpM + 1 * (kWarpM / 2) + lC_row * 4 + 0) *           \
               kStrideC +                                                      \
           warp_col * kWarpN + (B_half) * (kWarpN / 2) + lC_col * 4]) = c4;    \
    FETCH_FLOAT4(                                                              \
        gC[(warp_row * kWarpM + 1 * (kWarpM / 2) + lC_row * 4 + 1) *           \
               kStrideC +                                                      \
           warp_col * kWarpN + (B_half) * (kWarpN / 2) + lC_col * 4]) = c5;    \
    FETCH_FLOAT4(                                                              \
        gC[(warp_row * kWarpM + 1 * (kWarpM / 2) + lC_row * 4 + 2) *           \
               kStrideC +                                                      \
           warp_col * kWarpN + (B_half) * (kWarpN / 2) + lC_col * 4]) = c6;    \
    FETCH_FLOAT4(                                                              \
        gC[(warp_row * kWarpM + 1 * (kWarpM / 2) + lC_row * 4 + 3) *           \
               kStrideC +                                                      \
           warp_col * kWarpN + (B_half) * (kWarpN / 2) + lC_col * 4]) = c7;    \
  }

constexpr int kCtaM = 128;
constexpr int kCtaN = 128;
constexpr int kCtaK = 8;
constexpr int kBlockSwizzle = 8;

constexpr int kWarpsM = 4;
constexpr int kWarpsN = 2;
constexpr int kWarps = kWarpsM * kWarpsN;
constexpr int kWarpSize = 32;
constexpr int kThreads = kWarps * kWarpSize;

constexpr int kWarpM = kCtaM / kWarpsM; // 32
constexpr int kWarpN = kCtaN / kWarpsN; // 64
constexpr int kWarpThreadsM = 4;
constexpr int kWarpThreadsN = 8;
constexpr int kThreadM = kWarpM / kWarpThreadsM; // 8
constexpr int kThreadN = kWarpN / kWarpThreadsN; // 8
constexpr int kLaneMmaM = 4;
constexpr int kLaneMmaN = 4;
constexpr int kLaneLayoutInterleave = 2;

constexpr int kSmemStrideA = kCtaM;
constexpr int kSmemStrideB = kCtaN;
constexpr int kSmemAStage = kCtaK * kSmemStrideA;
constexpr int kSmemBStage = kCtaK * kSmemStrideB;

constexpr int kStages = 5;
constexpr int kSharedStorageBytes =
    kStages * (kSmemAStage + kSmemBStage) * sizeof(float);
constexpr int kOneCtaPerSmSmemBytes = 96 * 1024;

template <int A_N, int B_N> struct Stage {
  float A[A_N];
  float B[B_N];
};
template <int A_N, int B_N, int StageNum> struct Stages {
  Stage<A_N, B_N> stage[StageNum];
};

__device__ __forceinline__ unsigned smem_addr(const void *ptr) {
  return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

template <int Group>
__device__ __forceinline__ void
issue_tile_cp_async_group(Stages<kSmemAStage, kSmemBStage, kStages> *stages,
                          int smem_pipe_write, const float *gA, const float *gB,
                          int kStrideA, int kStrideB, int tA_row, int tA_col,
                          int tB_row, int tB_col) {
  static_assert(Group >= 0 && Group < 4);
  constexpr int kARowOffset = Group * 32;
  constexpr int kBRowOffset = Group * 2;

  CP_ASYNC_CA_4B(&stages->stage[smem_pipe_write]
                      .A[tA_row + kARowOffset + tA_col * kSmemStrideA],
                 &gA[(tA_row + kARowOffset) * kStrideA + tA_col]);
  CP_ASYNC_CA_4B(&stages->stage[smem_pipe_write]
                      .B[(tB_row + kBRowOffset) * kSmemStrideB + tB_col],
                 &gB[(tB_row + kBRowOffset) * kStrideB + tB_col]);
}

__device__ __forceinline__ void
issue_tile_cp_async(Stages<kSmemAStage, kSmemBStage, kStages> *stages,
                    int smem_pipe_write, const float *gA, const float *gB,
                    int kStrideA, int kStrideB, int tA_row, int tA_col,
                    int tB_row, int tB_col) {
  issue_tile_cp_async_group<0>(stages, smem_pipe_write, gA, gB, kStrideA,
                               kStrideB, tA_row, tA_col, tB_row, tB_col);
  issue_tile_cp_async_group<1>(stages, smem_pipe_write, gA, gB, kStrideA,
                               kStrideB, tA_row, tA_col, tB_row, tB_col);
  issue_tile_cp_async_group<2>(stages, smem_pipe_write, gA, gB, kStrideA,
                               kStrideB, tA_row, tA_col, tB_row, tB_col);
  issue_tile_cp_async_group<3>(stages, smem_pipe_write, gA, gB, kStrideA,
                               kStrideB, tA_row, tA_col, tB_row, tB_col);
}

template <bool UseCutlassWarpOrder, bool UseCutlassMainloopSchedule,
          bool UseCutlassSm80MmaOrder>
__global__ void sgemm_128x128x8stage5_kernel(float *A, float *B, float *C,
                                             int M, int N, int K) {
  int kStrideA = K;
  int kStrideB = N;
  int kStrideC = N;
  int tile_n_count = N / kCtaN;
  int tile_m = blockIdx.x / kBlockSwizzle;
  int tile_n = blockIdx.y * kBlockSwizzle + blockIdx.x % kBlockSwizzle;
  if (tile_n >= tile_n_count) {
    return;
  }

  const float *gA_base = A + tile_m * kCtaM * kStrideA;
  const float *gB_base = B + tile_n * kCtaN;
  float *gC = C + tile_m * kCtaM * kStrideC + tile_n * kCtaN;

  int tid = threadIdx.x;
  int warp_id = tid / kWarpSize;
  int warp_row = 0;
  int warp_col = 0;
  if constexpr (UseCutlassWarpOrder) {
    warp_row = warp_id % kWarpsM;
    warp_col = warp_id / kWarpsM;
  } else {
    warp_row = warp_id / kWarpsN;
    warp_col = warp_id % kWarpsN;
  }

  constexpr int K_BLOCK_MAX = kCtaK;
  constexpr int K_PIPE_MAX = kStages;

  float tCrC[kThreadM][kThreadN] = {};
  float4 tCrA[K_BLOCK_MAX][kLaneLayoutInterleave];
  float4 tCrB[K_BLOCK_MAX][kLaneLayoutInterleave];

  int lane_id = tid % 32;
  int lC_row = (lane_id >> 4) * kLaneLayoutInterleave +
               (lane_id & (kLaneLayoutInterleave - 1));
  int lC_col = (lane_id / kLaneLayoutInterleave) & (kWarpThreadsN - 1);
  extern __shared__ float smem_raw[];
  Stages<kSmemAStage, kSmemBStage, 5> *stages =
      reinterpret_cast<Stages<kSmemAStage, kSmemBStage, kStages> *>(smem_raw);

  int K_TILE_MAX = K / kCtaK;
  int k_tiles_to_issue = K_TILE_MAX;
  int k_tiles_to_compute = K_TILE_MAX;
  int k_tile_next = 0;

  int tA_row = tid / kCtaK;
  int tA_col = tid % kCtaK;

  int tB_row = tid / kCtaN;
  int tB_col = tid % kCtaN;

#pragma unroll
  for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe) {
    const float *gA = gA_base + k_tile_next * kCtaK;
    const float *gB = gB_base + k_tile_next * kCtaK * kStrideB;
    issue_tile_cp_async(stages, k_pipe, gA, gB, kStrideA, kStrideB, tA_row,
                        tA_col, tB_row, tB_col);
    CP_ASYNC_COMMIT_GROUP();
    --k_tiles_to_issue;
    ++k_tile_next;
  }

  // Current pipe index in smem to read from
  int smem_pipe_read = 0;
  // Current pipe index in smem to write to
  int smem_pipe_write = K_PIPE_MAX - 1;

  // PREFETCH register pipeline
  if constexpr (K_BLOCK_MAX > 1) {
    // Wait until our first prefetched tile is loaded in
    CP_ASYNC_WAIT_GROUP(K_PIPE_MAX - 2);
    __syncthreads();

    // Prefetch the first rmem from the first k-block
    tCrA[0][0] = FETCH_FLOAT4(
        stages->stage[smem_pipe_read]
            .A[(warp_row * kWarpM + 0 * (kWarpM / 2) + lC_row * 4) +
               0 * kSmemStrideA]);
    tCrA[0][1] = FETCH_FLOAT4(
        stages->stage[smem_pipe_read]
            .A[(warp_row * kWarpM + 1 * (kWarpM / 2) + lC_row * 4) +
               0 * kSmemStrideA]);
    tCrB[0][0] = FETCH_FLOAT4(
        stages->stage[smem_pipe_read]
            .B[(warp_col * kWarpN + 0 * (kWarpN / 2) + lC_col * 4) +
               0 * kSmemStrideB]);
    tCrB[0][1] = FETCH_FLOAT4(
        stages->stage[smem_pipe_read]
            .B[(warp_col * kWarpN + 1 * (kWarpN / 2) + lC_col * 4) +
               0 * kSmemStrideB]);
  }

  auto stage_A_p = stages->stage[smem_pipe_read].A;
  auto stage_B_p = stages->stage[smem_pipe_read].B;
  while (k_tiles_to_compute > 0) {
#pragma unroll
    for (int k_block = 0; k_block < K_BLOCK_MAX; k_block++) {
      if (k_block == K_BLOCK_MAX - 1) {
        stage_A_p = stages->stage[smem_pipe_read].A;
        stage_B_p = stages->stage[smem_pipe_read].B;
        CP_ASYNC_WAIT_GROUP(K_PIPE_MAX - 2);
        __syncthreads();
      }

      int k_block_next = (k_block + 1) % K_BLOCK_MAX;
      tCrA[k_block_next][0] = FETCH_FLOAT4(
          stage_A_p[(warp_row * kWarpM + 0 * (kWarpM / 2) + lC_row * 4) +
                    k_block_next * kSmemStrideA]);
      tCrA[k_block_next][1] = FETCH_FLOAT4(
          stage_A_p[(warp_row * kWarpM + 1 * (kWarpM / 2) + lC_row * 4) +
                    k_block_next * kSmemStrideA]);
      tCrB[k_block_next][0] = FETCH_FLOAT4(
          stage_B_p[(warp_col * kWarpN + 0 * (kWarpN / 2) + lC_col * 4) +
                    k_block_next * kSmemStrideB]);
      tCrB[k_block_next][1] = FETCH_FLOAT4(
          stage_B_p[(warp_col * kWarpN + 1 * (kWarpN / 2) + lC_col * 4) +
                    k_block_next * kSmemStrideB]);
      // Spread one tile's async copies over the MMA k-loop, but commit them as
      // a single group so the 5-stage pipeline still advances one tile at a
      // time.
      if constexpr (!UseCutlassMainloopSchedule) {
        if (k_block == 0) {
          if (k_tiles_to_issue > 0) {
            const float *gA = gA_base + k_tile_next * kCtaK;
            const float *gB = gB_base + k_tile_next * kCtaK * kStrideB;
            issue_tile_cp_async_group<0>(stages, smem_pipe_write, gA, gB,
                                         kStrideA, kStrideB, tA_row, tA_col,
                                         tB_row, tB_col);
          }
        }
        if (k_block == 2) {
          if (k_tiles_to_issue > 0) {
            const float *gA = gA_base + k_tile_next * kCtaK;
            const float *gB = gB_base + k_tile_next * kCtaK * kStrideB;
            issue_tile_cp_async_group<1>(stages, smem_pipe_write, gA, gB,
                                         kStrideA, kStrideB, tA_row, tA_col,
                                         tB_row, tB_col);
          }
        }
        if (k_block == 4) {
          if (k_tiles_to_issue > 0) {
            const float *gA = gA_base + k_tile_next * kCtaK;
            const float *gB = gB_base + k_tile_next * kCtaK * kStrideB;
            issue_tile_cp_async_group<2>(stages, smem_pipe_write, gA, gB,
                                         kStrideA, kStrideB, tA_row, tA_col,
                                         tB_row, tB_col);
          }
        }
        if (k_block == 6) {
          if (k_tiles_to_issue > 0) {
            const float *gA = gA_base + k_tile_next * kCtaK;
            const float *gB = gB_base + k_tile_next * kCtaK * kStrideB;
            issue_tile_cp_async_group<3>(stages, smem_pipe_write, gA, gB,
                                         kStrideA, kStrideB, tA_row, tA_col,
                                         tB_row, tB_col);
            --k_tiles_to_issue;
            ++k_tile_next;
          }
          CP_ASYNC_COMMIT_GROUP();
          smem_pipe_write = smem_pipe_read;
          smem_pipe_read = (smem_pipe_read + 1) % K_PIPE_MAX;
        }
      }
      if constexpr (UseCutlassSm80MmaOrder) {
        FFMA_8x8_CUTLASS_SM80(k_block);
      } else {
        FFMA_8x8(k_block);
      }
      if constexpr (UseCutlassMainloopSchedule) {
        if (k_block == 0) {
          if (k_tiles_to_issue > 0) {
            const float *gA = gA_base + k_tile_next * kCtaK;
            const float *gB = gB_base + k_tile_next * kCtaK * kStrideB;
            issue_tile_cp_async_group<0>(stages, smem_pipe_write, gA, gB,
                                         kStrideA, kStrideB, tA_row, tA_col,
                                         tB_row, tB_col);
          }
        }
        if (k_block == 1) {
          if (k_tiles_to_issue > 0) {
            const float *gA = gA_base + k_tile_next * kCtaK;
            const float *gB = gB_base + k_tile_next * kCtaK * kStrideB;
            issue_tile_cp_async_group<1>(stages, smem_pipe_write, gA, gB,
                                         kStrideA, kStrideB, tA_row, tA_col,
                                         tB_row, tB_col);
          }
        }
        if (k_block == 2) {
          if (k_tiles_to_issue > 0) {
            const float *gA = gA_base + k_tile_next * kCtaK;
            const float *gB = gB_base + k_tile_next * kCtaK * kStrideB;
            issue_tile_cp_async_group<2>(stages, smem_pipe_write, gA, gB,
                                         kStrideA, kStrideB, tA_row, tA_col,
                                         tB_row, tB_col);
          }
        }
        if (k_block == 3) {
          if (k_tiles_to_issue > 0) {
            const float *gA = gA_base + k_tile_next * kCtaK;
            const float *gB = gB_base + k_tile_next * kCtaK * kStrideB;
            issue_tile_cp_async_group<3>(stages, smem_pipe_write, gA, gB,
                                         kStrideA, kStrideB, tA_row, tA_col,
                                         tB_row, tB_col);
            --k_tiles_to_issue;
            ++k_tile_next;
          }
        }
        if (k_block == 6) {
          CP_ASYNC_COMMIT_GROUP();
          smem_pipe_write = smem_pipe_read;
          smem_pipe_read = (smem_pipe_read + 1) % K_PIPE_MAX;
        }
      }
    }
    --k_tiles_to_compute;
  }

  if constexpr (UseCutlassSm80MmaOrder) {
    STORE_4_ROWS_COLACC(0);
    STORE_4_ROWS_COLACC(1);
  } else {
    STORE_4_ROWS(0);
    STORE_4_ROWS(1);
  }
}

inline void launch_sgemm_128x128x8stage5(float *A, float *B, float *C, int M,
                                         int N, int K,
                                         cudaStream_t stream = 0) {
  int tile_m_count = M / kCtaM;
  int tile_n_count = N / kCtaN;
  dim3 block(kThreads);
  dim3 grid(tile_m_count * kBlockSwizzle,
            (tile_n_count + kBlockSwizzle - 1) / kBlockSwizzle);
  sgemm_128x128x8stage5_kernel<false, false, false>
      <<<grid, block, kSharedStorageBytes, stream>>>(A, B, C, M, N, K);
}

inline void launch_sgemm_128x128x8stage5_one_cta_per_sm(
    float *A, float *B, float *C, int M, int N, int K,
    cudaStream_t stream = 0) {
  int tile_m_count = M / kCtaM;
  int tile_n_count = N / kCtaN;
  dim3 block(kThreads);
  dim3 grid(tile_m_count * kBlockSwizzle,
            (tile_n_count + kBlockSwizzle - 1) / kBlockSwizzle);
  sgemm_128x128x8stage5_kernel<false, false, false>
      <<<grid, block, kOneCtaPerSmSmemBytes, stream>>>(A, B, C, M, N, K);
}

inline void launch_sgemm_128x128x8stage5_cutlass_warp_order(
    float *A, float *B, float *C, int M, int N, int K,
    cudaStream_t stream = 0) {
  int tile_m_count = M / kCtaM;
  int tile_n_count = N / kCtaN;
  dim3 block(kThreads);
  dim3 grid(tile_m_count * kBlockSwizzle,
            (tile_n_count + kBlockSwizzle - 1) / kBlockSwizzle);
  sgemm_128x128x8stage5_kernel<true, false, false>
      <<<grid, block, kSharedStorageBytes, stream>>>(A, B, C, M, N, K);
}

inline void launch_sgemm_128x128x8stage5_cutlass_schedule(
    float *A, float *B, float *C, int M, int N, int K,
    cudaStream_t stream = 0) {
  int tile_m_count = M / kCtaM;
  int tile_n_count = N / kCtaN;
  dim3 block(kThreads);
  dim3 grid(tile_m_count * kBlockSwizzle,
            (tile_n_count + kBlockSwizzle - 1) / kBlockSwizzle);
  sgemm_128x128x8stage5_kernel<true, true, false>
      <<<grid, block, kSharedStorageBytes, stream>>>(A, B, C, M, N, K);
}

inline void launch_sgemm_128x128x8stage5_cutlass_copy_schedule(
    float *A, float *B, float *C, int M, int N, int K,
    cudaStream_t stream = 0) {
  int tile_m_count = M / kCtaM;
  int tile_n_count = N / kCtaN;
  dim3 block(kThreads);
  dim3 grid(tile_m_count * kBlockSwizzle,
            (tile_n_count + kBlockSwizzle - 1) / kBlockSwizzle);
  sgemm_128x128x8stage5_kernel<false, true, false>
      <<<grid, block, kSharedStorageBytes, stream>>>(A, B, C, M, N, K);
}

inline void launch_sgemm_128x128x8stage5_cutlass_sm80_mma_order(
    float *A, float *B, float *C, int M, int N, int K,
    cudaStream_t stream = 0) {
  int tile_m_count = M / kCtaM;
  int tile_n_count = N / kCtaN;
  dim3 block(kThreads);
  dim3 grid(tile_m_count * kBlockSwizzle,
            (tile_n_count + kBlockSwizzle - 1) / kBlockSwizzle);
  sgemm_128x128x8stage5_kernel<false, false, true>
      <<<grid, block, kSharedStorageBytes, stream>>>(A, B, C, M, N, K);
}

} // namespace cutlass_like
