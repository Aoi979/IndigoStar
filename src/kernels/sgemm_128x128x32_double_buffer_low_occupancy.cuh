#include "sgemm_common.cuh"
namespace low_occupancy {

__global__ void sgemm_128x128x32_double_buffer(int M, int N, int K,
                                               float const *__restrict__ A,
                                               float const *__restrict__ B,
                                               float *__restrict__ C) {
  constexpr int kBM = 128;
  constexpr int kBN = 128;
  constexpr int kBK = 32;

  constexpr int MMA_M = 1;
  constexpr int MMA_N = 1;
  constexpr int MMA_K = 8;

  int AStride = K;
  int BStride = N;
  int CStride = N;

  constexpr int ASmemStride = kBM;
  constexpr int BSmemStride = kBN;

  constexpr int tCStride = (kBN / 2) / 4;
  constexpr int Threads = 256;
  constexpr int Warps = 8;

  constexpr int ASmemNumel = kBK * ASmemStride;
  constexpr int BSmemNumel = kBK * BSmemStride;

  const float *gA_base = A + blockIdx.y * kBM * AStride;
  const float *gB_base = B + blockIdx.x * kBN;
  float *gC = C + blockIdx.y * kBM * CStride + blockIdx.x * kBN;

  int tid = threadIdx.x;
  int tx = threadIdx.x % 16;
  int ty = threadIdx.x / 16;

  float tCrC[8][8] = {};
  int tC_row = tid / tCStride;
  int tC_col = tid % tCStride;

  constexpr int K_PIPE_MAX = 2;

  extern __shared__ float smem_raw[];
  Stage<ASmemNumel, BSmemNumel> *stages =
      reinterpret_cast<Stage<ASmemNumel, BSmemNumel> *>(smem_raw);

  float *sA = stages[0].A;
  float *sB = stages[0].B;

  int K_TILE_MAX = K / kBK;
  int K_BLOCK_MAX = kBK / MMA_K;

  int tA_row = tid / (kBK / 4);
  int tA_col = tid % (kBK / 4);

  int tB_row = tid / (kBN / 4);
  int tB_col = tid % (kBN / 4);

  float4 tArA[4];
  float4 tBrB[4];

  float4 tCrA[K_PIPE_MAX][2][MMA_K];
  float4 tCrB[K_PIPE_MAX][2][MMA_K];

  const float *gA = gA_base;
  const float *gB = gB_base;
  tArA[0] = FETCH_CONST_FLOAT4(gA[(tA_row + 0 * 32) * AStride + tA_col * 4]);
  tArA[1] = FETCH_CONST_FLOAT4(gA[(tA_row + 1 * 32) * AStride + tA_col * 4]);
  tArA[2] = FETCH_CONST_FLOAT4(gA[(tA_row + 2 * 32) * AStride + tA_col * 4]);
  tArA[3] = FETCH_CONST_FLOAT4(gA[(tA_row + 3 * 32) * AStride + tA_col * 4]);

  sA[(tA_col * 4 + 0) * ASmemStride + (0 * 32 + tA_row)] = tArA[0].x;
  sA[(tA_col * 4 + 1) * ASmemStride + (0 * 32 + tA_row)] = tArA[0].y;
  sA[(tA_col * 4 + 2) * ASmemStride + (0 * 32 + tA_row)] = tArA[0].z;
  sA[(tA_col * 4 + 3) * ASmemStride + (0 * 32 + tA_row)] = tArA[0].w;

  sA[(tA_col * 4 + 0) * ASmemStride + (1 * 32 + tA_row)] = tArA[1].x;
  sA[(tA_col * 4 + 1) * ASmemStride + (1 * 32 + tA_row)] = tArA[1].y;
  sA[(tA_col * 4 + 2) * ASmemStride + (1 * 32 + tA_row)] = tArA[1].z;
  sA[(tA_col * 4 + 3) * ASmemStride + (1 * 32 + tA_row)] = tArA[1].w;

  sA[(tA_col * 4 + 0) * ASmemStride + (2 * 32 + tA_row)] = tArA[2].x;
  sA[(tA_col * 4 + 1) * ASmemStride + (2 * 32 + tA_row)] = tArA[2].y;
  sA[(tA_col * 4 + 2) * ASmemStride + (2 * 32 + tA_row)] = tArA[2].z;
  sA[(tA_col * 4 + 3) * ASmemStride + (2 * 32 + tA_row)] = tArA[2].w;

  sA[(tA_col * 4 + 0) * ASmemStride + (3 * 32 + tA_row)] = tArA[3].x;
  sA[(tA_col * 4 + 1) * ASmemStride + (3 * 32 + tA_row)] = tArA[3].y;
  sA[(tA_col * 4 + 2) * ASmemStride + (3 * 32 + tA_row)] = tArA[3].z;
  sA[(tA_col * 4 + 3) * ASmemStride + (3 * 32 + tA_row)] = tArA[3].w;

  tBrB[0] = FETCH_CONST_FLOAT4(gB[(tB_row + 0 * 8) * BStride + tB_col * 4]);
  tBrB[1] = FETCH_CONST_FLOAT4(gB[(tB_row + 1 * 8) * BStride + tB_col * 4]);
  tBrB[2] = FETCH_CONST_FLOAT4(gB[(tB_row + 2 * 8) * BStride + tB_col * 4]);
  tBrB[3] = FETCH_CONST_FLOAT4(gB[(tB_row + 3 * 8) * BStride + tB_col * 4]);

  FETCH_FLOAT4(sB[(tB_row + 0 * 8) * BSmemStride + (tB_col * 4 + 0)]) = tBrB[0];

  FETCH_FLOAT4(sB[(tB_row + 1 * 8) * BSmemStride + (tB_col * 4 + 0)]) = tBrB[1];

  FETCH_FLOAT4(sB[(tB_row + 2 * 8) * BSmemStride + (tB_col * 4 + 0)]) = tBrB[2];

  FETCH_FLOAT4(sB[(tB_row + 3 * 8) * BSmemStride + (tB_col * 4 + 0)]) = tBrB[3];

  __syncthreads();

  tCrA[0][0][0] =
      FETCH_FLOAT4(sA[(0 * MMA_K + 0) * ASmemStride + tC_row * 4 + 0 * 64]);
  tCrA[0][0][1] =
      FETCH_FLOAT4(sA[(0 * MMA_K + 1) * ASmemStride + tC_row * 4 + 0 * 64]);
  tCrA[0][0][2] =
      FETCH_FLOAT4(sA[(0 * MMA_K + 2) * ASmemStride + tC_row * 4 + 0 * 64]);
  tCrA[0][0][3] =
      FETCH_FLOAT4(sA[(0 * MMA_K + 3) * ASmemStride + tC_row * 4 + 0 * 64]);
  tCrA[0][0][4] =
      FETCH_FLOAT4(sA[(0 * MMA_K + 4) * ASmemStride + tC_row * 4 + 0 * 64]);
  tCrA[0][0][5] =
      FETCH_FLOAT4(sA[(0 * MMA_K + 5) * ASmemStride + tC_row * 4 + 0 * 64]);
  tCrA[0][0][6] =
      FETCH_FLOAT4(sA[(0 * MMA_K + 6) * ASmemStride + tC_row * 4 + 0 * 64]);
  tCrA[0][0][7] =
      FETCH_FLOAT4(sA[(0 * MMA_K + 7) * ASmemStride + tC_row * 4 + 0 * 64]);

  tCrA[0][1][0] =
      FETCH_FLOAT4(sA[(0 * MMA_K + 0) * ASmemStride + tC_row * 4 + 1 * 64]);
  tCrA[0][1][1] =
      FETCH_FLOAT4(sA[(0 * MMA_K + 1) * ASmemStride + tC_row * 4 + 1 * 64]);
  tCrA[0][1][2] =
      FETCH_FLOAT4(sA[(0 * MMA_K + 2) * ASmemStride + tC_row * 4 + 1 * 64]);
  tCrA[0][1][3] =
      FETCH_FLOAT4(sA[(0 * MMA_K + 3) * ASmemStride + tC_row * 4 + 1 * 64]);
  tCrA[0][1][4] =
      FETCH_FLOAT4(sA[(0 * MMA_K + 4) * ASmemStride + tC_row * 4 + 1 * 64]);
  tCrA[0][1][5] =
      FETCH_FLOAT4(sA[(0 * MMA_K + 5) * ASmemStride + tC_row * 4 + 1 * 64]);
  tCrA[0][1][6] =
      FETCH_FLOAT4(sA[(0 * MMA_K + 6) * ASmemStride + tC_row * 4 + 1 * 64]);
  tCrA[0][1][7] =
      FETCH_FLOAT4(sA[(0 * MMA_K + 7) * ASmemStride + tC_row * 4 + 1 * 64]);

  tCrB[0][0][0] =
      FETCH_FLOAT4(sB[(0 * MMA_K + 0) * BSmemStride + tC_col * 4 + 0 * 64]);
  tCrB[0][0][1] =
      FETCH_FLOAT4(sB[(0 * MMA_K + 1) * BSmemStride + tC_col * 4 + 0 * 64]);
  tCrB[0][0][2] =
      FETCH_FLOAT4(sB[(0 * MMA_K + 2) * BSmemStride + tC_col * 4 + 0 * 64]);
  tCrB[0][0][3] =
      FETCH_FLOAT4(sB[(0 * MMA_K + 3) * BSmemStride + tC_col * 4 + 0 * 64]);
  tCrB[0][0][4] =
      FETCH_FLOAT4(sB[(0 * MMA_K + 4) * BSmemStride + tC_col * 4 + 0 * 64]);
  tCrB[0][0][5] =
      FETCH_FLOAT4(sB[(0 * MMA_K + 5) * BSmemStride + tC_col * 4 + 0 * 64]);
  tCrB[0][0][6] =
      FETCH_FLOAT4(sB[(0 * MMA_K + 6) * BSmemStride + tC_col * 4 + 0 * 64]);
  tCrB[0][0][7] =
      FETCH_FLOAT4(sB[(0 * MMA_K + 7) * BSmemStride + tC_col * 4 + 0 * 64]);

  tCrB[0][1][0] =
      FETCH_FLOAT4(sB[(0 * MMA_K + 0) * BSmemStride + tC_col * 4 + 1 * 64]);
  tCrB[0][1][1] =
      FETCH_FLOAT4(sB[(0 * MMA_K + 1) * BSmemStride + tC_col * 4 + 1 * 64]);
  tCrB[0][1][2] =
      FETCH_FLOAT4(sB[(0 * MMA_K + 2) * BSmemStride + tC_col * 4 + 1 * 64]);
  tCrB[0][1][3] =
      FETCH_FLOAT4(sB[(0 * MMA_K + 3) * BSmemStride + tC_col * 4 + 1 * 64]);
  tCrB[0][1][4] =
      FETCH_FLOAT4(sB[(0 * MMA_K + 4) * BSmemStride + tC_col * 4 + 1 * 64]);
  tCrB[0][1][5] =
      FETCH_FLOAT4(sB[(0 * MMA_K + 5) * BSmemStride + tC_col * 4 + 1 * 64]);
  tCrB[0][1][6] =
      FETCH_FLOAT4(sB[(0 * MMA_K + 6) * BSmemStride + tC_col * 4 + 1 * 64]);
  tCrB[0][1][7] =
      FETCH_FLOAT4(sB[(0 * MMA_K + 7) * BSmemStride + tC_col * 4 + 1 * 64]);

  // Current pipe index in smem to read from
  int smem_pipe_read = 0;
  // Current pipe index in smem to write to
  int smem_pipe_write = smem_pipe_read ^ 1;

  // Current pipe index in rmem to read from
  int rmem_pipe_read = 0;
  // Current pipe index in rmem to write to
  int rmem_pipe_write = smem_pipe_read ^ 1;

  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
    sA = stages[smem_pipe_write].A;
    sB = stages[smem_pipe_write].B;

    int k_tile_next = k_tile + 1;
    if (k_tile_next < K_TILE_MAX) {
      const float *gA = gA_base + k_tile_next * kBK;
      const float *gB = gB_base + k_tile_next * kBK * BStride;

      tArA[0] =
          FETCH_CONST_FLOAT4(gA[(tA_row + 0 * 32) * AStride + tA_col * 4]);
      tArA[1] =
          FETCH_CONST_FLOAT4(gA[(tA_row + 1 * 32) * AStride + tA_col * 4]);
      tArA[2] =
          FETCH_CONST_FLOAT4(gA[(tA_row + 2 * 32) * AStride + tA_col * 4]);
      tArA[3] =
          FETCH_CONST_FLOAT4(gA[(tA_row + 3 * 32) * AStride + tA_col * 4]);

      sA[(tA_col * 4 + 0) * ASmemStride + (0 * 32 + tA_row)] = tArA[0].x;
      sA[(tA_col * 4 + 1) * ASmemStride + (0 * 32 + tA_row)] = tArA[0].y;
      sA[(tA_col * 4 + 2) * ASmemStride + (0 * 32 + tA_row)] = tArA[0].z;
      sA[(tA_col * 4 + 3) * ASmemStride + (0 * 32 + tA_row)] = tArA[0].w;

      sA[(tA_col * 4 + 0) * ASmemStride + (1 * 32 + tA_row)] = tArA[1].x;
      sA[(tA_col * 4 + 1) * ASmemStride + (1 * 32 + tA_row)] = tArA[1].y;
      sA[(tA_col * 4 + 2) * ASmemStride + (1 * 32 + tA_row)] = tArA[1].z;
      sA[(tA_col * 4 + 3) * ASmemStride + (1 * 32 + tA_row)] = tArA[1].w;

      sA[(tA_col * 4 + 0) * ASmemStride + (2 * 32 + tA_row)] = tArA[2].x;
      sA[(tA_col * 4 + 1) * ASmemStride + (2 * 32 + tA_row)] = tArA[2].y;
      sA[(tA_col * 4 + 2) * ASmemStride + (2 * 32 + tA_row)] = tArA[2].z;
      sA[(tA_col * 4 + 3) * ASmemStride + (2 * 32 + tA_row)] = tArA[2].w;

      sA[(tA_col * 4 + 0) * ASmemStride + (3 * 32 + tA_row)] = tArA[3].x;
      sA[(tA_col * 4 + 1) * ASmemStride + (3 * 32 + tA_row)] = tArA[3].y;
      sA[(tA_col * 4 + 2) * ASmemStride + (3 * 32 + tA_row)] = tArA[3].z;
      sA[(tA_col * 4 + 3) * ASmemStride + (3 * 32 + tA_row)] = tArA[3].w;

      tBrB[0] = FETCH_CONST_FLOAT4(gB[(tB_row + 0 * 8) * BStride + tB_col * 4]);
      tBrB[1] = FETCH_CONST_FLOAT4(gB[(tB_row + 1 * 8) * BStride + tB_col * 4]);
      tBrB[2] = FETCH_CONST_FLOAT4(gB[(tB_row + 2 * 8) * BStride + tB_col * 4]);
      tBrB[3] = FETCH_CONST_FLOAT4(gB[(tB_row + 3 * 8) * BStride + tB_col * 4]);

      FETCH_FLOAT4(sB[(tB_row + 0 * 8) * BSmemStride + (tB_col * 4 + 0)]) =
          tBrB[0];

      FETCH_FLOAT4(sB[(tB_row + 1 * 8) * BSmemStride + (tB_col * 4 + 0)]) =
          tBrB[1];

      FETCH_FLOAT4(sB[(tB_row + 2 * 8) * BSmemStride + (tB_col * 4 + 0)]) =
          tBrB[2];

      FETCH_FLOAT4(sB[(tB_row + 3 * 8) * BSmemStride + (tB_col * 4 + 0)]) =
          tBrB[3];
    }

    sA = stages[smem_pipe_read].A;
    sB = stages[smem_pipe_read].B;

#pragma unroll
    for (int k_block = 0; k_block < K_BLOCK_MAX; k_block++) {

      int k_block_next = (k_block + 1) % K_BLOCK_MAX;
      if (k_block == K_BLOCK_MAX - 1) {
        __syncthreads();
        sA = stages[smem_pipe_write].A;
        sB = stages[smem_pipe_write].B;
      }

      tCrA[rmem_pipe_write][0][0] = FETCH_FLOAT4(
          sA[(k_block_next * MMA_K + 0) * ASmemStride + tC_row * 4 + 0 * 64]);
      tCrA[rmem_pipe_write][0][1] = FETCH_FLOAT4(
          sA[(k_block_next * MMA_K + 1) * ASmemStride + tC_row * 4 + 0 * 64]);
      tCrA[rmem_pipe_write][0][2] = FETCH_FLOAT4(
          sA[(k_block_next * MMA_K + 2) * ASmemStride + tC_row * 4 + 0 * 64]);
      tCrA[rmem_pipe_write][0][3] = FETCH_FLOAT4(
          sA[(k_block_next * MMA_K + 3) * ASmemStride + tC_row * 4 + 0 * 64]);
      tCrA[rmem_pipe_write][0][4] = FETCH_FLOAT4(
          sA[(k_block_next * MMA_K + 4) * ASmemStride + tC_row * 4 + 0 * 64]);
      tCrA[rmem_pipe_write][0][5] = FETCH_FLOAT4(
          sA[(k_block_next * MMA_K + 5) * ASmemStride + tC_row * 4 + 0 * 64]);
      tCrA[rmem_pipe_write][0][6] = FETCH_FLOAT4(
          sA[(k_block_next * MMA_K + 6) * ASmemStride + tC_row * 4 + 0 * 64]);
      tCrA[rmem_pipe_write][0][7] = FETCH_FLOAT4(
          sA[(k_block_next * MMA_K + 7) * ASmemStride + tC_row * 4 + 0 * 64]);

      tCrA[rmem_pipe_write][1][0] = FETCH_FLOAT4(
          sA[(k_block_next * MMA_K + 0) * ASmemStride + tC_row * 4 + 1 * 64]);
      tCrA[rmem_pipe_write][1][1] = FETCH_FLOAT4(
          sA[(k_block_next * MMA_K + 1) * ASmemStride + tC_row * 4 + 1 * 64]);
      tCrA[rmem_pipe_write][1][2] = FETCH_FLOAT4(
          sA[(k_block_next * MMA_K + 2) * ASmemStride + tC_row * 4 + 1 * 64]);
      tCrA[rmem_pipe_write][1][3] = FETCH_FLOAT4(
          sA[(k_block_next * MMA_K + 3) * ASmemStride + tC_row * 4 + 1 * 64]);
      tCrA[rmem_pipe_write][1][4] = FETCH_FLOAT4(
          sA[(k_block_next * MMA_K + 4) * ASmemStride + tC_row * 4 + 1 * 64]);
      tCrA[rmem_pipe_write][1][5] = FETCH_FLOAT4(
          sA[(k_block_next * MMA_K + 5) * ASmemStride + tC_row * 4 + 1 * 64]);
      tCrA[rmem_pipe_write][1][6] = FETCH_FLOAT4(
          sA[(k_block_next * MMA_K + 6) * ASmemStride + tC_row * 4 + 1 * 64]);
      tCrA[rmem_pipe_write][1][7] = FETCH_FLOAT4(
          sA[(k_block_next * MMA_K + 7) * ASmemStride + tC_row * 4 + 1 * 64]);

      tCrB[rmem_pipe_write][0][0] = FETCH_FLOAT4(
          sB[(k_block_next * MMA_K + 0) * BSmemStride + tC_col * 4 + 0 * 64]);
      tCrB[rmem_pipe_write][0][1] = FETCH_FLOAT4(
          sB[(k_block_next * MMA_K + 1) * BSmemStride + tC_col * 4 + 0 * 64]);
      tCrB[rmem_pipe_write][0][2] = FETCH_FLOAT4(
          sB[(k_block_next * MMA_K + 2) * BSmemStride + tC_col * 4 + 0 * 64]);
      tCrB[rmem_pipe_write][0][3] = FETCH_FLOAT4(
          sB[(k_block_next * MMA_K + 3) * BSmemStride + tC_col * 4 + 0 * 64]);
      tCrB[rmem_pipe_write][0][4] = FETCH_FLOAT4(
          sB[(k_block_next * MMA_K + 4) * BSmemStride + tC_col * 4 + 0 * 64]);
      tCrB[rmem_pipe_write][0][5] = FETCH_FLOAT4(
          sB[(k_block_next * MMA_K + 5) * BSmemStride + tC_col * 4 + 0 * 64]);
      tCrB[rmem_pipe_write][0][6] = FETCH_FLOAT4(
          sB[(k_block_next * MMA_K + 6) * BSmemStride + tC_col * 4 + 0 * 64]);
      tCrB[rmem_pipe_write][0][7] = FETCH_FLOAT4(
          sB[(k_block_next * MMA_K + 7) * BSmemStride + tC_col * 4 + 0 * 64]);

      tCrB[rmem_pipe_write][1][0] = FETCH_FLOAT4(
          sB[(k_block_next * MMA_K + 0) * BSmemStride + tC_col * 4 + 1 * 64]);
      tCrB[rmem_pipe_write][1][1] = FETCH_FLOAT4(
          sB[(k_block_next * MMA_K + 1) * BSmemStride + tC_col * 4 + 1 * 64]);
      tCrB[rmem_pipe_write][1][2] = FETCH_FLOAT4(
          sB[(k_block_next * MMA_K + 2) * BSmemStride + tC_col * 4 + 1 * 64]);
      tCrB[rmem_pipe_write][1][3] = FETCH_FLOAT4(
          sB[(k_block_next * MMA_K + 3) * BSmemStride + tC_col * 4 + 1 * 64]);
      tCrB[rmem_pipe_write][1][4] = FETCH_FLOAT4(
          sB[(k_block_next * MMA_K + 4) * BSmemStride + tC_col * 4 + 1 * 64]);
      tCrB[rmem_pipe_write][1][5] = FETCH_FLOAT4(
          sB[(k_block_next * MMA_K + 5) * BSmemStride + tC_col * 4 + 1 * 64]);
      tCrB[rmem_pipe_write][1][6] = FETCH_FLOAT4(
          sB[(k_block_next * MMA_K + 6) * BSmemStride + tC_col * 4 + 1 * 64]);
      tCrB[rmem_pipe_write][1][7] = FETCH_FLOAT4(
          sB[(k_block_next * MMA_K + 7) * BSmemStride + tC_col * 4 + 1 * 64]);

      COMPUTE_1X8(tCrC[0], tCrA[rmem_pipe_read][0], tCrB[rmem_pipe_read][0],
                  tCrB[rmem_pipe_read][1], x);
      COMPUTE_1X8(tCrC[1], tCrA[rmem_pipe_read][0], tCrB[rmem_pipe_read][0],
                  tCrB[rmem_pipe_read][1], y);
      COMPUTE_1X8(tCrC[2], tCrA[rmem_pipe_read][0], tCrB[rmem_pipe_read][0],
                  tCrB[rmem_pipe_read][1], z);
      COMPUTE_1X8(tCrC[3], tCrA[rmem_pipe_read][0], tCrB[rmem_pipe_read][0],
                  tCrB[rmem_pipe_read][1], w);

      COMPUTE_1X8(tCrC[4], tCrA[rmem_pipe_read][1], tCrB[rmem_pipe_read][0],
                  tCrB[rmem_pipe_read][1], x);
      COMPUTE_1X8(tCrC[5], tCrA[rmem_pipe_read][1], tCrB[rmem_pipe_read][0],
                  tCrB[rmem_pipe_read][1], y);
      COMPUTE_1X8(tCrC[6], tCrA[rmem_pipe_read][1], tCrB[rmem_pipe_read][0],
                  tCrB[rmem_pipe_read][1], z);
      COMPUTE_1X8(tCrC[7], tCrA[rmem_pipe_read][1], tCrB[rmem_pipe_read][0],
                  tCrB[rmem_pipe_read][1], w);

      rmem_pipe_read ^= 1;
      rmem_pipe_write ^= 1;
    }

    smem_pipe_read ^= 1;
    smem_pipe_write ^= 1;
  }

  FETCH_FLOAT4(gC[(0 * 64 + tC_row * 4 + 0) * CStride + 0 * 64 + tC_col * 4]) =
      FETCH_FLOAT4(tCrC[0][0]);

  FETCH_FLOAT4(gC[(0 * 64 + tC_row * 4 + 1) * CStride + 0 * 64 + tC_col * 4]) =
      FETCH_FLOAT4(tCrC[1][0]);

  FETCH_FLOAT4(gC[(0 * 64 + tC_row * 4 + 2) * CStride + 0 * 64 + tC_col * 4]) =
      FETCH_FLOAT4(tCrC[2][0]);

  FETCH_FLOAT4(gC[(0 * 64 + tC_row * 4 + 3) * CStride + 0 * 64 + tC_col * 4]) =
      FETCH_FLOAT4(tCrC[3][0]);

  FETCH_FLOAT4(gC[(0 * 64 + tC_row * 4 + 0) * CStride + 1 * 64 + tC_col * 4]) =
      FETCH_FLOAT4(tCrC[0][4]);

  FETCH_FLOAT4(gC[(0 * 64 + tC_row * 4 + 1) * CStride + 1 * 64 + tC_col * 4]) =
      FETCH_FLOAT4(tCrC[1][4]);

  FETCH_FLOAT4(gC[(0 * 64 + tC_row * 4 + 2) * CStride + 1 * 64 + tC_col * 4]) =
      FETCH_FLOAT4(tCrC[2][4]);

  FETCH_FLOAT4(gC[(0 * 64 + tC_row * 4 + 3) * CStride + 1 * 64 + tC_col * 4]) =
      FETCH_FLOAT4(tCrC[3][4]);

  FETCH_FLOAT4(gC[(1 * 64 + tC_row * 4 + 0) * CStride + 0 * 64 + tC_col * 4]) =
      FETCH_FLOAT4(tCrC[4][0]);

  FETCH_FLOAT4(gC[(1 * 64 + tC_row * 4 + 1) * CStride + 0 * 64 + tC_col * 4]) =
      FETCH_FLOAT4(tCrC[5][0]);

  FETCH_FLOAT4(gC[(1 * 64 + tC_row * 4 + 2) * CStride + 0 * 64 + tC_col * 4]) =
      FETCH_FLOAT4(tCrC[6][0]);

  FETCH_FLOAT4(gC[(1 * 64 + tC_row * 4 + 3) * CStride + 0 * 64 + tC_col * 4]) =
      FETCH_FLOAT4(tCrC[7][0]);

  FETCH_FLOAT4(gC[(1 * 64 + tC_row * 4 + 0) * CStride + 1 * 64 + tC_col * 4]) =
      FETCH_FLOAT4(tCrC[4][4]);

  FETCH_FLOAT4(gC[(1 * 64 + tC_row * 4 + 1) * CStride + 1 * 64 + tC_col * 4]) =
      FETCH_FLOAT4(tCrC[5][4]);

  FETCH_FLOAT4(gC[(1 * 64 + tC_row * 4 + 2) * CStride + 1 * 64 + tC_col * 4]) =
      FETCH_FLOAT4(tCrC[6][4]);

  FETCH_FLOAT4(gC[(1 * 64 + tC_row * 4 + 3) * CStride + 1 * 64 + tC_col * 4]) =
      FETCH_FLOAT4(tCrC[7][4]);
}

} // namespace low_occupancy
