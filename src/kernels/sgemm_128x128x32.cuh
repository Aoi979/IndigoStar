#include "sgemm_common.cuh"

namespace custom::detail {
__device__ __forceinline__ unsigned smem_addr(const void *ptr) {
  return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ int swizzle_a_row_128x128x32(int logical_row,
                                                        int k_index) {
  int row_group = logical_row >> 2;
  int row_inner = logical_row & 3;
  int swizzled_group = (row_group & ~7) + ((row_group + (k_index >> 2)) & 7);
  return (swizzled_group << 2) + row_inner;
}
} // namespace custom::detail

__global__ void sgemm_128x128x32(int M, int N, int K,
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

  __shared__ float smem[ASmemNumel + BSmemNumel];
  float *sA = smem;
  float *sB = smem + ASmemNumel;

  int K_TILE_MAX = K / kBK;
  int K_BLOCK_MAX = kBK / MMA_K;

  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
    const float *gA = gA_base + k_tile * kBK;
    const float *gB = gB_base + k_tile * kBK * BStride;

    int load_A_row = tid / (kBK / 4);
    int load_A_col = tid % (kBK / 4);

    float4 tArA[4];
    tArA[0] = FETCH_CONST_FLOAT4(
        gA[(load_A_row + 0 * 32) * AStride + load_A_col * 4]);
    tArA[1] = FETCH_CONST_FLOAT4(
        gA[(load_A_row + 1 * 32) * AStride + load_A_col * 4]);
    tArA[2] = FETCH_CONST_FLOAT4(
        gA[(load_A_row + 2 * 32) * AStride + load_A_col * 4]);
    tArA[3] = FETCH_CONST_FLOAT4(
        gA[(load_A_row + 3 * 32) * AStride + load_A_col * 4]);

    int load_A_k = load_A_col * 4;
    int load_A_row_swizzled =
        ((((load_A_row >> 2) + load_A_col) & 7) << 2) + (load_A_row & 3);
    int load_A_row0 = 0 * 32 + load_A_row_swizzled;
    int load_A_row1 = 1 * 32 + load_A_row_swizzled;
    int load_A_row2 = 2 * 32 + load_A_row_swizzled;
    int load_A_row3 = 3 * 32 + load_A_row_swizzled;

    sA[(load_A_k + 0) * ASmemStride + load_A_row0] = tArA[0].x;
    sA[(load_A_k + 1) * ASmemStride + load_A_row0] = tArA[0].y;
    sA[(load_A_k + 2) * ASmemStride + load_A_row0] = tArA[0].z;
    sA[(load_A_k + 3) * ASmemStride + load_A_row0] = tArA[0].w;

    sA[(load_A_k + 0) * ASmemStride + load_A_row1] = tArA[1].x;
    sA[(load_A_k + 1) * ASmemStride + load_A_row1] = tArA[1].y;
    sA[(load_A_k + 2) * ASmemStride + load_A_row1] = tArA[1].z;
    sA[(load_A_k + 3) * ASmemStride + load_A_row1] = tArA[1].w;

    sA[(load_A_k + 0) * ASmemStride + load_A_row2] = tArA[2].x;
    sA[(load_A_k + 1) * ASmemStride + load_A_row2] = tArA[2].y;
    sA[(load_A_k + 2) * ASmemStride + load_A_row2] = tArA[2].z;
    sA[(load_A_k + 3) * ASmemStride + load_A_row2] = tArA[2].w;

    sA[(load_A_k + 0) * ASmemStride + load_A_row3] = tArA[3].x;
    sA[(load_A_k + 1) * ASmemStride + load_A_row3] = tArA[3].y;
    sA[(load_A_k + 2) * ASmemStride + load_A_row3] = tArA[3].z;
    sA[(load_A_k + 3) * ASmemStride + load_A_row3] = tArA[3].w;

    int load_B_row = tid / (kBN / 4);
    int load_B_col = tid % (kBN / 4);

    float4 tBrB[4];
    tBrB[0] =
        FETCH_CONST_FLOAT4(gB[(load_B_row + 0 * 8) * BStride + load_B_col * 4]);
    tBrB[1] =
        FETCH_CONST_FLOAT4(gB[(load_B_row + 1 * 8) * BStride + load_B_col * 4]);
    tBrB[2] =
        FETCH_CONST_FLOAT4(gB[(load_B_row + 2 * 8) * BStride + load_B_col * 4]);
    tBrB[3] =
        FETCH_CONST_FLOAT4(gB[(load_B_row + 3 * 8) * BStride + load_B_col * 4]);

    FETCH_FLOAT4(
        sB[(load_B_row + 0 * 8) * BSmemStride + (load_B_col * 4 + 0)]) =
        tBrB[0];

    FETCH_FLOAT4(
        sB[(load_B_row + 1 * 8) * BSmemStride + (load_B_col * 4 + 0)]) =
        tBrB[1];

    FETCH_FLOAT4(
        sB[(load_B_row + 2 * 8) * BSmemStride + (load_B_col * 4 + 0)]) =
        tBrB[2];

    FETCH_FLOAT4(
        sB[(load_B_row + 3 * 8) * BSmemStride + (load_B_col * 4 + 0)]) =
        tBrB[3];

    __syncthreads();

    float4 tCrA[2][MMA_K];
    float4 tCrB[2][MMA_K];
// #pragma unroll
    for (int k_block = 0; k_block < kBK; k_block += MMA_K) {
      int k_group0 = (k_block >> 2) & 7;
      int k_group1 = ((k_block + 4) >> 2) & 7;
      int a_row_group0 = tC_row;
      int a_row_group1 = tC_row + 16;
      int a_row0_g0 =
          (((a_row_group0 & ~7) + ((a_row_group0 + k_group0) & 7)) << 2);
      int a_row0_g1 =
          (((a_row_group0 & ~7) + ((a_row_group0 + k_group1) & 7)) << 2);
      int a_row1_g0 =
          (((a_row_group1 & ~7) + ((a_row_group1 + k_group0) & 7)) << 2);
      int a_row1_g1 =
          (((a_row_group1 & ~7) + ((a_row_group1 + k_group1) & 7)) << 2);

      tCrA[0][0] = FETCH_FLOAT4(sA[(k_block + 0) * ASmemStride + a_row0_g0]);
      tCrA[0][1] = FETCH_FLOAT4(sA[(k_block + 1) * ASmemStride + a_row0_g0]);
      tCrA[0][2] = FETCH_FLOAT4(sA[(k_block + 2) * ASmemStride + a_row0_g0]);
      tCrA[0][3] = FETCH_FLOAT4(sA[(k_block + 3) * ASmemStride + a_row0_g0]);
      tCrA[0][4] = FETCH_FLOAT4(sA[(k_block + 4) * ASmemStride + a_row0_g1]);
      tCrA[0][5] = FETCH_FLOAT4(sA[(k_block + 5) * ASmemStride + a_row0_g1]);
      tCrA[0][6] = FETCH_FLOAT4(sA[(k_block + 6) * ASmemStride + a_row0_g1]);
      tCrA[0][7] = FETCH_FLOAT4(sA[(k_block + 7) * ASmemStride + a_row0_g1]);

      tCrA[1][0] = FETCH_FLOAT4(sA[(k_block + 0) * ASmemStride + a_row1_g0]);
      tCrA[1][1] = FETCH_FLOAT4(sA[(k_block + 1) * ASmemStride + a_row1_g0]);
      tCrA[1][2] = FETCH_FLOAT4(sA[(k_block + 2) * ASmemStride + a_row1_g0]);
      tCrA[1][3] = FETCH_FLOAT4(sA[(k_block + 3) * ASmemStride + a_row1_g0]);
      tCrA[1][4] = FETCH_FLOAT4(sA[(k_block + 4) * ASmemStride + a_row1_g1]);
      tCrA[1][5] = FETCH_FLOAT4(sA[(k_block + 5) * ASmemStride + a_row1_g1]);
      tCrA[1][6] = FETCH_FLOAT4(sA[(k_block + 6) * ASmemStride + a_row1_g1]);
      tCrA[1][7] = FETCH_FLOAT4(sA[(k_block + 7) * ASmemStride + a_row1_g1]);

      tCrB[0][0] =
          FETCH_FLOAT4(sB[(k_block + 0) * BSmemStride + tC_col * 4 + 0 * 64]);
      tCrB[0][1] =
          FETCH_FLOAT4(sB[(k_block + 1) * BSmemStride + tC_col * 4 + 0 * 64]);
      tCrB[0][2] =
          FETCH_FLOAT4(sB[(k_block + 2) * BSmemStride + tC_col * 4 + 0 * 64]);
      tCrB[0][3] =
          FETCH_FLOAT4(sB[(k_block + 3) * BSmemStride + tC_col * 4 + 0 * 64]);
      tCrB[0][4] =
          FETCH_FLOAT4(sB[(k_block + 4) * BSmemStride + tC_col * 4 + 0 * 64]);
      tCrB[0][5] =
          FETCH_FLOAT4(sB[(k_block + 5) * BSmemStride + tC_col * 4 + 0 * 64]);
      tCrB[0][6] =
          FETCH_FLOAT4(sB[(k_block + 6) * BSmemStride + tC_col * 4 + 0 * 64]);
      tCrB[0][7] =
          FETCH_FLOAT4(sB[(k_block + 7) * BSmemStride + tC_col * 4 + 0 * 64]);

      tCrB[1][0] =
          FETCH_FLOAT4(sB[(k_block + 0) * BSmemStride + tC_col * 4 + 1 * 64]);
      tCrB[1][1] =
          FETCH_FLOAT4(sB[(k_block + 1) * BSmemStride + tC_col * 4 + 1 * 64]);
      tCrB[1][2] =
          FETCH_FLOAT4(sB[(k_block + 2) * BSmemStride + tC_col * 4 + 1 * 64]);
      tCrB[1][3] =
          FETCH_FLOAT4(sB[(k_block + 3) * BSmemStride + tC_col * 4 + 1 * 64]);
      tCrB[1][4] =
          FETCH_FLOAT4(sB[(k_block + 4) * BSmemStride + tC_col * 4 + 1 * 64]);
      tCrB[1][5] =
          FETCH_FLOAT4(sB[(k_block + 5) * BSmemStride + tC_col * 4 + 1 * 64]);
      tCrB[1][6] =
          FETCH_FLOAT4(sB[(k_block + 6) * BSmemStride + tC_col * 4 + 1 * 64]);
      tCrB[1][7] =
          FETCH_FLOAT4(sB[(k_block + 7) * BSmemStride + tC_col * 4 + 1 * 64]);

      COMPUTE_1X8(tCrC[0], tCrA[0], tCrB[0], tCrB[1], x);
      COMPUTE_1X8(tCrC[1], tCrA[0], tCrB[0], tCrB[1], y);
      COMPUTE_1X8(tCrC[2], tCrA[0], tCrB[0], tCrB[1], z);
      COMPUTE_1X8(tCrC[3], tCrA[0], tCrB[0], tCrB[1], w);

      COMPUTE_1X8(tCrC[4], tCrA[1], tCrB[0], tCrB[1], x);
      COMPUTE_1X8(tCrC[5], tCrA[1], tCrB[0], tCrB[1], y);
      COMPUTE_1X8(tCrC[6], tCrA[1], tCrB[0], tCrB[1], z);
      COMPUTE_1X8(tCrC[7], tCrA[1], tCrB[0], tCrB[1], w);
    }

    __syncthreads();
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

