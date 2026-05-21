#include "sgemm_common.cuh"
__global__ void sgemm_128x128x32_K2(int M, int N, int K,
                                    float const *__restrict__ A,
                                    float const *__restrict__ B,
                                    float *__restrict__ C) {
  constexpr int kBM = 128;
  constexpr int kBN = 128;
  constexpr int kBK = 32;

  constexpr int MMA_M = 1;
  constexpr int MMA_N = 1;
  constexpr int MMA_K = 2;

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

    sA[(load_A_col * 4 + 0) * ASmemStride + (0 * 32 + load_A_row)] = tArA[0].x;
    sA[(load_A_col * 4 + 1) * ASmemStride + (0 * 32 + load_A_row)] = tArA[0].y;
    sA[(load_A_col * 4 + 2) * ASmemStride + (0 * 32 + load_A_row)] = tArA[0].z;
    sA[(load_A_col * 4 + 3) * ASmemStride + (0 * 32 + load_A_row)] = tArA[0].w;

    sA[(load_A_col * 4 + 0) * ASmemStride + (1 * 32 + load_A_row)] = tArA[1].x;
    sA[(load_A_col * 4 + 1) * ASmemStride + (1 * 32 + load_A_row)] = tArA[1].y;
    sA[(load_A_col * 4 + 2) * ASmemStride + (1 * 32 + load_A_row)] = tArA[1].z;
    sA[(load_A_col * 4 + 3) * ASmemStride + (1 * 32 + load_A_row)] = tArA[1].w;

    sA[(load_A_col * 4 + 0) * ASmemStride + (2 * 32 + load_A_row)] = tArA[2].x;
    sA[(load_A_col * 4 + 1) * ASmemStride + (2 * 32 + load_A_row)] = tArA[2].y;
    sA[(load_A_col * 4 + 2) * ASmemStride + (2 * 32 + load_A_row)] = tArA[2].z;
    sA[(load_A_col * 4 + 3) * ASmemStride + (2 * 32 + load_A_row)] = tArA[2].w;

    sA[(load_A_col * 4 + 0) * ASmemStride + (3 * 32 + load_A_row)] = tArA[3].x;
    sA[(load_A_col * 4 + 1) * ASmemStride + (3 * 32 + load_A_row)] = tArA[3].y;
    sA[(load_A_col * 4 + 2) * ASmemStride + (3 * 32 + load_A_row)] = tArA[3].z;
    sA[(load_A_col * 4 + 3) * ASmemStride + (3 * 32 + load_A_row)] = tArA[3].w;

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

    for (int k_block = 0; k_block < kBK; k_block += MMA_K) {

      tCrA[0][0] =
          FETCH_FLOAT4(sA[(k_block + 0) * ASmemStride + tC_row * 4 + 0 * 64]);
      tCrA[0][1] =
          FETCH_FLOAT4(sA[(k_block + 1) * ASmemStride + tC_row * 4 + 0 * 64]);

      tCrA[1][0] =
          FETCH_FLOAT4(sA[(k_block + 0) * ASmemStride + tC_row * 4 + 1 * 64]);
      tCrA[1][1] =
          FETCH_FLOAT4(sA[(k_block + 1) * ASmemStride + tC_row * 4 + 1 * 64]);

      tCrB[0][0] =
          FETCH_FLOAT4(sB[(k_block + 0) * BSmemStride + tC_col * 4 + 0 * 64]);
      tCrB[0][1] =
          FETCH_FLOAT4(sB[(k_block + 1) * BSmemStride + tC_col * 4 + 0 * 64]);

      tCrB[1][0] =
          FETCH_FLOAT4(sB[(k_block + 0) * BSmemStride + tC_col * 4 + 1 * 64]);
      tCrB[1][1] =
          FETCH_FLOAT4(sB[(k_block + 1) * BSmemStride + tC_col * 4 + 1 * 64]);

      COMPUTE_1X8_K2(tCrC[0], tCrA[0], tCrB[0], tCrB[1], x);
      COMPUTE_1X8_K2(tCrC[1], tCrA[0], tCrB[0], tCrB[1], y);
      COMPUTE_1X8_K2(tCrC[2], tCrA[0], tCrB[0], tCrB[1], z);
      COMPUTE_1X8_K2(tCrC[3], tCrA[0], tCrB[0], tCrB[1], w);

      COMPUTE_1X8_K2(tCrC[4], tCrA[1], tCrB[0], tCrB[1], x);
      COMPUTE_1X8_K2(tCrC[5], tCrA[1], tCrB[0], tCrB[1], y);
      COMPUTE_1X8_K2(tCrC[6], tCrA[1], tCrB[0], tCrB[1], z);
      COMPUTE_1X8_K2(tCrC[7], tCrA[1], tCrB[0], tCrB[1], w);
    }
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
