#pragma once

#include <cuda_runtime.h>

#include "cute_ampere_16816.cuh"

namespace cute_hgemm {

inline cudaError_t launch_hgemm_128x128_nn(const cute::half_t *A,
                                           const cute::half_t *B,
                                           cute::half_t *C,
                                           int M, int N, int K,
                                           cudaStream_t stream = 0) {
  using namespace cute;

  auto prob_shape = make_shape(M, N, K);

  auto dA = make_stride(K, Int<1>{});
  auto dB = make_stride(Int<1>{}, N);
  auto dC = make_stride(N, Int<1>{});

  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<64>{};
  auto cta_tiler = make_shape(bM, bN, bK);
  auto bP = Int<3>{};

  auto swizzle_atom_a = composition(
      Swizzle<3, 3, 3>{},
      Layout<Shape<_8, Shape<_8, _8>>, Stride<_8, Stride<_1, _64>>>{});

  auto swizzle_atom_b = composition(
      Swizzle<3, 3, 3>{},
      Layout<Shape<Shape<_8, _16>, _8>, Stride<Stride<_1, _64>, _8>>{});

  auto sA = tile_to_shape(swizzle_atom_a, make_shape(bM, bK, bP));
  auto sB = tile_to_shape(swizzle_atom_b, make_shape(bN, bK, bP));
  auto sC = make_layout(make_shape(bM, bN), make_stride(Int<136>{}, Int<1>{}));

  TiledCopy copyA = make_tiled_copy(
      cute::Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
      Layout<Shape<_16, _8>, Stride<_8, _1>>{},
      Layout<Shape<_1, _8>>{});
  TiledCopy copyB = make_tiled_copy(
      cute::Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
      Layout<Shape<_16, _8>, Stride<_1, _16>>{},
      Layout<Shape<_8, _1>>{});

  TiledMMA mmaC = make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
                                 Layout<Shape<_2, _2>>{},
                                 Tile<_32, _32, _16>{});

  cute::Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_A;
  cute::Copy_Atom<SM75_U16x8_LDSM_T, half_t> s2r_atom_B;

  float alpha = 1.0f;
  float beta = 0.0f;

  auto kernel_fptr = cute_ampere_hgemm_16816<
      decltype(prob_shape), decltype(cta_tiler), cute::half_t, decltype(dA),
      decltype(sA), decltype(copyA), decltype(s2r_atom_A), cute::half_t,
      decltype(dB), decltype(sB), decltype(copyB), decltype(s2r_atom_B),
      cute::half_t, decltype(dC), decltype(sC), decltype(mmaC),
      decltype(alpha), decltype(beta)>;

  int smem_size = int(sizeof(
      CuteHgemmSharedStorage<cute::half_t, cute::half_t, decltype(sA),
                             decltype(sB)>));

  cudaError_t err = cudaFuncSetAttribute(
      kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  if (err != cudaSuccess) return err;

  err = cudaFuncSetAttribute(kernel_fptr,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             100);
  if (err != cudaSuccess) return err;

  // Block swizzle: adjacent CTAs in the same strip reuse A rows.
  // Use exact division when tile_n_count is a multiple of swizzle to avoid
  // empty CTAs; otherwise fall back to ceil_div.
  constexpr int kBlockSwizzle = 8;
  int tile_m_count = size(ceil_div(M, bM));
  int tile_n_count = size(ceil_div(N, bN));
  dim3 dimBlock(size(mmaC));
  int grid_y = (tile_n_count % kBlockSwizzle == 0)
                   ? (tile_n_count / kBlockSwizzle)
                   : (tile_n_count + kBlockSwizzle - 1) / kBlockSwizzle;
  dim3 dimGrid(tile_m_count * kBlockSwizzle, grid_y);

  kernel_fptr<<<dimGrid, dimBlock, smem_size, stream>>>(
      prob_shape, cta_tiler, A, dA, sA, copyA, s2r_atom_A, B, dB, sB, copyB,
      s2r_atom_B, C, dC, sC, mmaC, alpha, beta, kBlockSwizzle);

  return cudaGetLastError();
}

} // namespace cute_hgemm
