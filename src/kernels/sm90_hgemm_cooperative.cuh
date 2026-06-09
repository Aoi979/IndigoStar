#pragma once
#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cluster.cuh"

namespace sm90_hgemm_cooperative {

struct SharedStorage {
  alignas(128) Element A[kBlockM * kBlockK * kStages];
  alignas(128) Element B[kBlockK * kBlockN * kStages];
  alignas(128) Element C[kBlockM * kBlockN];
};
inline cudaError_t map_cu_result(CUresult result) {
  return result == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}
template <int BlockMajor, int BlockMinor>
inline cudaError_t make_row_major_tensor_map(CUtensorMap *map,
                                             Element const *ptr, int height,
                                             int width) {
  static_assert(BlockMinor >= 64);
  static_assert(BlockMinor % 64 == 0);
  if (map == nullptr || ptr == nullptr || height <= 0 || width <= 0 ||
      width % 64 != 0) {
    return cudaErrorInvalidValue;
  }

  uint64_t shape[] = {64, static_cast<uint64_t>(height),
                      static_cast<uint64_t>(width / 64)};
  uint64_t stride[] = {static_cast<uint64_t>(sizeof(Element)) *
                           static_cast<uint64_t>(width),
                       64ull * sizeof(Element)};
  uint32_t box_shape[] = {64u, static_cast<uint32_t>(BlockMajor),
                          static_cast<uint32_t>(BlockMinor / 64)};
  uint32_t box_stride[] = {1u, 1u, 1u};

  CUresult result = cuTensorMapEncodeTiled(
      map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, const_cast<Element *>(ptr),
      shape, stride, box_shape, box_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return map_cu_result(result);
}

template <int BlockN, int BlockK>
inline cudaError_t make_b_row_major_tensor_map(CUtensorMap *map,
                                               Element const *ptr,
                                               int n,
                                               int k) {
  static_assert(BlockN >= 64);
  static_assert(BlockN % 64 == 0);
  static_assert(BlockK >= 64);
  if (map == nullptr || ptr == nullptr || n <= 0 || k <= 0 ||
      n % 64 != 0) {
    return cudaErrorInvalidValue;
  }

  // Logical tensor is B(n, k) with stride (1, N), backed by row-major B[k][n].
  // Keep the innermost TMA dimension at 64 half elements so each slice is a
  // 128B contiguous segment along the physical N-major access direction.
  uint64_t shape[] = {64, static_cast<uint64_t>(n / 64),
                      static_cast<uint64_t>(k)};
  uint64_t stride[] = {64ull * sizeof(Element),
                       static_cast<uint64_t>(n) * sizeof(Element)};
  uint32_t box_shape[] = {64u, static_cast<uint32_t>(BlockN / 64),
                          static_cast<uint32_t>(BlockK)};
  uint32_t box_stride[] = {1u, 1u, 1u};

  CUresult result = cuTensorMapEncodeTiled(
      map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3,
      const_cast<Element *>(ptr), shape, stride, box_shape, box_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return map_cu_result(result);
}
} // namespace sm90_hgemm_cooperative