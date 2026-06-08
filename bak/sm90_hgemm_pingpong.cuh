#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace sm90_hgemm_pingpong {

struct GemmTile {
  int m = -1;
  int n = -1;
  int batch = 0;
  bool valid = false;
};

namespace detail {

__host__ __device__ constexpr uint32_t ceil_div(uint32_t a, uint32_t b) {
  return b == 0 ? 0 : (a + b - 1) / b;
}

} // namespace detail

// GEMM-only persistent tile scheduler for a pingpong kernel.
//
// This intentionally has no cluster or TMA multicast logic:
// pingpong assigns different C tiles to different consumer warp-groups over
// time, while TMA multicast is a cluster-level cooperative CTA feature.
struct PersistentTileSchedulerSm90Params {
  uint32_t tiles_m = 0;
  uint32_t tiles_n = 0;
  uint32_t batches = 1;
  uint64_t tiles_per_batch = 0;
  uint64_t total_tiles = 0;

  __host__ __device__ void initialize(int M, int N, int cta_m, int cta_n,
                                      int batch_count = 1) {
    tiles_m = detail::ceil_div(static_cast<uint32_t>(M),
                               static_cast<uint32_t>(cta_m));
    tiles_n = detail::ceil_div(static_cast<uint32_t>(N),
                               static_cast<uint32_t>(cta_n));
    batches = batch_count <= 0 ? 1 : static_cast<uint32_t>(batch_count);
    tiles_per_batch = uint64_t{tiles_m} * uint64_t{tiles_n};
    total_tiles = tiles_per_batch * uint64_t{batches};
  }

  __host__ __device__ GemmTile tile_for_work(uint64_t work_idx) const {
    if (work_idx >= total_tiles || tiles_per_batch == 0) {
      return {};
    }

    uint64_t batch = work_idx / tiles_per_batch;
    uint64_t tile = work_idx - batch * tiles_per_batch;
    uint32_t tile_m = static_cast<uint32_t>(tile / tiles_n);
    uint32_t tile_n =
        static_cast<uint32_t>(tile - uint64_t{tile_m} * tiles_n);

    return {static_cast<int>(tile_m), static_cast<int>(tile_n),
            static_cast<int>(batch), true};
  }

  __host__ __device__ static dim3 get_grid_shape(
      int M, int N, int cta_m, int cta_n, int sm_count,
      int batch_count = 1) {
    PersistentTileSchedulerSm90Params params;
    params.initialize(M, N, cta_m, cta_n, batch_count);

    if (params.total_tiles == 0) {
      return dim3{0, 1, 1};
    }

    uint64_t ctas_to_launch = params.total_tiles;
    if (sm_count > 0 && ctas_to_launch > static_cast<uint64_t>(sm_count)) {
      ctas_to_launch = static_cast<uint64_t>(sm_count);
    }

    return dim3{static_cast<uint32_t>(ctas_to_launch), 1, 1};
  }
};

class GemmPersistentTileScheduler {
public:
  __device__ explicit GemmPersistentTileScheduler(
      PersistentTileSchedulerSm90Params const &params)
      : params_(params), work_idx_(blockIdx.x), work_stride_(gridDim.x) {}

  __device__ GemmTile current() const {
    return params_.tile_for_work(work_idx_);
  }

  __device__ GemmTile next() {
    work_idx_ += work_stride_;
    return current();
  }

private:
  PersistentTileSchedulerSm90Params params_{};
  uint64_t work_idx_ = 0;
  uint64_t work_stride_ = 1;
};

inline int query_sm_count(int device = -1) {
  if (device < 0 && cudaGetDevice(&device) != cudaSuccess) {
    return 0;
  }

  int sm_count = 0;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  return sm_count;
}

} // namespace sm90_hgemm_pingpong
