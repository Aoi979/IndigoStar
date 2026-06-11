#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "../../cluster.cuh"

namespace sm90_hgemm_pingpong {

enum class RasterOrder { AlongM, AlongN };

enum class RasterOrderOptions { Heuristic, AlongM, AlongN };

struct GemmTile {
  int m = -1;
  int n = -1;
  int batch = -1;
  bool valid = false;     // Still inside the persistent scheduler stream.
  bool in_bounds = false; // Inside the original, unpadded M/N tile space.
};

namespace detail {

__host__ __device__ constexpr uint32_t ceil_div(uint32_t a, uint32_t b) {
  return b == 0 ? 0 : (a + b - 1) / b;
}

__host__ __device__ constexpr uint32_t ceil_div_pos(int a, int b) {
  return (a <= 0 || b <= 0)
             ? 0u
             : ceil_div(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
}

__host__ __device__ constexpr uint32_t round_up(uint32_t value,
                                                uint32_t multiple) {
  return multiple == 0 ? value : ceil_div(value, multiple) * multiple;
}

__host__ __device__ constexpr int min_int(int a, int b) {
  return a < b ? a : b;
}

__host__ __device__ constexpr uint32_t max_u32(uint32_t a, uint32_t b) {
  return a > b ? a : b;
}

} // namespace detail

// CUTLASS SM90 static persistent tile scheduler parameters, specialized for a
// GEMM-only pingpong kernel.
struct PersistentTileSchedulerSm90Params {
  uint32_t logical_tiles_m = 0;
  uint32_t logical_tiles_n = 0;
  uint32_t problem_blocks_m = 0;
  uint32_t problem_blocks_n = 0;
  uint32_t problem_blocks_l = 1;
  uint32_t cluster_shape_m = 1;
  uint32_t cluster_shape_n = 1;

  uint64_t blocks_per_batch = 0;
  uint64_t blocks_per_problem = 0;

  int32_t log_swizzle_size = 0;
  RasterOrder raster_order = RasterOrder::AlongN;

  __host__ __device__ static int32_t
  get_log_swizzle_size(uint32_t problem_ctas_m, uint32_t problem_ctas_n,
                       int max_swizzle_size) {
    uint32_t min_cta_dim =
        problem_ctas_m < problem_ctas_n ? problem_ctas_m : problem_ctas_n;
    if (max_swizzle_size >= 8 && min_cta_dim >= 6) {
      return 3;
    }
    if (max_swizzle_size >= 4 && min_cta_dim >= 3) {
      return 2;
    }
    if (max_swizzle_size >= 2 && min_cta_dim >= 2) {
      return 1;
    }
    return 0;
  }

  __host__ __device__ static RasterOrder
  get_rasterization_order(uint32_t tiles_m, uint32_t tiles_n,
                          RasterOrderOptions option) {
    if (option == RasterOrderOptions::Heuristic) {
      return tiles_n > tiles_m ? RasterOrder::AlongM : RasterOrder::AlongN;
    }
    return option == RasterOrderOptions::AlongN ? RasterOrder::AlongN
                                                : RasterOrder::AlongM;
  }

  __host__ __device__ static uint32_t get_max_cta_occupancy(int max_sm_per_gpc,
                                                            uint32_t cluster_m,
                                                            uint32_t cluster_n,
                                                            int sm_count) {
    uint32_t cluster_size = detail::max_u32(cluster_m * cluster_n, 1u);
    if (sm_count <= 0 || max_sm_per_gpc <= 0) {
      return 0;
    }

    int min_num_gpc = sm_count < max_sm_per_gpc ? 1 : sm_count / max_sm_per_gpc;
    int max_cta_per_gpc =
        max_sm_per_gpc - (max_sm_per_gpc % static_cast<int>(cluster_size));
    int cta_per_device = min_num_gpc * max_cta_per_gpc;

    int residual_gpc =
        sm_count < max_sm_per_gpc ? 0 : sm_count % max_sm_per_gpc;
    int residual_cta =
        residual_gpc - (residual_gpc % static_cast<int>(cluster_size));
    cta_per_device += residual_cta;
    cta_per_device = sm_count < cta_per_device ? sm_count : cta_per_device;

    return static_cast<uint32_t>(cta_per_device);
  }

  __host__ __device__ void initialize(
      int M, int N, int cta_m, int cta_n, int batch_count = 1,
      uint32_t cluster_m = 1, uint32_t cluster_n = 1, int max_swizzle_size = 1,
      RasterOrderOptions raster_order_option = RasterOrderOptions::Heuristic) {
    uint32_t ctas_m = detail::ceil_div_pos(M, cta_m);
    uint32_t ctas_n = detail::ceil_div_pos(N, cta_n);
    uint32_t batches =
        batch_count <= 0 ? 1u : static_cast<uint32_t>(batch_count);
    initialize_from_tile_counts(ctas_m, ctas_n, batches, cluster_m, cluster_n,
                                max_swizzle_size, raster_order_option);
  }

  __host__ __device__ void initialize_from_tile_counts(
      uint32_t ctas_m, uint32_t ctas_n, uint32_t batches = 1,
      uint32_t cluster_m = 1, uint32_t cluster_n = 1, int max_swizzle_size = 1,
      RasterOrderOptions raster_order_option = RasterOrderOptions::Heuristic) {
    logical_tiles_m = ctas_m;
    logical_tiles_n = ctas_n;
    problem_blocks_l = batches == 0 ? 1u : batches;
    cluster_shape_m = detail::max_u32(cluster_m, 1u);
    cluster_shape_n = detail::max_u32(cluster_n, 1u);

    log_swizzle_size = get_log_swizzle_size(ctas_m, ctas_n, max_swizzle_size);
    uint32_t swizzle = 1u << log_swizzle_size;

    problem_blocks_m = detail::round_up(ctas_m, swizzle * cluster_shape_m);
    problem_blocks_n = detail::round_up(ctas_n, swizzle * cluster_shape_n);
    raster_order = get_rasterization_order(problem_blocks_m, problem_blocks_n,
                                           raster_order_option);

    blocks_per_batch = uint64_t{problem_blocks_m} * uint64_t{problem_blocks_n};
    blocks_per_problem = blocks_per_batch * uint64_t{problem_blocks_l};
  }

  __host__ __device__ GemmTile
  tile_for_linear_idx(uint64_t linear_idx, uint32_t cta_m_in_cluster = 0,
                      uint32_t cta_n_in_cluster = 0) const {
    if (linear_idx >= blocks_per_problem || blocks_per_batch == 0) {
      return {};
    }

    uint64_t work_idx_l = linear_idx / blocks_per_batch;
    uint64_t remainder = linear_idx - work_idx_l * blocks_per_batch;

    uint32_t cluster_major =
        raster_order == RasterOrder::AlongN ? cluster_shape_n : cluster_shape_m;
    uint32_t cluster_minor =
        raster_order == RasterOrder::AlongN ? cluster_shape_m : cluster_shape_n;
    uint32_t cluster_blk_major = raster_order == RasterOrder::AlongN
                                     ? problem_blocks_n / cluster_shape_n
                                     : problem_blocks_m / cluster_shape_m;

    uint64_t blk_per_grid_dim = remainder / cluster_minor;
    uint64_t cluster_id = blk_per_grid_dim / cluster_major;
    uint64_t cluster_major_offset =
        blk_per_grid_dim - cluster_id * cluster_major;
    uint64_t cluster_minor_offset = raster_order == RasterOrder::AlongN
                                        ? cta_m_in_cluster
                                        : cta_n_in_cluster;

    uint64_t swizzle_mask = (uint64_t{1} << log_swizzle_size) - 1u;
    uint64_t offset = cluster_id & swizzle_mask;
    uint64_t extra = cluster_id >> log_swizzle_size;
    uint64_t cluster_idx_major =
        cluster_blk_major == 0 ? 0 : extra % cluster_blk_major;
    uint64_t cluster_idx_minor_div_swizzle =
        cluster_blk_major == 0 ? 0 : extra / cluster_blk_major;
    uint64_t cluster_idx_minor =
        cluster_idx_minor_div_swizzle * (uint64_t{1} << log_swizzle_size) +
        offset;

    uint32_t minor_work_idx = static_cast<uint32_t>(
        cluster_idx_minor * cluster_minor + cluster_minor_offset);
    uint32_t major_work_idx = static_cast<uint32_t>(
        cluster_idx_major * cluster_major + cluster_major_offset);

    uint32_t tile_m =
        raster_order == RasterOrder::AlongN ? minor_work_idx : major_work_idx;
    uint32_t tile_n =
        raster_order == RasterOrder::AlongN ? major_work_idx : minor_work_idx;
    bool in_bounds = tile_m < logical_tiles_m && tile_n < logical_tiles_n &&
                     work_idx_l < problem_blocks_l;

    return {static_cast<int>(tile_m), static_cast<int>(tile_n),
            static_cast<int>(work_idx_l), true, in_bounds};
  }

  __host__ __device__ static dim3 get_grid_shape(
      int M, int N, int cta_m, int cta_n, int sm_count, int batch_count = 1,
      uint32_t cluster_m = 1, uint32_t cluster_n = 1, int max_swizzle_size = 1,
      RasterOrderOptions raster_order_option = RasterOrderOptions::Heuristic,
      int max_active_clusters = 0, bool truncate_by_problem_size = true) {
    PersistentTileSchedulerSm90Params params;
    params.initialize(M, N, cta_m, cta_n, batch_count, cluster_m, cluster_n,
                      max_swizzle_size, raster_order_option);
    return get_grid_shape(params, sm_count, max_active_clusters,
                          truncate_by_problem_size);
  }

  __host__ __device__ static dim3
  get_grid_shape(PersistentTileSchedulerSm90Params const &params, int sm_count,
                 int max_active_clusters = 0,
                 bool truncate_by_problem_size = true) {
    if (params.blocks_per_problem == 0) {
      return dim3{0, 1, 1};
    }

    uint32_t cluster_m = params.cluster_shape_m;
    uint32_t cluster_n = params.cluster_shape_n;
    uint32_t cluster_size = detail::max_u32(cluster_m * cluster_n, 1u);
    int problem_blocks_total = static_cast<int>(params.blocks_per_problem);

    dim3 launch_grid = params.raster_order == RasterOrder::AlongN
                           ? dim3(cluster_m, 1, 1)
                           : dim3(1, cluster_n, 1);

    if (cluster_size == 1) {
      if (params.raster_order == RasterOrder::AlongN) {
        launch_grid.y = truncate_by_problem_size
                            ? detail::min_int(sm_count, problem_blocks_total)
                            : sm_count;
      } else {
        launch_grid.x = truncate_by_problem_size
                            ? detail::min_int(sm_count, problem_blocks_total)
                            : sm_count;
      }
    } else if (max_active_clusters != 0 &&
               max_active_clusters * static_cast<int>(cluster_size) <=
                   sm_count) {
      if (params.raster_order == RasterOrder::AlongN) {
        int active_ctas = max_active_clusters * static_cast<int>(cluster_n);
        int problem_ctas = problem_blocks_total / static_cast<int>(cluster_m);
        launch_grid.y = truncate_by_problem_size
                            ? detail::min_int(active_ctas, problem_ctas)
                            : active_ctas;
      } else {
        int active_ctas = max_active_clusters * static_cast<int>(cluster_m);
        int problem_ctas = problem_blocks_total / static_cast<int>(cluster_n);
        launch_grid.x = truncate_by_problem_size
                            ? detail::min_int(active_ctas, problem_ctas)
                            : active_ctas;
      }
    } else {
      constexpr int kMaxSmPerGpcSm90 = 18;
      uint32_t cta_per_device = get_max_cta_occupancy(
          kMaxSmPerGpcSm90, cluster_m, cluster_n, sm_count);
      if (params.raster_order == RasterOrder::AlongN) {
        int active_ctas = static_cast<int>(cta_per_device / cluster_m);
        int problem_ctas = problem_blocks_total / static_cast<int>(cluster_m);
        launch_grid.y = truncate_by_problem_size
                            ? detail::min_int(active_ctas, problem_ctas)
                            : active_ctas;
      } else {
        int active_ctas = static_cast<int>(cta_per_device / cluster_n);
        int problem_ctas = problem_blocks_total / static_cast<int>(cluster_n);
        launch_grid.x = truncate_by_problem_size
                            ? detail::min_int(active_ctas, problem_ctas)
                            : active_ctas;
      }
    }

    return launch_grid;
  }
};

class GemmPersistentTileScheduler {
public:
  static constexpr uint32_t kNumMmaWarpGroups = 2;

  __device__ explicit GemmPersistentTileScheduler(
      PersistentTileSchedulerSm90Params const &params)
      : params_(params) {
    if (params_.raster_order == RasterOrder::AlongN) {
      current_work_linear_idx_ =
          uint64_t{blockIdx.x} + uint64_t{blockIdx.y} * uint64_t{gridDim.x};
    } else {
      current_work_linear_idx_ =
          uint64_t{blockIdx.x} * uint64_t{gridDim.y} + uint64_t{blockIdx.y};
    }
    total_grid_size_ =
        uint64_t{gridDim.x} * uint64_t{gridDim.y} * uint64_t{gridDim.z};
  }

  __device__ GemmTile current() const {
    dim3 cluster_cta = ::block_id_in_cluster();
    return params_.tile_for_linear_idx(current_work_linear_idx_, cluster_cta.x,
                                       cluster_cta.y);
  }

  __device__ void advance_to_next_work(uint32_t advance_count = 1) {
    current_work_linear_idx_ += total_grid_size_ * uint64_t{advance_count};
  }

  __device__ GemmTile next(uint32_t advance_count = 1) {
    advance_to_next_work(advance_count);
    return current();
  }

  __device__ GemmTile initial_consumer_tile(uint32_t consumer_warp_group_idx) {
    if (consumer_warp_group_idx == 1) {
      advance_to_next_work();
    }
    return current();
  }

  __device__ GemmTile next_consumer_tile() { return next(kNumMmaWarpGroups); }

  __device__ GemmTile next_producer_tile() { return next(); }

  __device__ bool is_last_tile(uint32_t advance_count = 1) const {
    dim3 cluster_cta = ::block_id_in_cluster();
    return !params_
                .tile_for_linear_idx(current_work_linear_idx_ +
                                         total_grid_size_ *
                                             uint64_t{advance_count},
                                     cluster_cta.x, cluster_cta.y)
                .valid;
  }

  __device__ bool is_last_consumer_tile() const {
    return is_last_tile(kNumMmaWarpGroups);
  }

private:
  PersistentTileSchedulerSm90Params params_{};
  uint64_t current_work_linear_idx_ = 0;
  uint64_t total_grid_size_ = 1;
};

inline int query_sm_count(int device = -1) {
  if (device < 0 && cudaGetDevice(&device) != cudaSuccess) {
    return 0;
  }

  int sm_count = 0;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  return sm_count;
}

namespace detail {

using Element = half;

constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 64;
constexpr int kStages = 6;
constexpr int kWarpGroupSize = 128;
constexpr int kNumMmaWarpGroups = 2;
constexpr int kNumWarpGroups = 1 + kNumMmaWarpGroups;
constexpr int kThreads = kNumWarpGroups * kWarpGroupSize;
constexpr int kInstM = 64;
constexpr int kClusterM = 2;
constexpr int kClusterN = 1;
constexpr int kClusterK = 1;
constexpr int kTmaBAtomN = 64;
constexpr int kTmaBAtomK = 8;
constexpr int kTmaBAtomsPerRank = (kBlockK / kTmaBAtomK) / kClusterM;
constexpr uint64_t kTmaCacheHintEvictLast = 0x14F0000000000000ull;
static_assert(kBlockN % kClusterM == 0);
static_assert(kBlockN % kTmaBAtomN == 0);
static_assert(kBlockK % kTmaBAtomK == 0);
static_assert((kBlockK / kTmaBAtomK) % kClusterM == 0);
static_assert(kClusterN == 1);

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
                                             Element const *ptr,
                                             int height,
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
      map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3,
      const_cast<Element *>(ptr), shape, stride, box_shape, box_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return map_cu_result(result);
}

template <int BlockN, int BlockK>
inline cudaError_t make_b_row_major_tensor_map(CUtensorMap *map,
                                               Element const *ptr,
                                               int n,
                                               int k) {
  static_assert(BlockN >= kTmaBAtomN);
  static_assert(BlockN % kTmaBAtomN == 0);
  static_assert(BlockK >= kTmaBAtomK);
  static_assert(BlockK % kTmaBAtomK == 0);
  if (map == nullptr || ptr == nullptr || n <= 0 || k <= 0 ||
      n % kTmaBAtomN != 0) {
    return cudaErrorInvalidValue;
  }

  // Logical tensor is B(n0, k, n64), backed by row-major B[k][n].
  // This matches CuTe's TMA gbasis for GMMA MN-major SW128:
  // box (64 N values, 8 K values, 2 N chunks) == one 128x8 B atom.
  uint64_t shape[] = {static_cast<uint64_t>(kTmaBAtomN),
                      static_cast<uint64_t>(k),
                      static_cast<uint64_t>(n / kTmaBAtomN)};
  uint64_t stride[] = {static_cast<uint64_t>(n) * sizeof(Element),
                       static_cast<uint64_t>(kTmaBAtomN) * sizeof(Element)};
  uint32_t box_shape[] = {static_cast<uint32_t>(kTmaBAtomN),
                          static_cast<uint32_t>(kTmaBAtomK),
                          static_cast<uint32_t>(BlockN / kTmaBAtomN)};
  uint32_t box_stride[] = {1u, 1u, 1u};

  CUresult result = cuTensorMapEncodeTiled(
      map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3,
      const_cast<Element *>(ptr), shape, stride, box_shape, box_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return map_cu_result(result);
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900

__device__ __forceinline__ uint64_t matrix_descriptor_encode(uint64_t x) {
  return (x & 0x3ffff) >> 4;
}

__device__ __forceinline__ uint64_t make_smem_desc_k_major(Element *ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  uint64_t desc = matrix_descriptor_encode(addr);
  desc |= matrix_descriptor_encode(16) << 16;
  desc |= matrix_descriptor_encode(1024) << 32;
  desc |= 1ull << 62;
  return desc;
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major_b(Element *ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  uint64_t desc = matrix_descriptor_encode(addr);
  // GMMA MN-major SW128 layout for a half B tile (N=128, K=64).
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-leading-dimension-byte-offset
  desc |= matrix_descriptor_encode(1024) << 16;
  desc |= matrix_descriptor_encode(2048) << 32;
  desc |= 1ull << 62;
  return desc;
}

__device__ __forceinline__ int b_mn_smem_offset(int n_offset, int k_offset) {
  return (n_offset / kTmaBAtomN) * (kTmaBAtomN * kTmaBAtomK) +
         (k_offset / kTmaBAtomK) * (kBlockN * kTmaBAtomK);
}

template <int RegCount>
__device__ __forceinline__ void warpgroup_reg_alloc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <int RegCount>
__device__ __forceinline__ void warpgroup_reg_dealloc() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit_group() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int PendingGroups>
__device__ __forceinline__ void wgmma_wait_group() {
  static_assert(PendingGroups >= 0 && PendingGroups <= 7);
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(PendingGroups)
               : "memory");
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmma_m64n128k16_f16(float d[8][8],
                                                     Element *sA,
                                                     Element *sB) {
  uint64_t desc_a = make_smem_desc_k_major(sA);
  uint64_t desc_b = make_smem_desc_mn_major_b(sB);
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
      " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
      " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
      " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
      " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
      " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
      " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
      " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
      " %64,"
      " %65,"
      " p,    %67,  %68,  %69,  %70;\n"
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
        "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
        "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
        "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
        "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
        "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
        "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
        "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
        "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
        "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
        "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
        "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7])
      : "l"(desc_a), "l"(desc_b), "r"(int32_t(ScaleD)),
        "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)), "n"(int32_t(TransA)),
        "n"(int32_t(TransB)));
}

__device__ __forceinline__ void init_barrier(uint64_t *bar,
                                             int arrive_count) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar_ptr),
               "r"(arrive_count)
               : "memory");
}

__device__ __forceinline__ void wait_barrier(uint64_t *bar, int phase) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("{\n"
               ".reg .pred p;\n"
               "wait_again:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
               "@p bra.uni wait_done;\n"
               "bra.uni wait_again;\n"
               "wait_done:\n"
               "}\n" ::"r"(bar_ptr),
               "r"(phase));
}

__device__ __forceinline__ void arrive_barrier(uint64_t *bar, int count = 1) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
               ::"r"(bar_ptr), "r"(count)
               : "memory");
}

__device__ __forceinline__ void expect_tma_bytes(uint64_t *bar,
                                                 uint32_t bytes) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
               ::"r"(bar_ptr), "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load(Element *dst,
                                         void const *tensor_map,
                                         uint64_t *bar,
                                         int global_col,
                                         int global_row) {
  uint64_t map_ptr = reinterpret_cast<uint64_t>(tensor_map);
  uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5}], [%2];\n" ::"r"(dst_ptr),
      "l"(map_ptr), "r"(bar_ptr), "n"(0), "r"(global_row),
      "r"(global_col / 64)
      : "memory");
}

__device__ __forceinline__ void tma_load_b_mn_multicast(Element *dst,
                                                        void const *tensor_map,
                                                        uint64_t *bar,
                                                        uint16_t multicast_mask,
                                                        int global_n,
                                                        int global_k) {
  uint64_t map_ptr = reinterpret_cast<uint64_t>(tensor_map);
  uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
      " [%0], [%1, {%4, %5, %6}], [%2], %3, %7;\n" ::"r"(dst_ptr),
      "l"(map_ptr), "r"(bar_ptr), "h"(multicast_mask), "n"(0),
      "r"(global_k), "r"(global_n / kTmaBAtomN),
      "l"(kTmaCacheHintEvictLast)
      : "memory");
}

__device__ __forceinline__ uint16_t multicast_mask_b_cluster_m() {
  uint16_t mask = 0;
  uint32_t cta_n = ::block_id_in_cluster().y;
#pragma unroll
  for (uint32_t m = 0; m < kClusterM; ++m) {
    mask |= uint16_t{1u} << (m * kClusterN + cta_n);
  }
  return mask;
}

__device__ __forceinline__ void tma_store(void const *tensor_map,
                                          Element *src,
                                          int global_row,
                                          int global_col) {
  uint64_t map_ptr = reinterpret_cast<uint64_t>(tensor_map);
  uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(src));
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
               " [%0, {%2, %3, %4}], [%1];\n" ::"l"(map_ptr),
               "r"(src_ptr), "n"(0), "r"(global_row), "r"(global_col / 64)
               : "memory");
}

__device__ __forceinline__ void arrive_barrier_remote(uint64_t *bar,
                                                      uint32_t dst_cta_rank) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("{\n"
               ".reg .b32 remote_bar;\n"
               "mapa.shared::cluster.u32 remote_bar, %0, %1;\n"
               "mbarrier.arrive.shared::cluster.b64 _, [remote_bar];\n"
               "}\n" ::"r"(bar_ptr),
               "r"(dst_cta_rank)
               : "memory");
}

__device__ __forceinline__ void arrive_cluster_empty_barrier(uint64_t *bar) {
  uint32_t rank = ::block_rank_in_cluster();
  arrive_barrier(bar);
  arrive_barrier_remote(bar, rank ^ 1u);
}

__device__ __forceinline__ void tma_commit_group() {
  asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template <int PendingGroups>
__device__ __forceinline__ void tma_wait_group() {
  asm volatile("cp.async.bulk.wait_group.read %0;\n" ::"n"(PendingGroups)
               : "memory");
}

__device__ __forceinline__ void fence_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void warpgroup_sync() {
  asm volatile("bar.sync 1, %0;\n" ::"n"(kWarpGroupSize) : "memory");
}

__device__ __forceinline__ void stmatrix(Element *smem_ptr, Element src[8]) {
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  uint32_t *regs = reinterpret_cast<uint32_t *>(src);
  asm volatile(
      "stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], "
      "{%1, %2, %3, %4};\n" ::"r"(smem),
      "r"(regs[0]), "r"(regs[1]), "r"(regs[2]), "r"(regs[3]));
}

__device__ __forceinline__ int swizzle_128b_half_offset(uint32_t base_addr,
                                                        int half_offset) {
  uint32_t const byte_addr =
      base_addr + static_cast<uint32_t>(half_offset) * sizeof(Element);
  uint32_t const swizzled_byte_addr = byte_addr ^ ((byte_addr & 0x380u) >> 3);
  return static_cast<int>((swizzled_byte_addr - base_addr) / sizeof(Element));
}

__device__ __forceinline__ void store_accumulators_tma(
    SharedStorage &smem, void const *tensor_map_c, GemmTile tile,
    int warp_group_thread_idx, float acc[2][8][8]) {
  // CUTLASS SM90_U32x4_STSM_N destination layout for one 64x128 GMMA
  // accumulator tile.  Each stmatrix.x4 atom writes an 8-column strip; the
  // second half-warp supplies the right 8 columns of each 16-column group.
  int const row = (warp_group_thread_idx & 0xf) +
                  (warp_group_thread_idx >> 5) * 16;
  int const col_lane = ((warp_group_thread_idx >> 4) & 0x1) * 8;
  uint32_t const smem_c_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem.C));

#pragma unroll
  for (int mma_m = 0; mma_m < kBlockM / kInstM; ++mma_m) {
#pragma unroll
    for (int inst_n = 0; inst_n < kBlockN / 16; ++inst_n) {
      alignas(16) Element frag[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        // SM90_U32x4_STSM_N consumes the GMMA C fragment in the original
        // register order. Its SrcLayout maps these eight values across the
        // destination lane addresses, so a software reorder here would break C.
        frag[i] = __float2half_rn(acc[mma_m][inst_n][i]);
      }

      int const col = inst_n * 16 + col_lane;
      int const col_chunk = col / 64;
      int const col_in_chunk = col - col_chunk * 64;
      int const addr = mma_m * kInstM * kBlockN +
                       col_chunk * kInstM * 64 + row * 64 + col_in_chunk;
      stmatrix(&smem.C[swizzle_128b_half_offset(smem_c_addr, addr)], frag);
    }

    fence_async_shared();
    warpgroup_sync();
    if (warp_group_thread_idx == 0) {
      tma_store(tensor_map_c, &smem.C[mma_m * kInstM * kBlockN],
                tile.m * kBlockM + mma_m * kInstM, tile.n * kBlockN);
      tma_commit_group();
    }
  }
  tma_wait_group<0>();
}

__device__ __forceinline__ void stage_next(int &stage, int &phase) {
  ++stage;
  if (stage == kStages) {
    stage = 0;
    phase ^= 1;
  }
}

__device__ __forceinline__ void stage_advance(int &stage, int &phase,
                                              int steps) {
  phase ^= ((stage + steps) / kStages) & 1;
  stage = (stage + steps) % kStages;
}

__device__ __forceinline__ void clear_accumulators(float acc[2][8][8]) {
#pragma unroll
  for (int m = 0; m < 2; ++m) {
#pragma unroll
    for (int n = 0; n < 8; ++n) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        acc[m][n][i] = 0.0f;
      }
    }
  }
}

__device__ __forceinline__ void keep_accumulators_live(float acc[2][8][8]) {
#pragma unroll
  for (int m = 0; m < 2; ++m) {
#pragma unroll
    for (int n = 0; n < 8; ++n) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        asm volatile("" : "+f"(acc[m][n][i]) :: "memory");
      }
    }
  }
}

template <bool ReleaseStages>
__device__ __forceinline__ void consume_k_tile(SharedStorage &smem,
                                               uint64_t *full_barriers,
                                               uint64_t *empty_barriers,
                                               int &stage,
                                               int &phase,
                                               int &previous_stage,
                                               int warp_group_thread_idx,
                                               float acc[2][8][8]) {
  wait_barrier(&full_barriers[stage], phase);
  wgmma_fence();

#pragma unroll
  for (int mma_m = 0; mma_m < kBlockM / kInstM; ++mma_m) {
#pragma unroll
    for (int mma_k = 0; mma_k < kBlockK; mma_k += 16) {
      Element *sA = &smem.A[stage * kBlockM * kBlockK +
                            mma_m * kInstM * kBlockK + mma_k];
      Element *sB =
          &smem.B[stage * kBlockN * kBlockK + mma_k * kBlockN];
      wgmma_m64n128k16_f16<1, 1, 1, 0, 1>(acc[mma_m], sA, sB);
    }
  }

  wgmma_commit_group();
  if constexpr (ReleaseStages) {
    wgmma_wait_group<1>();
    if (warp_group_thread_idx == 0) {
      arrive_cluster_empty_barrier(&empty_barriers[previous_stage]);
    }
  }
  previous_stage = stage;
  stage_next(stage, phase);
}

__device__ __forceinline__ void drain_invalid_tile(uint64_t *full_barriers,
                                                   uint64_t *empty_barriers,
                                                   int k_tiles,
                                                   int &stage,
                                                   int &phase,
                                                   int warp_group_thread_idx) {
  for (int k = 0; k < k_tiles; ++k) {
    int current_stage = stage;
    wait_barrier(&full_barriers[current_stage], phase);
    if (warp_group_thread_idx == 0) {
      arrive_cluster_empty_barrier(&empty_barriers[current_stage]);
    }
    stage_next(stage, phase);
  }
}

#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900

} // namespace detail

__global__ __launch_bounds__(detail::kThreads) void hgemm_pingpong_kernel(
    const __grid_constant__ CUtensorMap tensor_map_a,
    const __grid_constant__ CUtensorMap tensor_map_b,
    const __grid_constant__ CUtensorMap tensor_map_c,
    int M, int N, int K,
    PersistentTileSchedulerSm90Params scheduler_params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  using namespace detail;

  (void)M;
  (void)N;

  extern __shared__ __align__(128) uint8_t dynamic_smem[];
  SharedStorage &smem = *reinterpret_cast<SharedStorage *>(dynamic_smem);

  __shared__ __align__(8) uint64_t full_barriers[kStages];
  __shared__ __align__(8) uint64_t empty_barriers[kStages];
  __shared__ __align__(8) uint64_t math_turn[2];
  __shared__ __align__(8) uint64_t epilogue_turn[2];

  int const thread_idx = static_cast<int>(threadIdx.x);
  int const warp_group_idx = thread_idx / kWarpGroupSize;
  int const warp_group_thread_idx = thread_idx % kWarpGroupSize;

  if (thread_idx == 0) {
    for (int i = 0; i < kStages; ++i) {
      init_barrier(&full_barriers[i], 1);
      init_barrier(&empty_barriers[i], kClusterM * kClusterN);
    }
    init_barrier(&math_turn[0], 1);
    init_barrier(&math_turn[1], 1);
    init_barrier(&epilogue_turn[0], 1);
    init_barrier(&epilogue_turn[1], 1);
  }
  __syncthreads();
  fence_barrier_init();
  ::cluster_sync();

  int const k_tiles = K / kBlockK;
  GemmPersistentTileScheduler scheduler(scheduler_params);

  enum class WarpGroupRole {
    Producer,
    Consumer0,
    Consumer1
  };
  WarpGroupRole role = warp_group_idx == 0
                           ? WarpGroupRole::Producer
                           : (warp_group_idx == 1 ? WarpGroupRole::Consumer0
                                                  : WarpGroupRole::Consumer1);

  if (role == WarpGroupRole::Producer) {
    warpgroup_reg_dealloc<40>();

    if (warp_group_thread_idx == 0) {
      int stage = 0;
      int phase = 0;
      uint32_t const cluster_rank = ::block_rank_in_cluster();
      uint32_t const cluster_m_rank = ::block_id_in_cluster().x;
      uint16_t const b_multicast_mask = multicast_mask_b_cluster_m();
      constexpr uint32_t load_bytes_a =
          sizeof(Element) * kBlockM * kBlockK;
      constexpr uint32_t load_bytes_b =
          sizeof(Element) * kBlockK * kBlockN;

      for (GemmTile tile = scheduler.current(); tile.valid;
           tile = scheduler.next_producer_tile()) {
        for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
          wait_barrier(&empty_barriers[stage], phase);
          bool const has_valid_b_tile =
              tile.valid && tile.n >= 0 &&
              static_cast<uint32_t>(tile.n) <
                  scheduler_params.logical_tiles_n;
          uint32_t expected_bytes = 0;
          if (tile.in_bounds) {
            expected_bytes += load_bytes_a;
          }
          if (has_valid_b_tile &&
              (b_multicast_mask & (uint16_t{1u} << cluster_rank)) != 0) {
            expected_bytes += load_bytes_b;
          }

          if (expected_bytes != 0) {
            expect_tma_bytes(&full_barriers[stage], expected_bytes);
          }
          if (tile.in_bounds) {
            tma_load(&smem.A[stage * kBlockM * kBlockK], &tensor_map_a,
                     &full_barriers[stage], k_tile * kBlockK,
                     tile.m * kBlockM);
          }
          if (has_valid_b_tile && b_multicast_mask != 0) {
            // Each M-rank issues alternating 8-row K atoms. Every atom covers
            // the full 128-wide N tile in canonical MN-major SW128 order.
#pragma unroll
            for (int k_atom_iter = 0; k_atom_iter < kTmaBAtomsPerRank;
                 ++k_atom_iter) {
              int const k_atom =
                  (k_atom_iter * kClusterM +
                   static_cast<int>(cluster_m_rank)) *
                  kTmaBAtomK;
              Element *b_atom =
                  &smem.B[stage * kBlockN * kBlockK +
                          b_mn_smem_offset(0, k_atom)];
              tma_load_b_mn_multicast(
                  b_atom, &tensor_map_b, &full_barriers[stage],
                  b_multicast_mask, tile.n * kBlockN,
                  k_tile * kBlockK + k_atom);
            }
          }
          if (expected_bytes == 0) {
            arrive_barrier(&full_barriers[stage]);
          }
          stage_next(stage, phase);
        }
      }
    }
    return;
  }

  warpgroup_reg_alloc<232>();

  int const consumer_idx = role == WarpGroupRole::Consumer0 ? 0 : 1;
  int stage = 0;
  int phase = 0;
  int turn_phase = 0;

  if (consumer_idx == 0 && warp_group_thread_idx == 0) {
    for (int i = 0; i < kStages; ++i) {
      arrive_cluster_empty_barrier(&empty_barriers[i]);
    }
  }

  if (consumer_idx == 1) {
    if (warp_group_thread_idx == 0) {
      arrive_barrier(&math_turn[0]);
      arrive_barrier(&epilogue_turn[0]);
    }
    stage_advance(stage, phase, k_tiles);
  }

  for (GemmTile tile = scheduler.initial_consumer_tile(consumer_idx);
       tile.valid; tile = scheduler.next_consumer_tile()) {
    wait_barrier(&math_turn[consumer_idx], turn_phase);

    if (!tile.in_bounds) {
      drain_invalid_tile(full_barriers, empty_barriers, k_tiles, stage, phase,
                         warp_group_thread_idx);
      stage_advance(stage, phase, k_tiles);
      if (warp_group_thread_idx == 0) {
        arrive_barrier(&math_turn[1 - consumer_idx]);
      }

      wait_barrier(&epilogue_turn[consumer_idx], turn_phase);
      if (warp_group_thread_idx == 0) {
        arrive_barrier(&epilogue_turn[1 - consumer_idx]);
      }
      turn_phase ^= 1;
      continue;
    }

    float acc[2][8][8];
    clear_accumulators(acc);
    keep_accumulators_live(acc);

    int previous_stage = stage;
    consume_k_tile<false>(smem, full_barriers, empty_barriers, stage, phase,
                          previous_stage, warp_group_thread_idx, acc);
    for (int k_tile = 1; k_tile < k_tiles; ++k_tile) {
      consume_k_tile<true>(smem, full_barriers, empty_barriers, stage, phase,
                           previous_stage, warp_group_thread_idx, acc);
    }
    wgmma_wait_group<0>();
    if (warp_group_thread_idx == 0) {
      arrive_cluster_empty_barrier(&empty_barriers[previous_stage]);
    }

    stage_advance(stage, phase, k_tiles);
    if (warp_group_thread_idx == 0) {
      arrive_barrier(&math_turn[1 - consumer_idx]);
    }

    wait_barrier(&epilogue_turn[consumer_idx], turn_phase);

    store_accumulators_tma(smem, &tensor_map_c, tile, warp_group_thread_idx,
                           acc);
    if (warp_group_thread_idx == 0) {
      arrive_barrier(&epilogue_turn[1 - consumer_idx]);
    }
    turn_phase ^= 1;
  }
#else
  (void)tensor_map_a;
  (void)tensor_map_b;
  (void)tensor_map_c;
  (void)M;
  (void)N;
  (void)K;
  (void)scheduler_params;
#endif
}

inline cudaError_t launch_hgemm_128x128x64_pingpong(
    half const *A, half const *B, half *C, int M, int N, int K,
    cudaStream_t stream = 0, int max_swizzle_size = 1,
    RasterOrderOptions raster_order = RasterOrderOptions::Heuristic) {
  using namespace detail;

  if (A == nullptr || B == nullptr || C == nullptr || M <= 0 || N <= 0 ||
      K <= 0 || M % kBlockM != 0 || N % kBlockN != 0 ||
      K % kBlockK != 0) {
    return cudaErrorInvalidValue;
  }

  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) {
    return err;
  }
  int cluster_launch = 0;
  err = cudaDeviceGetAttribute(&cluster_launch, cudaDevAttrClusterLaunch,
                               device);
  if (err != cudaSuccess) {
    return err;
  }
  if (cluster_launch == 0) {
    return cudaErrorInvalidDeviceFunction;
  }

  PersistentTileSchedulerSm90Params scheduler;
  scheduler.initialize(M, N, kBlockM, kBlockN, 1, kClusterM, kClusterN,
                       max_swizzle_size, raster_order);

  CUtensorMap map_a;
  CUtensorMap map_b;
  CUtensorMap map_c;
  err = make_row_major_tensor_map<kBlockM, kBlockK>(&map_a, A, M, K);
  if (err != cudaSuccess) {
    return err;
  }
  err = make_b_row_major_tensor_map<kBlockN, kBlockK>(&map_b, B, N, K);
  if (err != cudaSuccess) {
    return err;
  }
  err = make_row_major_tensor_map<kInstM, kBlockN>(&map_c, C, M, N);
  if (err != cudaSuccess) {
    return err;
  }

  err = cudaFuncSetAttribute(
      hgemm_pingpong_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(sizeof(SharedStorage)));
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaFuncSetAttribute(hgemm_pingpong_kernel,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             100);
  if (err != cudaSuccess) {
    return err;
  }

  err = cudaFuncSetAttribute(hgemm_pingpong_kernel,
                             cudaFuncAttributeRequiredClusterWidth,
                             kClusterM);
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaFuncSetAttribute(hgemm_pingpong_kernel,
                             cudaFuncAttributeRequiredClusterHeight,
                             kClusterN);
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaFuncSetAttribute(hgemm_pingpong_kernel,
                             cudaFuncAttributeRequiredClusterDepth,
                             kClusterK);
  if (err != cudaSuccess) {
    return err;
  }

  dim3 grid = PersistentTileSchedulerSm90Params::get_grid_shape(
      scheduler, query_sm_count());
  if (grid.x == 0 || grid.y == 0 || grid.z == 0 ||
      grid.x % kClusterM != 0 || grid.y % kClusterN != 0 ||
      grid.z % kClusterK != 0) {
    return cudaErrorInvalidConfiguration;
  }

  cudaLaunchAttribute launch_attr[1] = {};
  launch_attr[0].id = cudaLaunchAttributeClusterDimension;
  launch_attr[0].val.clusterDim.x = kClusterM;
  launch_attr[0].val.clusterDim.y = kClusterN;
  launch_attr[0].val.clusterDim.z = kClusterK;

  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = dim3(kThreads, 1, 1);
  config.dynamicSmemBytes = sizeof(SharedStorage);
  config.stream = stream;
  config.attrs = launch_attr;
  config.numAttrs = 1;

  return cudaLaunchKernelEx(&config, hgemm_pingpong_kernel, map_a, map_b,
                            map_c, M, N, K, scheduler);
}

} // namespace sm90_hgemm_pingpong
