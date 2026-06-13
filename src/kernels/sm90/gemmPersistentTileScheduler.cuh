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

enum class RasterOrder { AlongM, AlongN };

enum class RasterOrderOptions { Heuristic, AlongM, AlongN };

struct GemmTile {
  int m = -1;
  int n = -1;
  bool valid = false;
};

struct PersistentTileSchedulerSm90Params {
  static constexpr uint32_t kLogSwizzleMask = 0x3u;
  static constexpr uint32_t kRasterAlongNBit = 0x4u;

  uint64_t blocks_per_problem = 0;

  uint32_t cluster_shape_major = 1;
  uint32_t cluster_shape_minor = 1;
  uint32_t cluster_blocks_major = 0;
  uint32_t scheduler_bits = kRasterAlongNBit;

  __host__ __device__ void set_scheduler_bits(int32_t log_swizzle_size,
                                              RasterOrder raster_order) {
    scheduler_bits =
        (static_cast<uint32_t>(log_swizzle_size) & kLogSwizzleMask) |
        (raster_order == RasterOrder::AlongN ? kRasterAlongNBit : 0u);
  }

  __host__ __device__ uint32_t log_swizzle_size() const {
    return scheduler_bits & kLogSwizzleMask;
  }

  __host__ __device__ bool raster_order_along_n() const {
    return (scheduler_bits & kRasterAlongNBit) != 0;
  }

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

  __host__ __device__ static int32_t
  fit_log_swizzle_size(uint32_t minor_clusters, int32_t log_swizzle) {
    while (log_swizzle > 0) {
      uint32_t swizzle = 1u << log_swizzle;
      if (minor_clusters % swizzle == 0) {
        break;
      }
      --log_swizzle;
    }
    return log_swizzle;
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
      int M, int N, int cta_m, int cta_n, uint32_t cluster_m = 1,
      uint32_t cluster_n = 1, int max_swizzle_size = 1,
      RasterOrderOptions raster_order_option = RasterOrderOptions::Heuristic) {
    // Precondition: M and N are exact multiples of the CTA shape.
    uint32_t ctas_m =
        (M > 0 && cta_m > 0) ? static_cast<uint32_t>(M / cta_m) : 0u;
    uint32_t ctas_n =
        (N > 0 && cta_n > 0) ? static_cast<uint32_t>(N / cta_n) : 0u;
    initialize_from_tile_counts(ctas_m, ctas_n, cluster_m, cluster_n,
                                max_swizzle_size, raster_order_option);
  }

  __host__ __device__ void initialize_from_tile_counts(
      uint32_t ctas_m, uint32_t ctas_n, uint32_t cluster_m = 1,
      uint32_t cluster_n = 1, int max_swizzle_size = 1,
      RasterOrderOptions raster_order_option = RasterOrderOptions::Heuristic) {
    uint32_t problem_blocks_m = ctas_m;
    uint32_t problem_blocks_n = ctas_n;
    uint32_t cluster_shape_m = detail::max_u32(cluster_m, 1u);
    uint32_t cluster_shape_n = detail::max_u32(cluster_n, 1u);

    RasterOrder raster_order = get_rasterization_order(
        problem_blocks_m, problem_blocks_n, raster_order_option);
    cluster_shape_major =
        raster_order == RasterOrder::AlongN ? cluster_shape_n : cluster_shape_m;
    cluster_shape_minor =
        raster_order == RasterOrder::AlongN ? cluster_shape_m : cluster_shape_n;

    bool exact_cluster_tiling = problem_blocks_m % cluster_shape_m == 0 &&
                                problem_blocks_n % cluster_shape_n == 0;
    if (!exact_cluster_tiling || problem_blocks_m == 0 ||
        problem_blocks_n == 0) {
      blocks_per_problem = 0;
      set_scheduler_bits(0, raster_order);
      cluster_blocks_major = 0;
      return;
    }

    uint32_t minor_clusters = raster_order == RasterOrder::AlongN
                                  ? problem_blocks_m / cluster_shape_m
                                  : problem_blocks_n / cluster_shape_n;

    // Avoid CUTLASS-style padded swizzle tiles in this exact-tiling kernel.
    int32_t log_swizzle_size = fit_log_swizzle_size(
        minor_clusters, get_log_swizzle_size(problem_blocks_m, problem_blocks_n,
                                             max_swizzle_size));
    set_scheduler_bits(log_swizzle_size, raster_order);
    cluster_blocks_major = raster_order == RasterOrder::AlongN
                               ? problem_blocks_n / cluster_shape_n
                               : problem_blocks_m / cluster_shape_m;

    blocks_per_problem =
        uint64_t{problem_blocks_m} * uint64_t{problem_blocks_n};
  }

  __host__ __device__ GemmTile tile_for_linear_idx(uint64_t linear_idx) const {
    if (linear_idx >= blocks_per_problem) {
      return {};
    }

    // The launch grid keeps the cluster-minor CTA offset in the low digit.
    uint64_t cluster_minor_offset = linear_idx % cluster_shape_minor;
    uint64_t linear_idx_without_minor_offset = linear_idx / cluster_shape_minor;
    uint64_t cluster_major_offset =
        linear_idx_without_minor_offset % cluster_shape_major;
    uint64_t cluster_id = linear_idx_without_minor_offset / cluster_shape_major;

    uint32_t log_swizzle = log_swizzle_size();
    uint64_t swizzle_size = uint64_t{1} << log_swizzle;
    uint64_t offset = cluster_id & (swizzle_size - 1u);
    uint64_t extra = cluster_id >> log_swizzle;
    uint64_t cluster_idx_major = extra % cluster_blocks_major;
    uint64_t cluster_idx_minor_div_swizzle = extra / cluster_blocks_major;
    uint64_t cluster_idx_minor =
        cluster_idx_minor_div_swizzle * swizzle_size + offset;

    uint32_t minor_work_idx = static_cast<uint32_t>(
        cluster_idx_minor * cluster_shape_minor + cluster_minor_offset);
    uint32_t major_work_idx = static_cast<uint32_t>(
        cluster_idx_major * cluster_shape_major + cluster_major_offset);

    bool along_n = raster_order_along_n();
    uint32_t tile_m = along_n ? minor_work_idx : major_work_idx;
    uint32_t tile_n = along_n ? major_work_idx : minor_work_idx;

    return {static_cast<int>(tile_m), static_cast<int>(tile_n), true};
  }

  __host__ __device__ static dim3 get_grid_shape(
      int M, int N, int cta_m, int cta_n, int sm_count, uint32_t cluster_m = 1,
      uint32_t cluster_n = 1, int max_swizzle_size = 1,
      RasterOrderOptions raster_order_option = RasterOrderOptions::Heuristic,
      int max_active_clusters = 0, bool truncate_by_problem_size = true) {
    PersistentTileSchedulerSm90Params params;
    params.initialize(M, N, cta_m, cta_n, cluster_m, cluster_n,
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

    bool along_n = params.raster_order_along_n();
    uint32_t cluster_m =
        along_n ? params.cluster_shape_minor : params.cluster_shape_major;
    uint32_t cluster_n =
        along_n ? params.cluster_shape_major : params.cluster_shape_minor;
    uint32_t cluster_size = detail::max_u32(cluster_m * cluster_n, 1u);
    int problem_blocks_total = static_cast<int>(params.blocks_per_problem);

    dim3 launch_grid = along_n ? dim3(cluster_m, 1, 1) : dim3(1, cluster_n, 1);

    if (cluster_size == 1) {
      if (along_n) {
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
      if (along_n) {
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
      if (along_n) {
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

static_assert(sizeof(PersistentTileSchedulerSm90Params) == 24,
              "Keep SM90 persistent scheduler params compact.");

class GemmPersistentTileScheduler {
public:
  __device__ explicit GemmPersistentTileScheduler(
      PersistentTileSchedulerSm90Params const &params)
      : params_(params) {
    bool along_n = params_.raster_order_along_n();
    uint64_t block_minor =
        along_n ? uint64_t{blockIdx.x} : uint64_t{blockIdx.y};
    uint64_t block_major =
        along_n ? uint64_t{blockIdx.y} : uint64_t{blockIdx.x};
    uint64_t grid_minor = along_n ? uint64_t{gridDim.x} : uint64_t{gridDim.y};
    current_work_linear_idx_ = block_minor + block_major * grid_minor;
    total_grid_size_ =
        uint64_t{gridDim.x} * uint64_t{gridDim.y} * uint64_t{gridDim.z};
  }

  __device__ GemmTile current() const {
    return params_.tile_for_linear_idx(current_work_linear_idx_);
  }

  __device__ GemmTile initial_consumer_tile(uint32_t consumer_warp_group_idx) {
    (void)consumer_warp_group_idx;
    return current();
  }

  __device__ void advance_to_next_work(uint32_t advance_count = 1) {
    current_work_linear_idx_ += total_grid_size_ * uint64_t{advance_count};
  }

  __device__ GemmTile next(uint32_t advance_count = 1) {
    advance_to_next_work(advance_count);
    return current();
  }

  __device__ GemmTile next_producer_tile() { return next(); }

  __device__ GemmTile next_consumer_tile() { return next(); }

  __device__ bool is_last_tile(uint32_t advance_count = 1) const {
    return !params_
                .tile_for_linear_idx(current_work_linear_idx_ +
                                     total_grid_size_ * uint64_t{advance_count})
                .valid;
  }

  __device__ bool is_last_consumer_tile() const { return is_last_tile(); }

private:
  PersistentTileSchedulerSm90Params params_{};
  uint64_t current_work_linear_idx_ = 0;
  uint64_t total_grid_size_ = 1;
};