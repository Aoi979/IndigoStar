#pragma once
#include "../../cluster.cuh"
#include "../../gemmPersistentTileScheduler.cuh"
#include "../../mbarrier.cuh"
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

constexpr int kWarpSize = 32;
constexpr int kWarpGroupSize = 4 * kWarpSize;
using Element = half;

__device__ __forceinline__ uint64_t matrix_descriptor_encode(uint64_t x) {
  return (x & 0x3ffff) >> 4;
}

// TODO: Fix descriptor
__device__ __forceinline__ uint64_t make_smem_desc_k_major(Element *ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  uint64_t desc = matrix_descriptor_encode(addr);
  desc |= matrix_descriptor_encode(16) << 16;
  desc |= matrix_descriptor_encode(1024) << 32;
  desc |= 1ull << 62;
  return desc;
}

// TODO: Fix descriptor
__device__ __forceinline__ uint64_t make_smem_desc_mn_major_b(Element *ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  uint64_t desc = matrix_descriptor_encode(addr);
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-leading-dimension-byte-offset
  desc |= matrix_descriptor_encode(1024) << 16;
  desc |= matrix_descriptor_encode(2048) << 32;
  desc |= 1ull << 62;
  return desc;
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
__device__ __forceinline__ void wgmma256(float d[16][8], half *sA, half *sB) {
  uint64_t desc_a = make_smem_desc_k_major(sA);
  uint64_t desc_b = make_smem_desc_mn_major_b(sB);
  asm volatile("{\n"
               ".reg .pred p;\n"
               "setp.ne.b32 p, %130, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 "
               "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
               " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
               " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
               " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
               " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
               " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
               " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
               " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
               " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
               " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
               " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
               " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
               " %96,  %97,  %98,  %99,  %100, %101, %102, %103,  "
               " %104, %105, %106, %107, %108, %109, %110, %111,  "
               " %112, %113, %114, %115, %116, %117, %118, %119,  "
               " %120, %121, %122, %123, %124, %125, %126, %127},"
               " %128,"
               " %129,"
               " p,    %131,  %132,  %133,  %134;\n"
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
                 "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7]),
                 "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
                 "+f"(d[8][4]), "+f"(d[8][5]), "+f"(d[8][6]), "+f"(d[8][7]),
                 "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]),
                 "+f"(d[9][4]), "+f"(d[9][5]), "+f"(d[9][6]), "+f"(d[9][7]),
                 "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]),
                 "+f"(d[10][4]), "+f"(d[10][5]), "+f"(d[10][6]), "+f"(d[10][7]),
                 "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]),
                 "+f"(d[11][4]), "+f"(d[11][5]), "+f"(d[11][6]), "+f"(d[11][7]),
                 "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]),
                 "+f"(d[12][4]), "+f"(d[12][5]), "+f"(d[12][6]), "+f"(d[12][7]),
                 "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]),
                 "+f"(d[13][4]), "+f"(d[13][5]), "+f"(d[13][6]), "+f"(d[13][7]),
                 "+f"(d[14][0]), "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3]),
                 "+f"(d[14][4]), "+f"(d[14][5]), "+f"(d[14][6]), "+f"(d[14][7]),
                 "+f"(d[15][0]), "+f"(d[15][1]), "+f"(d[15][2]), "+f"(d[15][3]),
                 "+f"(d[15][4]), "+f"(d[15][5]), "+f"(d[15][6]), "+f"(d[15][7])
               : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)),
                 "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)),
                 "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template <uint32_t RegCount> __device__ void warpgroup_reg_dealloc() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <uint32_t RegCount> __device__ void warpgroup_reg_alloc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

__device__ __forceinline__ void arrive_cluster_empty_barrier(uint64_t *bar,
                                                             uint32_t rank_id) {
  arrive_barrier_remote(bar, rank_id);
}

template <int CM, int CN>
__device__ __forceinline__ int cluster_rank_mn(int cm, int cn) {
  return cm + CM * cn;
}

template <int CM, int CN>
__device__ __forceinline__ uint16_t A_mcast_mask(int cm) {
  uint16_t mask = 0;
#pragma unroll
  for (int cn = 0; cn < CN; ++cn) {
    mask |= uint16_t(1u << cluster_rank_mn<CM, CN>(cm, cn));
  }
  return mask;
}

template <int CM, int CN>
__device__ __forceinline__ uint16_t B_mcast_mask(int cn) {
  uint16_t mask = 0;
#pragma unroll
  for (int cm = 0; cm < CM; ++cm) {
    mask |= uint16_t(1u << cluster_rank_mn<CM, CN>(cm, cn));
  }
  return mask;
}

template <int kStages> struct PipelineState {
  int phase = 0;
  int stage_id = 0;
  __device__ void advance() {
    stage_id++;
    if (stage_id == kStages) {
      phase ^= 1;
      stage_id = 0;
    }
  }
};

__device__ __forceinline__ void tma_load(half *dst, void const *tensor_map,
                                         uint64_t *bar, int global_col,
                                         int global_row) {
  uint64_t map_ptr = reinterpret_cast<uint64_t>(tensor_map);
  uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::"
               "complete_tx::bytes"
               " [%0], [%1, {%3, %4, %5}], [%2];\n" ::"r"(dst_ptr),
               "l"(map_ptr), "r"(bar_ptr), "n"(0), "r"(global_row),
               "r"(global_col / 64)
               : "memory");
}

__global__ void
hgemm_cooperative_kernel(int M, int N, int K,
                         const __grid_constant__ CUtensorMap tensorMapA,
                         const __grid_constant__ CUtensorMap tensorMapB,
                         const __grid_constant__ CUtensorMap tensorMapC,
                         PersistentTileSchedulerSm90Params scheduler_params) {
  constexpr int kStages = 5;
  constexpr int kClusterN = 2;
  constexpr int kClusterM = 1;
  constexpr int kConsumers = 2;
  constexpr int kCtaK = 64;
  constexpr int kCtaM = 128;
  constexpr int kCtaN = 256;

  using Element = half;

  constexpr int kGmemASliceSize = sizeof(Element) * kCtaK * kCtaM;
  constexpr int kGmemBSliceSize = sizeof(Element) & kCtaK * kCtaN;
  constexpr int kExpected_bytes = kGmemASliceSize + kGmemBSliceSize;

  int warp_group_id = threadIdx.x / kWarpGroupSize;
  int const warp_group_thread_idx = threadIdx.x % kWarpGroupSize;
  int warp_in_wg = warp_group_id % 4;
  enum class WarpGroupRole { Producer, Consumer0, Consumer1 };
  WarpGroupRole role = warp_group_id == 0
                           ? WarpGroupRole::Producer
                           : (warp_group_id == 1 ? WarpGroupRole::Consumer0
                                                 : WarpGroupRole::Consumer1);
  int lane_id = threadIdx.x % kWarpSize;
  dim3 cluster_ctaid = block_id_in_cluster();
  int cm = cluster_ctaid.x;
  int cn = cluster_ctaid.y;

  extern __shared__ char shared_memory[];

  int const K_TILE_MAX = K / kCtaK;
  GemmPersistentTileScheduler scheduler(scheduler_params);
  __shared__ uint64_t full[kStages];
  __shared__ uint64_t empty[kStages];

  if (threadIdx.x == 0) {
    for (int i = 0; i < kStages; i++) {
      init_barrier(&full[i], 1);
      init_barrier(&empty[i], kConsumers * (kClusterM + kClusterN - 1));
    }
  }
  fence_barrier_init();
  cluster_sync();

  if (role == WarpGroupRole::Producer) {
    warpgroup_reg_dealloc<40>();
    if (warp_in_wg == 0) {
      if (lane_id == 0) {
        PipelineState<kStages> pipeline_stage;
        auto cluster_m_rank = block_id_in_cluster().x;
        auto a_multicast_mask =
            A_mcast_mask<kClusterM, kClusterN>(cluster_m_rank);
        for (GemmTile tile = scheduler.current(); tile.valid;
             tile = scheduler.next_producer_tile()) {
          for (int k_tile = 0; k_tile < K_TILE_MAX; k_tile++) {
            wait_barrier(&empty[pipeline_stage.stage_id], pipeline_stage.phase);
            expect_tma_bytes(&full[pipeline_stage.stage_id], kExpected_bytes);
            pipeline_stage.advance();
          }
        }
      }
    }
  } else {
    const int consumer_id = role == WarpGroupRole::Consumer0 ? 0 : 1;
    warpgroup_reg_alloc<240>();
    float accumulator[1][kCtaN / 16][8];
    PipelineState<kStages> pipeline_stage;
    for (int i = 0; i < kStages; i++) {
      if (warp_group_thread_idx < (kClusterM + kClusterN - 1)) {
        arrive_cluster_empty_barrier(&empty[i], warp_group_thread_idx);
      }
    }
    for (GemmTile tile = scheduler.initial_consumer_tile(consumer_id);
         tile.valid; tile = scheduler.next_consumer_tile()) {
      {
        wait_barrier(&full[pipeline_stage.stage_id], pipeline_stage.phase);
      }
    }
  }
}

inline cudaError_t map_cu_result(CUresult result) {
  return result == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

template <int BlockMajor, int BlockMinor, typename Element = half>
inline cudaError_t make_tensor_map(CUtensorMap *map, Element const *ptr,
                                   int height, int width) {
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

template <int BlockMajor, int BlockMinor, typename Element = half>
inline cudaError_t make_a_tensor_map(CUtensorMap *map, Element const *ptr,
                                     int height, int width) {
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
  uint32_t box_shape[] = {64u, static_cast<uint32_t>((BlockMajor / 2)),
                          static_cast<uint32_t>(BlockMinor / 64)};
  uint32_t box_stride[] = {1u, 1u, 1u};

  CUresult result = cuTensorMapEncodeTiled(
      map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, const_cast<Element *>(ptr),
      shape, stride, box_shape, box_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return map_cu_result(result);
}

namespace sm90_hgemm_cooperative {

inline cudaError_t
launch_hgemm_128x256x64_cooperative(half const *A, half const *B, half *C,
                                    int M, int N, int K,
                                    cudaStream_t stream = 0) {
  // TODO: implement handwritten SM90 cooperative kernel here.
  // Currently returns not-supported so the build stays clean.
  (void)A;
  (void)B;
  (void)C;
  (void)M;
  (void)N;
  (void)K;
  (void)stream;
  return cudaErrorNotSupported;
}

} // namespace sm90_hgemm_cooperative
