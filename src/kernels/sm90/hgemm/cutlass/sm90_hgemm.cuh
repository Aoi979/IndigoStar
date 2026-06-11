#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"

namespace cutlass_sm90_hgemm {

namespace detail {

class WorkspaceCache {
public:
  WorkspaceCache() = default;
  WorkspaceCache(const WorkspaceCache &) = delete;
  WorkspaceCache &operator=(const WorkspaceCache &) = delete;

  ~WorkspaceCache() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }

  cutlass::Status ensure(std::size_t bytes) {
    if (bytes <= bytes_) {
      return cutlass::Status::kSuccess;
    }
    if (ptr_ != nullptr) {
      cudaError_t err = cudaFree(ptr_);
      ptr_ = nullptr;
      bytes_ = 0;
      if (err != cudaSuccess) {
        return cutlass::Status::kErrorInternal;
      }
    }
    if (bytes == 0) {
      return cutlass::Status::kSuccess;
    }
    cudaError_t err = cudaMalloc(&ptr_, bytes);
    if (err != cudaSuccess) {
      return cutlass::Status::kErrorMemoryAllocation;
    }
    bytes_ = bytes;
    return cutlass::Status::kSuccess;
  }

  void *get() const { return ptr_; }

private:
  void *ptr_ = nullptr;
  std::size_t bytes_ = 0;
};

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

template <class MainloopSchedule, class EpilogueSchedule>
struct HgemmSm90Config {
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = void;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;

  using LayoutA = cutlass::layout::RowMajor;
  // CUTLASS 3.x models B as shape (N,K). ColumnMajor here maps to this
  // project's physical row-major B[K][N] storage.
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  static constexpr int kAlignmentA = 16 / sizeof(ElementA);
  static constexpr int kAlignmentB = 16 / sizeof(ElementB);
  static constexpr int kAlignmentC = 1;
  static constexpr int kAlignmentD = 16 / sizeof(ElementD);

  using TileShape = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using MainloopClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;
  using EpilogueClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape,
          EpilogueClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementCompute, ElementC, LayoutC, kAlignmentC,
          ElementD, LayoutD, kAlignmentD, EpilogueSchedule,
          cutlass::epilogue::fusion::LinearCombination<
              ElementD, ElementCompute, ElementC, ElementCompute>>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, ElementA,
          LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
          ElementAccumulator, TileShape, MainloopClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<
              static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
          MainloopSchedule>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>,
                                           CollectiveMainloop,
                                           CollectiveEpilogue,
                                           cutlass::gemm::PersistentScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename Gemm>
cutlass::Status launch_gemm(typename Gemm::ElementA const *A,
                            typename Gemm::ElementB const *B,
                            typename Gemm::ElementD *D,
                            int M, int N, int K,
                            cudaStream_t stream) {
  if (A == nullptr || B == nullptr || D == nullptr ||
      M <= 0 || N <= 0 || K <= 0) {
    return cutlass::Status::kErrorInvalidProblem;
  }

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using ElementC = typename Gemm::ElementC;
  using ElementD = typename Gemm::ElementD;

  auto problem_shape = cute::make_shape(M, N, K, 1);
  StrideA stride_A =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B =
      cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
  StrideC stride_C =
      cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D =
      cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) {
    return cutlass::Status::kErrorInternal;
  }

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device);

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_shape,
      {A, stride_A, B, stride_B},
      {{ElementD(1.0f), ElementD(0.0f)},
       static_cast<ElementC const *>(nullptr), stride_C, D, stride_D},
      hw_info};

  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = 0.0f;

  Gemm gemm;
  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  std::size_t workspace_size = Gemm::get_workspace_size(arguments);
  static WorkspaceCache workspace;
  status = workspace.ensure(workspace_size);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  status = gemm.initialize(arguments, workspace.get(), stream);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }
  return gemm.run(stream);
}

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

} // namespace detail

inline cutlass::Status launch_hgemm_pingpong(
    cutlass::half_t const *A, cutlass::half_t const *B, cutlass::half_t *C,
    int M, int N, int K, cudaStream_t stream = 0) {
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  using Gemm = typename detail::HgemmSm90Config<
      cutlass::gemm::KernelTmaWarpSpecializedPingpong,
      cutlass::epilogue::TmaWarpSpecialized>::Gemm;
  return detail::launch_gemm<Gemm>(A, B, C, M, N, K, stream);
#else
  (void)A;
  (void)B;
  (void)C;
  (void)M;
  (void)N;
  (void)K;
  (void)stream;
  return cutlass::Status::kErrorNotSupported;
#endif
}

inline cutlass::Status launch_hgemm_cooperative(
    cutlass::half_t const *A, cutlass::half_t const *B, cutlass::half_t *C,
    int M, int N, int K, cudaStream_t stream = 0) {
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  using Gemm = typename detail::HgemmSm90Config<
      cutlass::gemm::KernelTmaWarpSpecializedCooperative,
      cutlass::epilogue::TmaWarpSpecializedCooperative>::Gemm;
  return detail::launch_gemm<Gemm>(A, B, C, M, N, K, stream);
#else
  (void)A;
  (void)B;
  (void)C;
  (void)M;
  (void)N;
  (void)K;
  (void)stream;
  return cutlass::Status::kErrorNotSupported;
#endif
}

} // namespace cutlass_sm90_hgemm
