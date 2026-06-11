#include "benchmark.h"
#include "device_utils.h"

#include "kernels/reference/sgemm_naive.cuh"

#if ENABLE_SM80_KERNELS
#include "kernels/sm80/sgemm/handwritten/custom_128x128x32.cuh"
#include "kernels/sm80/sgemm/handwritten/double_buffer_dev_128x128x32.cuh"
#include "kernels/sm80/sgemm/handwritten/external_128x128x16.cuh"
#include "kernels/sm80/sgemm/handwritten/cutlass_like_stage5.cuh"
#include "kernels/sm80/sgemm/cutlass/ref_stage5.cuh"
#include "kernels/sm80/hgemm/handwritten/cute_128x128_nn.cuh"
#include "kernels/sm80/hgemm/handwritten/cute_128x128_nn_no_reg_prefetch.cuh"
#include "kernels/sm80/hgemm/handwritten/cute_ampere_16816.cuh"
#include "kernels/sm80/hgemm/handwritten/cute_ampere_16816_no_reg_prefetch.cuh"
#include "kernels/sm80/hgemm/handwritten/sm80_hgemm.cuh"
#include "kernels/sm80/hgemm/cutlass/tensorop.cuh"
#endif

#if ENABLE_SM90_KERNELS
#include "kernels/sm90/hgemm/handwritten/pingpong.cuh"
#include "kernels/sm90/hgemm/handwritten/cooperative.cuh"
#include "kernels/sm90/hgemm/cutlass/sm90_hgemm.cuh"
#endif

#include <cuda_fp16.h>

// ---------------------------------------------------------------------------
// SGEMM kernel launch wrappers
// ---------------------------------------------------------------------------

#if ENABLE_SM80_KERNELS
bool launch_custom(const Options &options, float *A, float *B, float *C) {
  dim3 block(256);
  dim3 grid(options.n / 128, options.m / 128);
  sgemm_128x128x32<<<grid, block>>>(options.m, options.n, options.k, A, B, C);
  return CUDA_CHECK(cudaGetLastError());
}

bool launch_cutlass_like_stage5(const Options &options, float *A, float *B, float *C) {
  if (options.m % cutlass_like::kCtaM != 0 ||
      options.n % cutlass_like::kCtaN != 0 ||
      options.k % cutlass_like::kCtaK != 0 ||
      options.k < cutlass_like::kCtaK * (cutlass_like::kStages - 1)) {
    std::cerr << "cutlass-stage5 requires M and N to be multiples of 128, "
                 "and K to be a multiple of 8 with K >= 32.\n";
    return false;
  }

  cutlass_like::launch_sgemm_128x128x8stage5(A, B, C, options.m, options.n,
                                             options.k);
  return CUDA_CHECK(cudaGetLastError());
}

bool launch_cutlass_like_stage5_one_cta(const Options &options, float *A,
                                        float *B, float *C) {
  if (options.m % cutlass_like::kCtaM != 0 ||
      options.n % cutlass_like::kCtaN != 0 ||
      options.k % cutlass_like::kCtaK != 0 ||
      options.k < cutlass_like::kCtaK * (cutlass_like::kStages - 1)) {
    std::cerr << "cutlass-stage5-1cta requires M and N to be multiples "
                 "of 128, and K to be a multiple of 8 with K >= 32.\n";
    return false;
  }
  if (!CUDA_CHECK(cudaFuncSetAttribute(
          cutlass_like::sgemm_128x128x8stage5_kernel<false, false, false>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          cutlass_like::kOneCtaPerSmSmemBytes))) {
    return false;
  }

  cutlass_like::launch_sgemm_128x128x8stage5_one_cta_per_sm(
      A, B, C, options.m, options.n, options.k);
  return CUDA_CHECK(cudaGetLastError());
}

bool launch_cutlass_like_stage5_warp_order(const Options &options, float *A,
                                           float *B, float *C) {
  if (options.m % cutlass_like::kCtaM != 0 ||
      options.n % cutlass_like::kCtaN != 0 ||
      options.k % cutlass_like::kCtaK != 0 ||
      options.k < cutlass_like::kCtaK * (cutlass_like::kStages - 1)) {
    std::cerr << "cutlass-stage5-warporder requires M and N to be multiples "
                 "of 128, and K to be a multiple of 8 with K >= 32.\n";
    return false;
  }

  cutlass_like::launch_sgemm_128x128x8stage5_cutlass_warp_order(
      A, B, C, options.m, options.n, options.k);
  return CUDA_CHECK(cudaGetLastError());
}

bool launch_cutlass_like_stage5_schedule(const Options &options, float *A,
                                         float *B, float *C) {
  if (options.m % cutlass_like::kCtaM != 0 ||
      options.n % cutlass_like::kCtaN != 0 ||
      options.k % cutlass_like::kCtaK != 0 ||
      options.k < cutlass_like::kCtaK * (cutlass_like::kStages - 1)) {
    std::cerr << "cutlass-stage5-schedule requires M and N to be multiples "
                 "of 128, and K to be a multiple of 8 with K >= 32.\n";
    return false;
  }

  cutlass_like::launch_sgemm_128x128x8stage5_cutlass_schedule(
      A, B, C, options.m, options.n, options.k);
  return CUDA_CHECK(cudaGetLastError());
}

bool launch_cutlass_like_stage5_copy_schedule(const Options &options, float *A,
                                              float *B, float *C) {
  if (options.m % cutlass_like::kCtaM != 0 ||
      options.n % cutlass_like::kCtaN != 0 ||
      options.k % cutlass_like::kCtaK != 0 ||
      options.k < cutlass_like::kCtaK * (cutlass_like::kStages - 1)) {
    std::cerr << "cutlass-stage5-copyorder requires M and N to be multiples "
                 "of 128, and K to be a multiple of 8 with K >= 32.\n";
    return false;
  }

  cutlass_like::launch_sgemm_128x128x8stage5_cutlass_copy_schedule(
      A, B, C, options.m, options.n, options.k);
  return CUDA_CHECK(cudaGetLastError());
}

bool launch_cutlass_like_stage5_mma_order(const Options &options, float *A,
                                          float *B, float *C) {
  if (options.m % cutlass_like::kCtaM != 0 ||
      options.n % cutlass_like::kCtaN != 0 ||
      options.k % cutlass_like::kCtaK != 0 ||
      options.k < cutlass_like::kCtaK * (cutlass_like::kStages - 1)) {
    std::cerr << "cutlass-stage5-mmaorder requires M and N to be multiples "
                 "of 128, and K to be a multiple of 8 with K >= 32.\n";
    return false;
  }

  cutlass_like::launch_sgemm_128x128x8stage5_cutlass_sm80_mma_order(
      A, B, C, options.m, options.n, options.k);
  return CUDA_CHECK(cudaGetLastError());
}

bool launch_cutlass_ref_stage5(const Options &options, float *A, float *B,
                               float *C) {
  cutlass::Status status = cutlass_ref::launch_sgemm_128x128x8stage5(
      A, B, C, options.m, options.n, options.k);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "cutlass-ref failed: "
              << cutlass::cutlassGetStatusString(status) << '\n';
    return false;
  }
  return CUDA_CHECK(cudaGetLastError());
}

#endif

bool launch_naive(const Options &options, float *A, float *B, float *C) {
  dim3 block(16, 16);
  dim3 grid((options.n + block.x - 1) / block.x,
            (options.m + block.y - 1) / block.y);
  sgemm_naive<<<grid, block>>>(A, B, C, options.m, options.n, options.k);
  return CUDA_CHECK(cudaGetLastError());
}

bool launch_cublas(const Options &options, float *A, float *B, float *C) {
  static CuBlasHandle handle;
  if (!handle.ok()) {
    std::cerr << "cublasCreate failed\n";
    return false;
  }
  float alpha = 1.0F;
  float beta  = 0.0F;
  cublasStatus_t status = cublasSgemm(
      handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
      options.n, options.m, options.k,
      &alpha, B, options.n,
      A, options.k,
      &beta, C, options.n);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasSgemm failed\n";
    return false;
  }
  return CUDA_CHECK(cudaGetLastError());
}

#if ENABLE_SM80_KERNELS
bool launch_external_double_buffer(const Options &options, float *A, float *B, float *C) {
  dim3 block(256);
  dim3 grid(options.n / 128, options.m / 128);
  external::double_buffer::sgemm<<<grid, block>>>(
      options.m, options.n, options.k, A, B, C);
  return CUDA_CHECK(cudaGetLastError());
}

bool launch_external_no_double_buffer(const Options &options, float *A, float *B, float *C) {
  dim3 block(256);
  dim3 grid(options.n / 128, options.m / 128);
  external::no_double_buffer::sgemm<<<grid, block>>>(
      options.m, options.n, options.k, A, B, C);
  return CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// HGEMM kernel launch wrappers
// ---------------------------------------------------------------------------

bool launch_cute_hgemm(const Options &options, cute::half_t *A, cute::half_t *B,
                       cute::half_t *C) {
  if (options.m % 128 != 0 || options.n % 128 != 0 || options.k % 64 != 0) {
    std::cerr << "cute-hgemm requires M and N to be multiples of 128, "
                 "and K to be a multiple of 64.\n";
    return false;
  }
  cudaError_t err = cute_hgemm::launch_hgemm_128x128_nn(
      A, B, C, options.m, options.n, options.k);
  if (err != cudaSuccess) {
    std::cerr << "cute-hgemm launch failed: " << cudaGetErrorString(err) << '\n';
    return false;
  }
  return CUDA_CHECK(cudaGetLastError());
}

bool launch_cute_hgemm_no_reg_prefetch(const Options &options, cute::half_t *A,
                                       cute::half_t *B, cute::half_t *C) {
  if (options.m % 128 != 0 || options.n % 128 != 0 || options.k % 64 != 0) {
    std::cerr << "cute-hgemm-noreg requires M and N to be multiples of 128, "
                 "and K to be a multiple of 64.\n";
    return false;
  }
  cudaError_t err = cute_hgemm::launch_hgemm_128x128_nn_no_reg_prefetch(
      A, B, C, options.m, options.n, options.k);
  if (err != cudaSuccess) {
    std::cerr << "cute-hgemm-noreg launch failed: "
              << cudaGetErrorString(err) << '\n';
    return false;
  }
  return CUDA_CHECK(cudaGetLastError());
}

bool launch_cutlass_hgemm(const Options &options, cute::half_t *A,
                          cute::half_t *B, cute::half_t *C) {
  if (options.n % 8 != 0 || options.k % 8 != 0) {
    std::cerr << "cutlass-hgemm requires N and K to be multiples of 8.\n";
    return false;
  }

  auto *cutlass_A = reinterpret_cast<cutlass::half_t *>(A);
  auto *cutlass_B = reinterpret_cast<cutlass::half_t *>(B);
  auto *cutlass_C = reinterpret_cast<cutlass::half_t *>(C);
  cutlass::Status status = cutlass_hgemm::launch_hgemm_sm80_tensorop(
      cutlass_A, cutlass_B, cutlass_C, options.m, options.n, options.k);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "cutlass-hgemm failed: "
              << cutlass::cutlassGetStatusString(status) << '\n';
    return false;
  }
  return CUDA_CHECK(cudaGetLastError());
}

#endif

#if ENABLE_SM90_KERNELS
bool launch_sm90_hgemm_pingpong(const Options &options, cute::half_t *A,
                                cute::half_t *B, cute::half_t *C) {
  static_assert(sizeof(cute::half_t) == sizeof(half));
  if (options.m % 128 != 0 || options.n % 128 != 0 || options.k % 64 != 0) {
    std::cerr << "sm90-hgemm-pingpong requires M and N to be multiples of 128, "
                 "and K to be a multiple of 64.\n";
    return false;
  }

  int device = 0;
  cudaDeviceProp props{};
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) {
    std::cerr << "sm90-hgemm-pingpong cudaGetDevice failed: "
              << cudaGetErrorString(err) << '\n';
    return false;
  }
  err = cudaGetDeviceProperties(&props, device);
  if (err != cudaSuccess) {
    std::cerr << "sm90-hgemm-pingpong cudaGetDeviceProperties failed: "
              << cudaGetErrorString(err) << '\n';
    return false;
  }
  if (props.major < 9) {
    std::cerr << "sm90-hgemm-pingpong requires an SM90+ GPU. Current device is "
              << props.major << "." << props.minor << ".\n";
    return false;
  }

  auto *half_A = reinterpret_cast<half *>(A);
  auto *half_B = reinterpret_cast<half *>(B);
  auto *half_C = reinterpret_cast<half *>(C);
  err = sm90_hgemm_pingpong::launch_hgemm_128x128x64_pingpong(
      half_A, half_B, half_C, options.m, options.n, options.k);
  if (err != cudaSuccess) {
    std::cerr << "sm90-hgemm-pingpong launch failed: "
              << cudaGetErrorString(err)
              << " (build for Hopper with -DCMAKE_CUDA_ARCHITECTURES=90a, "
                 "or 90 if your toolchain accepts the GMMA/TMA asm there).\n";
    return false;
  }
  return CUDA_CHECK(cudaGetLastError());
}

bool launch_sm90_hgemm_cooperative(const Options &options,
                                   cute::half_t *A, cute::half_t *B,
                                   cute::half_t *C) {
  static_assert(sizeof(cute::half_t) == sizeof(half));
  if (options.m % 128 != 0 || options.n % 128 != 0 || options.k % 64 != 0) {
    std::cerr << "sm90-hgemm-cooperative requires M and N to be multiples of 128, "
                 "and K to be a multiple of 64.\n";
    return false;
  }

  int device = 0;
  cudaDeviceProp props{};
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) {
    std::cerr << "sm90-hgemm-cooperative cudaGetDevice failed: "
              << cudaGetErrorString(err) << '\n';
    return false;
  }
  err = cudaGetDeviceProperties(&props, device);
  if (err != cudaSuccess) {
    std::cerr << "sm90-hgemm-cooperative cudaGetDeviceProperties failed: "
              << cudaGetErrorString(err) << '\n';
    return false;
  }
  if (props.major < 9) {
    std::cerr << "sm90-hgemm-cooperative requires an SM90+ GPU. Current device is "
              << props.major << "." << props.minor << ".\n";
    return false;
  }

  auto *half_A = reinterpret_cast<half *>(A);
  auto *half_B = reinterpret_cast<half *>(B);
  auto *half_C = reinterpret_cast<half *>(C);
  err = sm90_hgemm_cooperative::launch_hgemm_128x128x64_cooperative(
      half_A, half_B, half_C, options.m, options.n, options.k);
  if (err != cudaSuccess) {
    std::cerr << "sm90-hgemm-cooperative launch failed: "
              << cudaGetErrorString(err)
              << " (build for Hopper with -DCMAKE_CUDA_ARCHITECTURES=90a, "
                 "or 90 if your toolchain accepts the GMMA/TMA asm there).\n";
    return false;
  }
  return CUDA_CHECK(cudaGetLastError());
}

bool launch_cutlass_sm90_hgemm_pingpong(const Options &options,
                                        cute::half_t *A, cute::half_t *B,
                                        cute::half_t *C) {
  static_assert(sizeof(cute::half_t) == sizeof(cutlass::half_t));
  if (options.n % 8 != 0 || options.k % 8 != 0) {
    std::cerr << "cutlass-sm90-hgemm-pingpong requires N and K to be "
                 "multiples of 8 for 16-byte TMA alignment.\n";
    return false;
  }
  if (!require_sm90_device("cutlass-sm90-hgemm-pingpong")) {
    return false;
  }

  auto const *cutlass_A = reinterpret_cast<cutlass::half_t const *>(A);
  auto const *cutlass_B = reinterpret_cast<cutlass::half_t const *>(B);
  auto *cutlass_C = reinterpret_cast<cutlass::half_t *>(C);
  cutlass::Status status = cutlass_sm90_hgemm::launch_hgemm_pingpong(
      cutlass_A, cutlass_B, cutlass_C, options.m, options.n, options.k);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "cutlass-sm90-hgemm-pingpong failed: "
              << cutlass::cutlassGetStatusString(status)
              << " (build for Hopper with -DCMAKE_CUDA_ARCHITECTURES=90a).\n";
    return false;
  }
  return CUDA_CHECK(cudaGetLastError());
}

bool launch_cutlass_sm90_hgemm_cooperative(const Options &options,
                                           cute::half_t *A, cute::half_t *B,
                                           cute::half_t *C) {
  static_assert(sizeof(cute::half_t) == sizeof(cutlass::half_t));
  if (options.n % 8 != 0 || options.k % 8 != 0) {
    std::cerr << "cutlass-sm90-hgemm-cooperative requires N and K to be "
                 "multiples of 8 for 16-byte TMA alignment.\n";
    return false;
  }
  if (!require_sm90_device("cutlass-sm90-hgemm-cooperative")) {
    return false;
  }

  auto const *cutlass_A = reinterpret_cast<cutlass::half_t const *>(A);
  auto const *cutlass_B = reinterpret_cast<cutlass::half_t const *>(B);
  auto *cutlass_C = reinterpret_cast<cutlass::half_t *>(C);
  cutlass::Status status = cutlass_sm90_hgemm::launch_hgemm_cooperative(
      cutlass_A, cutlass_B, cutlass_C, options.m, options.n, options.k);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "cutlass-sm90-hgemm-cooperative failed: "
              << cutlass::cutlassGetStatusString(status)
              << " (build for Hopper with -DCMAKE_CUDA_ARCHITECTURES=90a).\n";
    return false;
  }
  return CUDA_CHECK(cudaGetLastError());
}

#endif

bool launch_cublas_hgemm(const Options &options, cute::half_t *A, cute::half_t *B,
                         cute::half_t *C) {
  static CuBlasHandle handle;
  if (!handle.ok()) {
    std::cerr << "cublasCreate failed\n";
    return false;
  }
  __half alpha = __float2half(1.0F);
  __half beta  = __float2half(0.0F);
  cublasStatus_t status = cublasGemmEx(
      handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
      options.n, options.m, options.k,
      &alpha,
      B, CUDA_R_16F, options.n,
      A, CUDA_R_16F, options.k,
      &beta,
      C, CUDA_R_16F, options.n,
      CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasGemmEx (HGEMM) failed\n";
    return false;
  }
  return CUDA_CHECK(cudaGetLastError());
}

bool launch_cublas_hgemm_fp32acc(const Options &options, cute::half_t *A, cute::half_t *B,
                                 cute::half_t *C) {
  static CuBlasHandle handle;
  if (!handle.ok()) {
    std::cerr << "cublasCreate failed\n";
    return false;
  }
  float alpha = 1.0F;
  float beta  = 0.0F;
  cublasStatus_t status = cublasGemmEx(
      handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
      options.n, options.m, options.k,
      &alpha,
      B, CUDA_R_16F, options.n,
      A, CUDA_R_16F, options.k,
      &beta,
      C, CUDA_R_16F, options.n,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasGemmEx (HGEMM fp32acc) failed\n";
    return false;
  }
  return CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Launcher dispatch
// ---------------------------------------------------------------------------

LaunchFn select_launcher(KernelType type) {
  switch (type) {
#if ENABLE_SM80_KERNELS
    case KernelType::SgemmCustom:                 return launch_custom;
    case KernelType::SgemmCutlassLikeS5:      return launch_cutlass_like_stage5;
    case KernelType::SgemmCutlassLikeS5OneCta: return launch_cutlass_like_stage5_one_cta;
    case KernelType::SgemmCutlassLikeS5WarpOrder: return launch_cutlass_like_stage5_warp_order;
    case KernelType::SgemmCutlassLikeS5Schedule:  return launch_cutlass_like_stage5_schedule;
    case KernelType::SgemmCutlassLikeS5CopySchedule: return launch_cutlass_like_stage5_copy_schedule;
    case KernelType::SgemmCutlassLikeS5MmaOrder:  return launch_cutlass_like_stage5_mma_order;
    case KernelType::SgemmCutlassRefS5:       return launch_cutlass_ref_stage5;
    case KernelType::SgemmExternalDb:   return launch_external_double_buffer;
    case KernelType::SgemmExternalNodb: return launch_external_no_double_buffer;
#endif
    case KernelType::SgemmNaive:                  return launch_naive;
    case KernelType::SgemmCuBlas:                 return launch_cublas;
    case KernelType::HgemmCute:
    case KernelType::HgemmCuteNoreg:
    case KernelType::HgemmCutlassSm80:
    case KernelType::HgemmSm90Pingpong:
    case KernelType::HgemmCutlassSm90Pingpong:
    case KernelType::HgemmCutlassSm90Cooperative:
    case KernelType::HgemmCuBlasFp16Acc:
    case KernelType::HgemmCuBlasFp32Acc:
      return nullptr;
  }
  return nullptr;
}

HalfLaunchFn select_half_launcher(KernelType type) {
  switch (type) {
#if ENABLE_SM80_KERNELS
    case KernelType::HgemmCute:              return launch_cute_hgemm;
    case KernelType::HgemmCuteNoreg: return launch_cute_hgemm_no_reg_prefetch;
    case KernelType::HgemmCutlassSm80:           return launch_cutlass_hgemm;
#endif
#if ENABLE_SM90_KERNELS
    case KernelType::HgemmSm90Pingpong:      return launch_sm90_hgemm_pingpong;
    case KernelType::HgemmSm90Cooperative:   return launch_sm90_hgemm_cooperative;
    case KernelType::HgemmCutlassSm90Pingpong: return launch_cutlass_sm90_hgemm_pingpong;
    case KernelType::HgemmCutlassSm90Cooperative: return launch_cutlass_sm90_hgemm_cooperative;
#endif
    case KernelType::HgemmCuBlasFp16Acc:            return launch_cublas_hgemm;
    case KernelType::HgemmCuBlasFp32Acc:            return launch_cublas_hgemm_fp32acc;
    default:                                 return nullptr;
  }
}

// ---------------------------------------------------------------------------
// Benchmark helpers
// ---------------------------------------------------------------------------

bool run_once(const Options &options, LaunchFn launch,
              float *A, float *B, float *C) {
  if (!launch(options, A, B, C)) return false;
  return CUDA_CHECK(cudaDeviceSynchronize());
}

BenchmarkResult run_benchmark(const Options &options, KernelType kt, LaunchFn launch,
                              float *A, float *B, float *C) {
  BenchmarkResult result{};

  for (int i = 0; i < options.warmup; i++) {
    if (!launch(options, A, B, C)) return result;
  }
  if (!CUDA_CHECK(cudaDeviceSynchronize())) return result;

  CudaEventTimer timer;
  if (!timer.record_start()) return result;

  for (int i = 0; i < options.iterations; i++) {
    if (!launch(options, A, B, C)) return result;
  }

  if (!timer.record_stop_and_sync()) return result;

  float elapsed_ms = 0.0F;
  if (!timer.elapsed_ms(&elapsed_ms)) return result;

  result.avg_ms = static_cast<double>(elapsed_ms) / static_cast<double>(options.iterations);
  const double flops = 2.0 * static_cast<double>(options.m) *
                       static_cast<double>(options.n) *
                       static_cast<double>(options.k);
  result.tflops = flops / (result.avg_ms * 1.0e-3) / 1.0e12;

  std::cout << std::fixed << std::setprecision(4)
            << std::left << std::setw(15) << kernel_name(kt)
            << " M=" << options.m << " N=" << options.n << " K=" << options.k
            << " warmup=" << options.warmup
            << " iters=" << options.iterations
            << " avg_ms=" << result.avg_ms
            << " TFLOP/s=" << result.tflops << '\n';
  return result;
}

bool run_once_half(const Options &options, HalfLaunchFn launch,
                   cute::half_t *A, cute::half_t *B, cute::half_t *C) {
  if (!launch(options, A, B, C)) return false;
  return CUDA_CHECK(cudaDeviceSynchronize());
}

BenchmarkResult run_benchmark_half(const Options &options, KernelType kt,
                                   HalfLaunchFn launch,
                                   cute::half_t *A, cute::half_t *B,
                                   cute::half_t *C) {
  BenchmarkResult result{};

  for (int i = 0; i < options.warmup; i++) {
    if (!launch(options, A, B, C)) return result;
  }
  if (!CUDA_CHECK(cudaDeviceSynchronize())) return result;

  CudaEventTimer timer;
  if (!timer.record_start()) return result;

  for (int i = 0; i < options.iterations; i++) {
    if (!launch(options, A, B, C)) return result;
  }

  if (!timer.record_stop_and_sync()) return result;

  float elapsed_ms = 0.0F;
  if (!timer.elapsed_ms(&elapsed_ms)) return result;

  result.avg_ms = static_cast<double>(elapsed_ms) / static_cast<double>(options.iterations);
  const double flops = 2.0 * static_cast<double>(options.m) *
                       static_cast<double>(options.n) *
                       static_cast<double>(options.k);
  result.tflops = flops / (result.avg_ms * 1.0e-3) / 1.0e12;

  std::cout << std::fixed << std::setprecision(4)
            << std::left << std::setw(15) << kernel_name(kt)
            << " M=" << options.m << " N=" << options.n << " K=" << options.k
            << " warmup=" << options.warmup
            << " iters=" << options.iterations
            << " avg_ms=" << result.avg_ms
            << " TFLOP/s=" << result.tflops << '\n';
  return result;
}

void print_comparison(const std::vector<std::pair<KernelType, BenchmarkResult>> &results) {
  std::size_t baseline_idx = 0;
  for (std::size_t i = 0; i < results.size(); ++i) {
    if (results[i].first == KernelType::SgemmCuBlas ||
        results[i].first == KernelType::HgemmCuBlasFp16Acc) {
      baseline_idx = i;
      break;
    }
  }
  const double baseline_tflops = results[baseline_idx].second.tflops;
  const char *baseline_name = kernel_name(results[baseline_idx].first);

  std::cout << "\n--- Relative speedup vs " << baseline_name << " ---\n";
  for (const auto &[kt, res] : results) {
    double ratio = (baseline_tflops > 0.0) ? (res.tflops / baseline_tflops) : 0.0;
    std::cout << std::left << std::setw(15) << kernel_name(kt)
              << std::fixed << std::setprecision(2) << ratio << "x\n";
  }
}
