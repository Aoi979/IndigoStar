#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "sgemm.cuh"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

inline bool cuda_check(cudaError_t error, const char *call) {
  if (error == cudaSuccess) return true;
  std::cerr << call << " failed: " << cudaGetErrorString(error) << '\n';
  return false;
}

#define CUDA_CHECK(expr) cuda_check((expr), #expr)

// ---------------------------------------------------------------------------
// RAII wrappers
// ---------------------------------------------------------------------------

class DeviceBuffer {
public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }

  bool allocate(std::size_t elements, const char *name) {
    return cuda_check(cudaMalloc(&ptr_, elements * sizeof(float)), name);
  }
  float *get() const { return ptr_; }
private:
  float *ptr_ = nullptr;
};

class CudaEventTimer {
public:
  CudaEventTimer() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }
  ~CudaEventTimer() {
    if (start_) cudaEventDestroy(start_);
    if (stop_) cudaEventDestroy(stop_);
  }
  CudaEventTimer(const CudaEventTimer &) = delete;
  CudaEventTimer &operator=(const CudaEventTimer &) = delete;

  bool record_start() { return CUDA_CHECK(cudaEventRecord(start_)); }
  bool record_stop_and_sync() {
    return CUDA_CHECK(cudaEventRecord(stop_)) &&
           CUDA_CHECK(cudaEventSynchronize(stop_));
  }
  bool elapsed_ms(float *out) const {
    return CUDA_CHECK(cudaEventElapsedTime(out, start_, stop_));
  }
private:
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

class CuBlasHandle {
public:
  CuBlasHandle() { cublasCreate(&handle_); }
  ~CuBlasHandle() { if (handle_) cublasDestroy(handle_); }
  CuBlasHandle(const CuBlasHandle &) = delete;
  CuBlasHandle &operator=(const CuBlasHandle &) = delete;
  bool ok() const { return handle_ != nullptr; }
  cublasHandle_t get() const { return handle_; }
private:
  cublasHandle_t handle_ = nullptr;
};

// ---------------------------------------------------------------------------
// Options & CLI parsing
// ---------------------------------------------------------------------------

enum class KernelType {
  Custom,
  CutlassLikeStage5,
  CutlassLikeStage5OneCta,
  CutlassLikeStage5WarpOrder,
  CutlassLikeStage5Schedule,
  CutlassLikeStage5CopySchedule,
  CutlassLikeStage5MmaOrder,
  CutlassRefStage5,
  Naive,
  CuBlas,
  ExternalDoubleBuffer,
  ExternalNoDoubleBuffer,
};

struct Options {
  int m = 1024;
  int n = 1024;
  int k = 1024;
  int warmup = 10;
  int iterations = 100;
  std::uint32_t seed = 20260518U;
  bool benchmark = false;
  bool verify = false;
  std::vector<KernelType> kernels;
  bool help = false;
};

void print_usage(const char *program) {
  std::cerr <<
    "Usage: " << program << " [bench|--benchmark] [--kernel TYPE] ... [--m M] [--n N] [--k K]\n"
    "       " << program << " [--size N] [--iters N] [--warmup N] [--seed N] [--verify]\n"
    "\n"
    "Kernel types (can specify multiple):\n"
    "  custom        (default) sgemm_128x128x32\n"
    "  cutlass-stage5 cutlass-like 128x128x8 SGEMM with 5-stage cp.async\n"
    "  cutlass-stage5-1cta same kernel with extra smem to reduce CTA residency\n"
    "  cutlass-stage5-warporder same kernel with CUTLASS warp tile order\n"
    "  cutlass-stage5-schedule CUTLASS warp order plus CUTLASS-like copy schedule\n"
    "  cutlass-stage5-copyorder CUTLASS-like copy schedule only\n"
    "  cutlass-stage5-mmaorder CUTLASS SM80 thread-level FFMA order\n"
    "  cutlass-ref   CUTLASS SM80 SIMT 128x128x8 stage-5 reference\n"
    "  naive         naive sgemm\n"
    "  cublas        cuBLAS reference\n"
    "  external-db   external 128x128x16 SGEMM with smem double buffering\n"
    "  external-nodb external 128x128x16 SGEMM without smem double buffering\n"
    "\n"
    "Legacy flags (also supported):\n"
    "  --naive, --cublas, --external-db, --external-nodb, --cutlass-stage5, "
    "--cutlass-stage5-1cta, --cutlass-stage5-warporder, --cutlass-stage5-schedule, "
    "--cutlass-stage5-copyorder, --cutlass-stage5-mmaorder, --cutlass-ref\n";
}

bool parse_positive_int(std::string_view value, int *out) {
  int parsed = 0;
  auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (ec != std::errc{} || ptr != value.data() + value.size() || parsed <= 0)
    return false;
  *out = parsed;
  return true;
}

bool parse_seed(std::string_view value, std::uint32_t *out) {
  std::uint32_t parsed = 0;
  auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (ec != std::errc{} || ptr != value.data() + value.size())
    return false;
  *out = parsed;
  return true;
}

bool sv_starts_with(std::string_view s, std::string_view prefix) {
  return s.size() >= prefix.size() && s.substr(0, prefix.size()) == prefix;
}

std::optional<std::string_view> read_option_value(int argc, char **argv,
                                                  int *index,
                                                  std::string_view name) {
  const std::string_view arg(argv[*index]);
  if (arg == name) {
    if (*index + 1 >= argc) {
      std::cerr << name << " requires a value.\n";
      return std::nullopt;
    }
    *index += 1;
    return std::string_view(argv[*index]);
  }
  if (sv_starts_with(arg, std::string(name) + "=")) {
    return arg.substr(name.size() + 1);
  }
  return std::nullopt;
}

bool parse_kernel_type(std::string_view value, KernelType *out) {
  if (value == "custom")                   { *out = KernelType::Custom;            return true; }
  if (value == "cutlass-stage5")           { *out = KernelType::CutlassLikeStage5; return true; }
  if (value == "cutlass-stage5-1cta")      { *out = KernelType::CutlassLikeStage5OneCta; return true; }
  if (value == "cutlass-stage5-warporder") { *out = KernelType::CutlassLikeStage5WarpOrder; return true; }
  if (value == "cutlass-stage5-schedule")  { *out = KernelType::CutlassLikeStage5Schedule; return true; }
  if (value == "cutlass-stage5-copyorder") { *out = KernelType::CutlassLikeStage5CopySchedule; return true; }
  if (value == "cutlass-stage5-mmaorder")  { *out = KernelType::CutlassLikeStage5MmaOrder; return true; }
  if (value == "cutlass-ref")              { *out = KernelType::CutlassRefStage5;  return true; }
  if (value == "naive")                    { *out = KernelType::Naive;             return true; }
  if (value == "cublas")                   { *out = KernelType::CuBlas;            return true; }
  if (value == "external-db")              { *out = KernelType::ExternalDoubleBuffer;  return true; }
  if (value == "external-nodb")            { *out = KernelType::ExternalNoDoubleBuffer; return true; }
  return false;
}

bool parse_args(int argc, char **argv, Options *options) {
  for (int i = 1; i < argc; i++) {
    const std::string_view arg(argv[i]);

    if (arg == "bench" || arg == "--bench" || arg == "--benchmark") {
      options->benchmark = true;
      continue;
    }
    if (arg == "--verify") {
      options->verify = true;
      continue;
    }
    if (arg == "-h" || arg == "--help") {
      options->help = true;
      return true;
    }

    if (arg == "--naive")                    { options->kernels.push_back(KernelType::Naive);             continue; }
    if (arg == "--cublas")                   { options->kernels.push_back(KernelType::CuBlas);            continue; }
    if (arg == "--cutlass-stage5")           { options->kernels.push_back(KernelType::CutlassLikeStage5); continue; }
    if (arg == "--cutlass-stage5-1cta")      { options->kernels.push_back(KernelType::CutlassLikeStage5OneCta); continue; }
    if (arg == "--cutlass-stage5-warporder") { options->kernels.push_back(KernelType::CutlassLikeStage5WarpOrder); continue; }
    if (arg == "--cutlass-stage5-schedule")  { options->kernels.push_back(KernelType::CutlassLikeStage5Schedule); continue; }
    if (arg == "--cutlass-stage5-copyorder") { options->kernels.push_back(KernelType::CutlassLikeStage5CopySchedule); continue; }
    if (arg == "--cutlass-stage5-mmaorder")  { options->kernels.push_back(KernelType::CutlassLikeStage5MmaOrder); continue; }
    if (arg == "--cutlass-ref")              { options->kernels.push_back(KernelType::CutlassRefStage5);  continue; }
    if (arg == "--external-db")              { options->kernels.push_back(KernelType::ExternalDoubleBuffer);  continue; }
    if (arg == "--external-nodb")            { options->kernels.push_back(KernelType::ExternalNoDoubleBuffer); continue; }

    if (auto value = read_option_value(argc, argv, &i, "--kernel")) {
      KernelType kt;
      if (!parse_kernel_type(*value, &kt)) {
        std::cerr << "--kernel must be one of: custom, cutlass-stage5, "
                     "cutlass-stage5-1cta, cutlass-stage5-warporder, "
                     "cutlass-stage5-schedule, cutlass-stage5-copyorder, "
                     "cutlass-stage5-mmaorder, cutlass-ref, naive, cublas, "
                     "external-db, external-nodb\n";
        return false;
      }
      options->kernels.push_back(kt);
      continue;
    }

    if (auto value = read_option_value(argc, argv, &i, "--size")) {
      int size = 0;
      if (!value || !parse_positive_int(*value, &size)) {
        std::cerr << "--size must be a positive integer.\n";
        return false;
      }
      options->m = size;
      options->n = size;
      options->k = size;
      continue;
    }

    if (auto value = read_option_value(argc, argv, &i, "--m")) {
      if (!value || !parse_positive_int(*value, &options->m)) {
        std::cerr << "--m must be a positive integer.\n";
        return false;
      }
      continue;
    }
    if (auto value = read_option_value(argc, argv, &i, "--n")) {
      if (!value || !parse_positive_int(*value, &options->n)) {
        std::cerr << "--n must be a positive integer.\n";
        return false;
      }
      continue;
    }
    if (auto value = read_option_value(argc, argv, &i, "--k")) {
      if (!value || !parse_positive_int(*value, &options->k)) {
        std::cerr << "--k must be a positive integer.\n";
        return false;
      }
      continue;
    }

    if (auto value = read_option_value(argc, argv, &i, "--iters")) {
      if (!value || !parse_positive_int(*value, &options->iterations)) {
        std::cerr << "--iters must be a positive integer.\n";
        return false;
      }
      continue;
    }
    if (auto value = read_option_value(argc, argv, &i, "--iterations")) {
      if (!value || !parse_positive_int(*value, &options->iterations)) {
        std::cerr << "--iterations must be a positive integer.\n";
        return false;
      }
      continue;
    }
    if (auto value = read_option_value(argc, argv, &i, "--warmup")) {
      if (!value || !parse_positive_int(*value, &options->warmup)) {
        std::cerr << "--warmup must be a positive integer.\n";
        return false;
      }
      continue;
    }
    if (auto value = read_option_value(argc, argv, &i, "--seed")) {
      if (!value || !parse_seed(*value, &options->seed)) {
        std::cerr << "--seed must be an unsigned 32-bit integer.\n";
        return false;
      }
      continue;
    }

    std::cerr << "Unknown argument: " << arg << '\n';
    print_usage(argv[0]);
    return false;
  }

  if (options->kernels.empty()) {
    options->kernels.push_back(KernelType::Custom);
  }
  return true;
}

// ---------------------------------------------------------------------------
// Data utilities
// ---------------------------------------------------------------------------

std::size_t matrix_elements(int rows, int cols) {
  return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
}

void fill_random(std::vector<float> *values, int fan_in, std::mt19937 *rng) {
  const float limit = 1.0F / std::sqrt(static_cast<float>(fan_in));
  std::uniform_real_distribution<float> dist(-limit, limit);
  for (float &value : *values) value = dist(*rng);
}

double checksum(const std::vector<float> &values) {
  double sum = 0.0;
  for (float v : values) sum += static_cast<double>(v);
  return sum;
}

std::vector<float> compute_reference(int m, int n, int k,
                                     const std::vector<float> &A,
                                     const std::vector<float> &B) {
  std::vector<float> C(matrix_elements(m, n), 0.0F);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      double acc = 0.0;
      for (int l = 0; l < k; ++l) {
        acc += static_cast<double>(A[i * k + l]) *
               static_cast<double>(B[l * n + j]);
      }
      C[i * n + j] = static_cast<float>(acc);
    }
  }
  return C;
}

bool verify_against_reference(const std::vector<float> &host_C,
                              const std::vector<float> &ref_C) {
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  for (std::size_t i = 0; i < host_C.size(); ++i) {
    const double actual   = static_cast<double>(host_C[i]);
    const double expected = static_cast<double>(ref_C[i]);
    const double abs_error = std::abs(actual - expected);
    const double rel_error = abs_error / std::max(1.0, std::abs(expected));
    max_abs_error = std::max(max_abs_error, abs_error);
    max_rel_error = std::max(max_rel_error, rel_error);
  }
  std::cout << "verify max_abs_error=" << max_abs_error
            << " max_rel_error=" << max_rel_error << '\n';
  return max_abs_error < 1.0e-3 || max_rel_error < 1.0e-3;
}

// ---------------------------------------------------------------------------
// Kernel launch wrappers (normalize signatures)
// ---------------------------------------------------------------------------

using LaunchFn = bool(*)(const Options &, float *, float *, float *);

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

LaunchFn select_launcher(KernelType type) {
  switch (type) {
    case KernelType::Custom:                 return launch_custom;
    case KernelType::CutlassLikeStage5:      return launch_cutlass_like_stage5;
    case KernelType::CutlassLikeStage5OneCta: return launch_cutlass_like_stage5_one_cta;
    case KernelType::CutlassLikeStage5WarpOrder: return launch_cutlass_like_stage5_warp_order;
    case KernelType::CutlassLikeStage5Schedule:  return launch_cutlass_like_stage5_schedule;
    case KernelType::CutlassLikeStage5CopySchedule: return launch_cutlass_like_stage5_copy_schedule;
    case KernelType::CutlassLikeStage5MmaOrder:  return launch_cutlass_like_stage5_mma_order;
    case KernelType::CutlassRefStage5:       return launch_cutlass_ref_stage5;
    case KernelType::Naive:                  return launch_naive;
    case KernelType::CuBlas:                 return launch_cublas;
    case KernelType::ExternalDoubleBuffer:   return launch_external_double_buffer;
    case KernelType::ExternalNoDoubleBuffer: return launch_external_no_double_buffer;
  }
  return launch_custom;
}

const char *kernel_name(KernelType type) {
  switch (type) {
    case KernelType::Custom:                 return "custom";
    case KernelType::CutlassLikeStage5:      return "cutlass_stage5";
    case KernelType::CutlassLikeStage5OneCta: return "cutlass_stage5_1cta";
    case KernelType::CutlassLikeStage5WarpOrder: return "cutlass_stage5_wo";
    case KernelType::CutlassLikeStage5Schedule:  return "cutlass_stage5_sched";
    case KernelType::CutlassLikeStage5CopySchedule: return "cutlass_stage5_copy";
    case KernelType::CutlassLikeStage5MmaOrder:  return "cutlass_stage5_mma";
    case KernelType::CutlassRefStage5:       return "cutlass_ref";
    case KernelType::Naive:                  return "naive";
    case KernelType::CuBlas:                 return "cublas";
    case KernelType::ExternalDoubleBuffer:   return "external_db";
    case KernelType::ExternalNoDoubleBuffer: return "external_nodb";
  }
  return "unknown";
}

// ---------------------------------------------------------------------------
// Benchmark / run
// ---------------------------------------------------------------------------

bool run_once(const Options &options, LaunchFn launch,
              float *A, float *B, float *C) {
  if (!launch(options, A, B, C)) return false;
  return CUDA_CHECK(cudaDeviceSynchronize());
}

struct BenchmarkResult {
  double tflops = 0.0;
  double avg_ms = 0.0;
};

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

void print_comparison(const std::vector<std::pair<KernelType, BenchmarkResult>> &results) {
  std::size_t baseline_idx = 0;
  for (std::size_t i = 0; i < results.size(); ++i) {
    if (results[i].first == KernelType::CuBlas) {
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

} // namespace

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
  Options options;
  if (!parse_args(argc, argv, &options)) return 1;
  if (options.help) { print_usage(argv[0]); return 0; }

  int device_count = 0;
  if (!CUDA_CHECK(cudaGetDeviceCount(&device_count)) || device_count == 0) {
    std::cerr << "No CUDA device found.\n";
    return 1;
  }

  const std::size_t elements_A = matrix_elements(options.m, options.k);
  const std::size_t elements_B = matrix_elements(options.k, options.n);
  const std::size_t elements_C = matrix_elements(options.m, options.n);

  std::mt19937 rng(options.seed);
  std::vector<float> host_A(elements_A);
  std::vector<float> host_B(elements_B);
  std::vector<float> host_C(elements_C, 0.0F);

  fill_random(&host_A, options.k, &rng);
  fill_random(&host_B, options.k, &rng);

  DeviceBuffer device_A, device_B, device_C;
  if (!device_A.allocate(elements_A, "cudaMalloc A") ||
      !device_B.allocate(elements_B, "cudaMalloc B") ||
      !device_C.allocate(elements_C, "cudaMalloc C")) {
    return 1;
  }

  if (!CUDA_CHECK(cudaMemcpy(device_A.get(), host_A.data(),
                             elements_A * sizeof(float), cudaMemcpyHostToDevice)) ||
      !CUDA_CHECK(cudaMemcpy(device_B.get(), host_B.data(),
                             elements_B * sizeof(float), cudaMemcpyHostToDevice))) {
    return 1;
  }

  // Pre-compute CPU reference if verification is requested.
  std::optional<std::vector<float>> cpu_ref;
  if (options.verify) {
    cpu_ref = compute_reference(options.m, options.n, options.k, host_A, host_B);
  }

  std::vector<std::pair<KernelType, BenchmarkResult>> benchmark_results;

  for (KernelType kt : options.kernels) {
    LaunchFn launch = select_launcher(kt);

    if (options.benchmark) {
      auto result = run_benchmark(options, kt, launch,
                                  device_A.get(), device_B.get(), device_C.get());
      if (result.tflops <= 0.0) return 1;
      benchmark_results.emplace_back(kt, result);
    } else {
      if (!run_once(options, launch,
                    device_A.get(), device_B.get(), device_C.get())) {
        return 1;
      }
    }

    if (!CUDA_CHECK(cudaMemcpy(host_C.data(), device_C.get(),
                               elements_C * sizeof(float), cudaMemcpyDeviceToHost))) {
      return 1;
    }

    std::cout << std::setprecision(8)
              << kernel_name(kt) << " C[0]=" << host_C.front()
              << " checksum=" << checksum(host_C) << '\n';

    if (options.verify && cpu_ref.has_value()) {
      if (!verify_against_reference(host_C, *cpu_ref)) {
        std::cerr << "Verification FAILED for " << kernel_name(kt) << "\n";
        return 1;
      }
    }
  }

  if (options.benchmark && benchmark_results.size() > 1) {
    print_comparison(benchmark_results);
  }

  return 0;
}
