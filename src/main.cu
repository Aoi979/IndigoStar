#include "core/cli.h"
#include "core/benchmark.h"
#include "core/device_utils.h"
#include "core/data_utils.h"
#include "core/verify.h"

#include <cuda_runtime.h>
#include <cute/numeric/numeric_types.hpp>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <optional>
#include <random>
#include <vector>

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

  bool needs_hgemm = false;
  for (KernelType kt : options.kernels) {
    if (is_hgemm_kernel(kt)) {
      needs_hgemm = true;
      break;
    }
  }

  std::mt19937 rng(options.seed);
  std::vector<float> host_A(elements_A);
  std::vector<float> host_B(elements_B);
  std::vector<float> host_C(elements_C, 0.0F);

  fill_random(&host_A, options.k, &rng);
  fill_random(&host_B, options.k, &rng);

  DeviceBuffer<float> device_A, device_B, device_C;
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

  // Half-precision host buffers and device buffers
  std::vector<cute::half_t> host_A_h, host_B_h, host_C_h;
  DeviceBuffer<cute::half_t> device_A_h, device_B_h, device_C_h;
  if (needs_hgemm) {
    host_A_h.resize(elements_A);
    host_B_h.resize(elements_B);
    host_C_h.resize(elements_C, cute::half_t(0.0F));
    for (std::size_t i = 0; i < elements_A; ++i)
      host_A_h[i] = cute::half_t(host_A[i]);
    for (std::size_t i = 0; i < elements_B; ++i)
      host_B_h[i] = cute::half_t(host_B[i]);

    if (!device_A_h.allocate(elements_A, "cudaMalloc A_h") ||
        !device_B_h.allocate(elements_B, "cudaMalloc B_h") ||
        !device_C_h.allocate(elements_C, "cudaMalloc C_h")) {
      return 1;
    }

    if (!CUDA_CHECK(cudaMemcpy(device_A_h.get(), host_A_h.data(),
                               elements_A * sizeof(cute::half_t),
                               cudaMemcpyHostToDevice)) ||
        !CUDA_CHECK(cudaMemcpy(device_B_h.get(), host_B_h.data(),
                               elements_B * sizeof(cute::half_t),
                               cudaMemcpyHostToDevice))) {
      return 1;
    }
  }

  // Pre-compute CPU reference if verification is requested.
  std::optional<std::vector<float>> cpu_ref;
  std::optional<std::vector<cute::half_t>> cpu_ref_h;
  if (options.verify) {
    cpu_ref = compute_reference(options.m, options.n, options.k, host_A, host_B);
    if (needs_hgemm) {
      cpu_ref_h = compute_reference_half(options.m, options.n, options.k,
                                         host_A_h, host_B_h);
    }
  }

  std::vector<std::pair<KernelType, BenchmarkResult>> benchmark_results;

  for (KernelType kt : options.kernels) {
    if (is_hgemm_kernel(kt)) {
      HalfLaunchFn launch = select_half_launcher(kt);
      if (!launch) {
        std::cerr << kernel_name(kt) << " launcher is not available on this architecture.\n";
        return 1;
      }

      if (options.benchmark) {
        auto result = run_benchmark_half(options, kt, launch,
                                         device_A_h.get(), device_B_h.get(),
                                         device_C_h.get());
        if (result.tflops <= 0.0) return 1;
        benchmark_results.emplace_back(kt, result);
      } else {
        if (!run_once_half(options, launch,
                           device_A_h.get(), device_B_h.get(), device_C_h.get())) {
          return 1;
        }
      }

      if (!CUDA_CHECK(cudaMemcpy(host_C_h.data(), device_C_h.get(),
                                 elements_C * sizeof(cute::half_t),
                                 cudaMemcpyDeviceToHost))) {
        return 1;
      }

      std::cout << std::setprecision(8)
                << kernel_name(kt) << " C[0]=" << static_cast<float>(host_C_h.front())
                << " checksum=" << checksum_half(host_C_h) << '\n';

      if (options.verify && cpu_ref_h.has_value()) {
        if (!verify_half_against_reference(host_C_h, *cpu_ref_h, options.n)) {
          std::cerr << "Verification FAILED for " << kernel_name(kt) << "\n";
          return 1;
        }
      }
    } else {
      LaunchFn launch = select_launcher(kt);
      if (!launch) {
        std::cerr << kernel_name(kt) << " launcher is not available on this architecture.\n";
        return 1;
      }

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
  }

  if (options.benchmark && benchmark_results.size() > 1) {
    print_comparison(benchmark_results);
  }

  return 0;
}
