#pragma once

#include "cli.h"
#include <cute/numeric/numeric_types.hpp>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>

using LaunchFn = bool(*)(const Options &, float *, float *, float *);
using HalfLaunchFn = bool(*)(const Options &, cute::half_t *, cute::half_t *, cute::half_t *);

LaunchFn select_launcher(KernelType type);
HalfLaunchFn select_half_launcher(KernelType type);

struct BenchmarkResult {
  double tflops = 0.0;
  double avg_ms = 0.0;
};

BenchmarkResult run_benchmark(const Options &options, KernelType kt, LaunchFn launch,
                              float *A, float *B, float *C);
BenchmarkResult run_benchmark_half(const Options &options, KernelType kt,
                                   HalfLaunchFn launch,
                                   cute::half_t *A, cute::half_t *B,
                                   cute::half_t *C);

bool run_once(const Options &options, LaunchFn launch,
              float *A, float *B, float *C);
bool run_once_half(const Options &options, HalfLaunchFn launch,
                   cute::half_t *A, cute::half_t *B, cute::half_t *C);

void print_comparison(const std::vector<std::pair<KernelType, BenchmarkResult>> &results);
