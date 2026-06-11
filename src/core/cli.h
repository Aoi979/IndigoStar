#pragma once

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

enum class KernelType {
  SgemmCustom,
  SgemmCutlassLikeS5,
  SgemmCutlassLikeS5OneCta,
  SgemmCutlassLikeS5WarpOrder,
  SgemmCutlassLikeS5Schedule,
  SgemmCutlassLikeS5CopySchedule,
  SgemmCutlassLikeS5MmaOrder,
  SgemmCutlassRefS5,
  SgemmNaive,
  SgemmCuBlas,
  SgemmExternalDb,
  SgemmExternalNodb,
  HgemmCute,
  HgemmCuteNoreg,
  HgemmSm80Handwritten,
  HgemmCutlassSm80,
  HgemmSm90Pingpong,
  HgemmSm90Cooperative,
  HgemmCutlassSm90Pingpong,
  HgemmCutlassSm90Cooperative,
  HgemmCuBlasFp16Acc,
  HgemmCuBlasFp32Acc,
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

void print_usage(const char *program);
bool parse_args(int argc, char **argv, Options *options);
const char *kernel_name(KernelType type);
bool is_hgemm_kernel(KernelType type);
