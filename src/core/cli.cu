#include "cli.h"

void print_usage(const char *program) {
  std::cerr <<
    "Usage: " << program << " [bench|--benchmark] [--kernel TYPE] ... [--m M] [--n N] [--k K]\n"
    "       " << program << " [--size N] [--iters N] [--warmup N] [--seed N] [--verify]\n"
    "\n"
    "Kernel types (can specify multiple):\n"
    "  sgemm-custom           (default) handwritten SGEMM 128x128x32\n"
    "  sgemm-cutlass-like-s5  handwritten CUTLASS-like 128x128x8 stage-5 SGEMM\n"
    "  sgemm-cutlass-like-s5-1cta   same with 1 CTA/SM extra smem\n"
    "  sgemm-cutlass-like-s5-warporder  same with CUTLASS warp tile order\n"
    "  sgemm-cutlass-like-s5-schedule   CUTLASS warp order + copy schedule\n"
    "  sgemm-cutlass-like-s5-copyorder  CUTLASS-like copy schedule only\n"
    "  sgemm-cutlass-like-s5-mmaorder   CUTLASS SM80 thread-level FFMA order\n"
    "  sgemm-cutlass-ref-s5   CUTLASS library SM80 SIMT 128x128x8 stage-5\n"
    "  sgemm-naive            naive global-memory SGEMM\n"
    "  sgemm-cublas           cuBLAS SGEMM baseline\n"
    "  sgemm-external-db      external 128x128x16 SGEMM double-buffer\n"
    "  sgemm-external-nodb    external 128x128x16 SGEMM no double-buffer\n"
    "  hgemm-cute             CuTe handwritten 128x128x64 HGEMM\n"
    "  hgemm-cute-noreg       CuTe HGEMM without register prefetch\n"
    "  hgemm-cutlass-sm80     CUTLASS library SM80 TensorOp HGEMM\n"
    "  hgemm-sm90-pingpong    handwritten SM90 TMA+GMMA pingpong HGEMM\n"
    "  hgemm-cutlass-sm90-pp  CUTLASS library SM90 pingpong schedule\n"
    "  hgemm-cutlass-sm90-coop CUTLASS library SM90 cooperative schedule\n"
    "  hgemm-cublas-fp16acc   cuBLAS HGEMM baseline (fp16 accumulator)\n"
    "  hgemm-cublas-fp32acc   cuBLAS HGEMM baseline (fp32 accumulator)\n"
    "\n"
    "Legacy flags (also supported):\n"
    "  --naive, --cublas, --external-db, --external-nodb, --cutlass-stage5, "
    "--cutlass-stage5-1cta, --cutlass-stage5-warporder, --cutlass-stage5-schedule, "
    "--cutlass-stage5-copyorder, --cutlass-stage5-mmaorder, --cutlass-ref, "
    "--cute-hgemm, --cute-hgemm-noreg, --cutlass-hgemm, "
    "--sm90-hgemm-pingpong, --cutlass-sm90-hgemm-pingpong, "
    "--cutlass-sm90-hgemm-cooperative, --cublas-hgemm\n";
}

static bool parse_positive_int(std::string_view value, int *out) {
  int parsed = 0;
  auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (ec != std::errc{} || ptr != value.data() + value.size() || parsed <= 0)
    return false;
  *out = parsed;
  return true;
}

static bool parse_seed(std::string_view value, std::uint32_t *out) {
  std::uint32_t parsed = 0;
  auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (ec != std::errc{} || ptr != value.data() + value.size())
    return false;
  *out = parsed;
  return true;
}

static bool sv_starts_with(std::string_view s, std::string_view prefix) {
  return s.size() >= prefix.size() && s.substr(0, prefix.size()) == prefix;
}

static std::optional<std::string_view> read_option_value(int argc, char **argv,
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

static bool parse_kernel_type(std::string_view value, KernelType *out) {
  if (value == "sgemm-custom")                  { *out = KernelType::SgemmCustom;            return true; }
  if (value == "sgemm-cutlass-like-s5")         { *out = KernelType::SgemmCutlassLikeS5; return true; }
  if (value == "sgemm-cutlass-like-s5-1cta")    { *out = KernelType::SgemmCutlassLikeS5OneCta; return true; }
  if (value == "sgemm-cutlass-like-s5-warporder") { *out = KernelType::SgemmCutlassLikeS5WarpOrder; return true; }
  if (value == "sgemm-cutlass-like-s5-schedule")  { *out = KernelType::SgemmCutlassLikeS5Schedule; return true; }
  if (value == "sgemm-cutlass-like-s5-copyorder") { *out = KernelType::SgemmCutlassLikeS5CopySchedule; return true; }
  if (value == "sgemm-cutlass-like-s5-mmaorder")  { *out = KernelType::SgemmCutlassLikeS5MmaOrder; return true; }
  if (value == "sgemm-cutlass-ref-s5")          { *out = KernelType::SgemmCutlassRefS5;  return true; }
  if (value == "sgemm-naive")                   { *out = KernelType::SgemmNaive;             return true; }
  if (value == "sgemm-cublas")                  { *out = KernelType::SgemmCuBlas;            return true; }
  if (value == "sgemm-external-db")             { *out = KernelType::SgemmExternalDb;  return true; }
  if (value == "sgemm-external-nodb")           { *out = KernelType::SgemmExternalNodb; return true; }
  if (value == "hgemm-cute")                    { *out = KernelType::HgemmCute;             return true; }
  if (value == "hgemm-cute-noreg")              { *out = KernelType::HgemmCuteNoreg; return true; }
  if (value == "hgemm-cutlass-sm80")            { *out = KernelType::HgemmCutlassSm80;          return true; }
  if (value == "hgemm-sm90-pingpong")           { *out = KernelType::HgemmSm90Pingpong;     return true; }
  if (value == "hgemm-cutlass-sm90-pp" ||
      value == "hgemm-cutlass-sm90-pingpong")   { *out = KernelType::HgemmCutlassSm90Pingpong; return true; }
  if (value == "hgemm-cutlass-sm90-coop" ||
      value == "hgemm-cutlass-sm90-cooperative") { *out = KernelType::HgemmCutlassSm90Cooperative; return true; }
  if (value == "hgemm-cublas-fp16acc")          { *out = KernelType::HgemmCuBlasFp16Acc;           return true; }
  if (value == "hgemm-cublas-fp32acc")          { *out = KernelType::HgemmCuBlasFp32Acc;           return true; }
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

    if (arg == "--naive")                    { options->kernels.push_back(KernelType::SgemmNaive);             continue; }
    if (arg == "--cublas")                   { options->kernels.push_back(KernelType::SgemmCuBlas);            continue; }
    if (arg == "--cutlass-stage5")           { options->kernels.push_back(KernelType::SgemmCutlassLikeS5); continue; }
    if (arg == "--cutlass-stage5-1cta")      { options->kernels.push_back(KernelType::SgemmCutlassLikeS5OneCta); continue; }
    if (arg == "--cutlass-stage5-warporder") { options->kernels.push_back(KernelType::SgemmCutlassLikeS5WarpOrder); continue; }
    if (arg == "--cutlass-stage5-schedule")  { options->kernels.push_back(KernelType::SgemmCutlassLikeS5Schedule); continue; }
    if (arg == "--cutlass-stage5-copyorder") { options->kernels.push_back(KernelType::SgemmCutlassLikeS5CopySchedule); continue; }
    if (arg == "--cutlass-stage5-mmaorder")  { options->kernels.push_back(KernelType::SgemmCutlassLikeS5MmaOrder); continue; }
    if (arg == "--cutlass-ref")              { options->kernels.push_back(KernelType::SgemmCutlassRefS5);  continue; }
    if (arg == "--external-db")              { options->kernels.push_back(KernelType::SgemmExternalDb);  continue; }
    if (arg == "--external-nodb")            { options->kernels.push_back(KernelType::SgemmExternalNodb); continue; }
    if (arg == "--cute-hgemm")               { options->kernels.push_back(KernelType::HgemmCute);             continue; }
    if (arg == "--cute-hgemm-noreg")         { options->kernels.push_back(KernelType::HgemmCuteNoreg); continue; }
    if (arg == "--cutlass-hgemm")            { options->kernels.push_back(KernelType::HgemmCutlassSm80);          continue; }
    if (arg == "--sm90-hgemm-pingpong")      { options->kernels.push_back(KernelType::HgemmSm90Pingpong);     continue; }
    if (arg == "--cutlass-sm90-hgemm-pingpong") { options->kernels.push_back(KernelType::HgemmCutlassSm90Pingpong); continue; }
    if (arg == "--cutlass-sm90-hgemm-cooperative") { options->kernels.push_back(KernelType::HgemmCutlassSm90Cooperative); continue; }
    if (arg == "--cublas-hgemm")             { options->kernels.push_back(KernelType::HgemmCuBlasFp16Acc);           continue; }
    if (arg == "--cublas-hgemm-fp32acc")     { options->kernels.push_back(KernelType::HgemmCuBlasFp32Acc);           continue; }

    if (auto value = read_option_value(argc, argv, &i, "--kernel")) {
      KernelType kt;
      if (!parse_kernel_type(*value, &kt)) {
        std::cerr << "--kernel must be one of: sgemm-custom, sgemm-cutlass-like-s5, "
                     "sgemm-cutlass-like-s5-1cta, sgemm-cutlass-like-s5-warporder, "
                     "sgemm-cutlass-like-s5-schedule, sgemm-cutlass-like-s5-copyorder, "
                     "sgemm-cutlass-like-s5-mmaorder, sgemm-cutlass-ref-s5, sgemm-naive, sgemm-cublas, "
                     "sgemm-external-db, sgemm-external-nodb, hgemm-cute, "
                     "hgemm-cute-noreg, hgemm-cutlass-sm80, "
                     "hgemm-sm90-pingpong, hgemm-cutlass-sm90-pp, "
                     "hgemm-cutlass-sm90-coop, hgemm-cublas-fp16acc, hgemm-cublas-fp32acc\n";
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
    options->kernels.push_back(KernelType::SgemmCustom);
  }

  // Validate architecture support
  for (auto kt : options->kernels) {
#if !ENABLE_SM80_KERNELS
    if (kt == KernelType::SgemmCustom ||
        kt == KernelType::SgemmCutlassLikeS5 ||
        kt == KernelType::SgemmCutlassLikeS5OneCta ||
        kt == KernelType::SgemmCutlassLikeS5WarpOrder ||
        kt == KernelType::SgemmCutlassLikeS5Schedule ||
        kt == KernelType::SgemmCutlassLikeS5CopySchedule ||
        kt == KernelType::SgemmCutlassLikeS5MmaOrder ||
        kt == KernelType::SgemmCutlassRefS5 ||
        kt == KernelType::SgemmExternalDb ||
        kt == KernelType::SgemmExternalNodb ||
        kt == KernelType::HgemmCute ||
        kt == KernelType::HgemmCuteNoreg ||
        kt == KernelType::HgemmCutlassSm80) {
      std::cerr << kernel_name(kt) << " is not available on this architecture "
                   "(built without SM80 kernel support).\n";
      return false;
    }
#endif
#if !ENABLE_SM90_KERNELS
    if (kt == KernelType::HgemmSm90Pingpong ||
        kt == KernelType::HgemmCutlassSm90Pingpong ||
        kt == KernelType::HgemmCutlassSm90Cooperative) {
      std::cerr << kernel_name(kt) << " is not available on this architecture "
                   "(built without SM90 kernel support).\n";
      return false;
    }
#endif
  }

  // Auto-add cuBLAS baseline
  bool has_sgemm = false, has_hgemm = false;
  bool has_cublas = false, has_cublas_hgemm = false;
  for (auto kt : options->kernels) {
    if (kt == KernelType::SgemmCuBlas) has_cublas = true;
    if (kt == KernelType::HgemmCuBlasFp16Acc) has_cublas_hgemm = true;
    if (is_hgemm_kernel(kt) && kt != KernelType::HgemmCuBlasFp16Acc && kt != KernelType::HgemmCuBlasFp32Acc) has_hgemm = true;
    if (!is_hgemm_kernel(kt) && kt != KernelType::SgemmCuBlas) has_sgemm = true;
  }
  if (has_sgemm && !has_cublas) options->kernels.push_back(KernelType::SgemmCuBlas);
  if (has_hgemm && !has_cublas_hgemm) options->kernels.push_back(KernelType::HgemmCuBlasFp16Acc);

  return true;
}

const char *kernel_name(KernelType type) {
  switch (type) {
    case KernelType::SgemmCustom:                 return "sgemm_custom";
    case KernelType::SgemmCutlassLikeS5:      return "sgemm_cutlass_like_s5";
    case KernelType::SgemmCutlassLikeS5OneCta: return "sgemm_cutlass_like_s5_1cta";
    case KernelType::SgemmCutlassLikeS5WarpOrder: return "sgemm_cutlass_like_s5_warporder";
    case KernelType::SgemmCutlassLikeS5Schedule:  return "sgemm_cutlass_like_s5_schedule";
    case KernelType::SgemmCutlassLikeS5CopySchedule: return "sgemm_cutlass_like_s5_copyorder";
    case KernelType::SgemmCutlassLikeS5MmaOrder:  return "sgemm_cutlass_like_s5_mmaorder";
    case KernelType::SgemmCutlassRefS5:       return "sgemm_cutlass_ref_s5";
    case KernelType::SgemmNaive:                  return "sgemm_naive";
    case KernelType::SgemmCuBlas:                 return "sgemm_cublas";
    case KernelType::SgemmExternalDb:   return "sgemm_external_db";
    case KernelType::SgemmExternalNodb: return "sgemm_external_nodb";
    case KernelType::HgemmCute:              return "hgemm_cute";
    case KernelType::HgemmCuteNoreg: return "hgemm_cute_noreg";
    case KernelType::HgemmCutlassSm80:           return "hgemm_cutlass_sm80";
    case KernelType::HgemmSm90Pingpong:      return "hgemm_sm90_pingpong";
    case KernelType::HgemmCutlassSm90Pingpong: return "hgemm_cutlass_sm90_pingpong";
    case KernelType::HgemmCutlassSm90Cooperative: return "hgemm_cutlass_sm90_cooperative";
    case KernelType::HgemmCuBlasFp16Acc:            return "hgemm_cublas_fp16acc";
    case KernelType::HgemmCuBlasFp32Acc:            return "hgemm_cublas_fp32acc";
  }
  return "unknown";
}

bool is_hgemm_kernel(KernelType type) {
  return type == KernelType::HgemmCute ||
         type == KernelType::HgemmCuteNoreg ||
         type == KernelType::HgemmCutlassSm80 ||
         type == KernelType::HgemmSm90Pingpong ||
         type == KernelType::HgemmCutlassSm90Pingpong ||
         type == KernelType::HgemmCutlassSm90Cooperative ||
         type == KernelType::HgemmCuBlasFp16Acc ||
         type == KernelType::HgemmCuBlasFp32Acc;
}
