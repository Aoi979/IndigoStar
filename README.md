# LearnCUDA — CUDA GEMM 手写 Kernel 学习与 Benchmark

本项目是一个**从 naive 到 Hopper 的 CUDA GEMM 手写 kernel 集合**，覆盖 FP32（SGEMM）和 FP16（HGEMM）两种精度，支持横向性能对比与正确性验证。

核心目标：
- **学习**：理解 CUTLASS、CuTe、TMA/GMMA 等底层优化技术
- **实验**：快速迭代新的 tile 策略、调度策略、内存布局
- **对比**：统一 benchmark 框架，一键对比多个 kernel 的 TFLOP/s

---

## 快速开始

```bash
cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build .
```

> 若目标 GPU 为 Hopper（SM90），请将架构改为 `90a`：
> ```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90a
```

CMake 会根据 `CMAKE_CUDA_ARCHITECTURES` **自动裁剪编译的 kernel 集合**：
- SM80 / SM89：只编译 SM80 目录下的算子
- SM90 / SM90a：只编译 SM90 目录下的算子
- 同时指定 `80,90a`：两者都编译

### 横向 Benchmark

指定要测的算子，程序**自动加入对应的 cuBLAS 基线**：

```bash
# 只传手写算子，自动补上 cublas
./learn_cuda bench --kernel sgemm-custom --kernel sgemm-naive --size 1024

# 同时测 SGEMM 和 HGEMM，自动补上 cublas + cublas-hgemm
./learn_cuda bench --kernel sgemm-custom --kernel hgemm-cute --size 1024

# 显式传了 cublas 则不再重复添加
./learn_cuda bench --kernel sgemm-custom --kernel sgemm-cublas --size 1024
```

### 正确性验证

```bash
./learn_cuda --kernel sgemm-custom --size 256 --verify
```

更多用法详见 [USAGE.md](USAGE.md)。

---

## 目录结构

```
├── CMakeLists.txt
├── README.md
├── USAGE.md
├── scripts/
│   └── gen_kernel_report.py          # 编译后自动生成 kernel 资源报告
├── cutlass/                            # CUTLASS submodule
├── note/                               # 学习笔记（与代码演进一一对应）
├── src/
│   ├── main.cu                         # 入口：参数解析、资源分配、主流程编排
│   ├── core/                           # 核心基础设施（从原 main.cu 拆分）
│   │   ├── cli.h / cli.cu              # CLI 解析、KernelType 枚举
│   │   ├── benchmark.h / benchmark.cu  # Kernel 启动包装器 + Benchmark 逻辑
│   │   ├── verify.h                    # CPU 参考结果逐元素对比
│   │   ├── data_utils.h                # 随机数据生成、参考计算
│   │   └── device_utils.h              # CUDA RAII 包装器、设备查询
│   └── kernels/                        # 全部 Kernel 实现
│       ├── common.hpp                  # 公共宏（cp.async、FETCH_FLOAT4 等）
│       ├── reference/                  # 参考实现（不限定架构）
│       │   └── sgemm_naive.cuh
│       ├── sm80/                       # Ampere 架构 Kernel
│       │   ├── sgemm/                  #   FP32 SGEMM
│       │   │   ├── handwritten/        #     手写算子
│       │   │   │   ├── custom_128x128x32.cuh
│       │   │   │   ├── double_buffer_dev_128x128x32.cuh
│       │   │   │   ├── external_128x128x16.cuh
│       │   │   │   └── cutlass_like_stage5.cuh
│       │   │   └── cutlass/            #     直接调 CUTLASS 库
│       │   │       └── ref_stage5.cuh
│       │   └── hgemm/                  #   FP16 HGEMM
│       │       ├── handwritten/        #     手写算子
│       │       │   ├── cute_128x128_nn.cuh
│       │       │   ├── cute_128x128_nn_no_reg_prefetch.cuh
│       │       │   ├── cute_ampere_16816.cuh
│       │       │   ├── cute_ampere_16816_no_reg_prefetch.cuh
│       │       │   └── sm80_hgemm.cuh
│       │       └── cutlass/            #     直接调 CUTLASS 库
│       │           └── tensorop.cuh
│       └── sm90/                       # Hopper 架构 Kernel
│           ├── cluster.cuh             #   Cluster 同步原语
│           └── hgemm/                  #   FP16 HGEMM (TMA + GMMA)
│               ├── handwritten/        #     手写算子
│               │   ├── pingpong.cuh
│               │   └── bf16_ref.cuh
│               └── cutlass/            #     直接调 CUTLASS 库
│                   ├── sm90_hgemm.cuh
│                   └── cooperative.cuh
```

---

## 学习笔记索引

| 笔记 | 对应代码/主题 |
|------|--------------|
| [cutlass_stage5_alignment_report.md](note/cutlass_stage5_alignment_report.md) | `sm80/sgemm/cutlass_like_stage5.cuh` 内存对齐分析 |
| [sgemm_custom_vs_external_nodb_note.md](note/sgemm_custom_vs_external_nodb_note.md) | custom vs external-nodb 性能差异分析 |
| [sgemm_double_buffer_note.md](note/sgemm_double_buffer_note.md) | 共享内存双缓冲原理与实现 |
| [sgemm_external_nodb_fast_reason_note.md](note/sgemm_external_nodb_fast_reason_note.md) | external-nodb 为什么比 custom 快 |
| [sgemm_occupancy_wave_note.md](note/sgemm_occupancy_wave_note.md) | Occupancy 与 Wave 量化效应 |
| [sgemm_simt_tensorcore_optimization_lesson.md](note/sgemm_simt_tensorcore_optimization_lesson.md) | SIMT -> TensorCore 演进路线 |
| [cute_hgemm_register_prefetch_short_dependency_note.md](note/cute_hgemm_register_prefetch_short_dependency_note.md) | CuTe HGEMM register prefetch 与短依赖链 |

---

## 如何添加新 Kernel

1. **在 `src/kernels/<arch>/<precision>/handwritten/` 下实现 kernel**，函数签名建议统一为：
   ```cuda
   __global__ void my_kernel(int M, int N, int K,
                             float const *__restrict__ A,
                             float const *__restrict__ B,
                             float *__restrict__ C);
   ```

2. **在 `src/core/benchmark.cu` 中注册**：
   - 添加 `launch_mykernel()` 包装函数
   - 在 `select_launcher()` 或 `select_half_launcher()` 中加入 switch case
   - 如果属于特定架构，用 `#if ENABLE_SM80_KERNELS` 等宏包裹

3. **在 `src/core/cli.cu` 中注册**：
   - 在 `KernelType` 枚举末尾添加新值（如 `SgemmMyKernel`）
   - 在 `kernel_name()` 中添加字符串名称（如 `"sgemm_mykernel"`）
   - 在 `parse_args()` 中支持新的 `--kernel sgemm-mykernel` 值

4. **重新编译**即可通过 `--kernel sgemm-mykernel` 调用。

> 新 kernel 只要实现正确，无需修改 benchmark、verify 或 CLI 逻辑即可自动享受全套基础设施。
