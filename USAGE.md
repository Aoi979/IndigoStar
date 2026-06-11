# Indigo Star 使用文档

CUDA GEMM 手写 kernel 学习与 benchmark 项目，支持 SGEMM（FP32）和 HGEMM（FP16）两种精度，覆盖从 naive 到 Hopper 的多级优化实现。

---

## 一键脚本（推荐）

项目提供三个自动检测 GPU 架构的封装脚本，无需手动 cmake：

### `scripts/build.sh` — 自动编译

自动检测当前 GPU 的 compute capability，清理旧 build，配置 cmake 并编译：

```bash
./scripts/build.sh
```

输出示例：
```
Detected GPU compute capability: 89
Machine:    RTX 4060 (Ada, SM89)
CMake arch: 89
...
Build complete: .../build/indigo_star
```

| 机器 | 检测值 | CMake 架构 | 编译范围 |
|------|--------|-----------|---------|
| RTX 4060 | `89` | `89` | SM80 算子（Ada 兼容 Ampere） |
| A100 | `80` | `80` | SM80 算子 |
| H100 | `90` | `90a` | SM90 算子 |

### `scripts/bench.sh` — 全量 Benchmark

自动检测架构，运行当前机器上**全部可用 kernel** 的 benchmark，并自动加入 cuBLAS 基线：

```bash
# 默认参数: size=1024, iters=100, warmup=10
./scripts/bench.sh

# 自定义参数: size=2048, iters=200, warmup=20
./scripts/bench.sh 2048 200 20
```

**4060 / A100 跑的 kernel：**
- SGEMM: `sgemm-custom`, `sgemm-naive`, `sgemm-external-db`, `sgemm-external-nodb`, `sgemm-cutlass-like-s5`（及变体）, `sgemm-cutlass-ref-s5`
- HGEMM: `hgemm-cute`, `hgemm-cute-noreg`, `hgemm-cutlass-sm80`
- 基线: 自动加 `sgemm-cublas` + `hgemm-cublas-fp16acc`，显式加 `hgemm-cublas-fp32acc`

**H100 跑的 kernel：**
- HGEMM: `hgemm-sm90-pingpong`, `hgemm-cutlass-sm90-pp`, `hgemm-cutlass-sm90-coop`
- 基线: 自动加 `hgemm-cublas-fp16acc`，显式加 `hgemm-cublas-fp32acc`

### `scripts/verify.sh` — 全量精度验证

自动检测架构，运行当前机器上全部可用 kernel 的**正确性验证**（和 CPU 双精度参考结果对比）：

```bash
# 默认 size=512
./scripts/verify.sh

# 自定义尺寸
./scripts/verify.sh 1024
```

> 验证通过的 kernel 会输出 `verify max_abs_error=... max_rel_error=...`；若失败则报错退出。

---

## 手动构建

如需手动 cmake：

```bash
cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build .
```

可执行文件位于 `build/indigo_star`。

如果在不同 GPU 上切换测试，建议重新指定 CUDA 架构，避免用错 SASS：

```bash
# A100
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=80

# Ada / RTX 40 系列
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89

# Hopper H100
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=90a
```

> CMake 会根据 `CMAKE_CUDA_ARCHITECTURES` **自动裁剪编译的 kernel 集合**：
> - SM80/SM89：只编译 `sm80/` 下的算子
> - SM90：只编译 `sm90/` 下的算子
> - 同时指定多个架构（如 `80;90a`）：两者都编译

---

## 三种典型使用方式

### 1. Benchmark 横向对比多个 Kernel

指定要测的算子，程序**自动加入对应的 cuBLAS 基线**，最后以 cuBLAS 为 baseline 输出相对加速比。

```bash
# 只传手写算子，自动补上 sgemm_cublas
./indigo_star bench --kernel sgemm-custom --kernel sgemm-naive --size 1024

# 同时测 SGEMM + HGEMM，自动补上 sgemm_cublas + hgemm_cublas_fp16acc
./indigo_star bench --kernel sgemm-custom --kernel hgemm-cute --size 1024

# 旧式 legacy flags（向后兼容）
./indigo_star bench --naive --cublas --size 1024
```

输出示例：

```
sgemm_custom   M=1024 N=1024 K=1024 warmup=10 iters=100 avg_ms=0.3661 TFLOP/s=5.8663
sgemm_custom C[0]=0.00732559 checksum=-1.40202135
sgemm_naive    M=1024 N=1024 K=1024 warmup=10 iters=100 avg_ms=2.7076 TFLOP/s=0.7931
sgemm_naive C[0]=0.00732559 checksum=-1.40202135
sgemm_cublas   M=1024 N=1024 K=1024 warmup=10 iters=100 avg_ms=0.2817 TFLOP/s=7.6232
sgemm_cublas C[0]=0.00732559 checksum=-1.40203089

--- Relative speedup vs sgemm_cublas ---
sgemm_custom   0.77x
sgemm_naive    0.10x
sgemm_cublas   1.00x
```

### 2. 正确性验证（不开 Benchmark）

只跑一次 kernel，把结果拷贝回 Host，和 CPU 端双精度参考结果逐元素对比。

```bash
./indigo_star --kernel sgemm-custom --size 256 --verify
```

输出示例：

```
sgemm_custom C[0]=0.012853746 checksum=-1.4290881
verify max_abs_error=5.9604645e-08 max_rel_error=5.9604645e-08
```

> 也可以一次验证多个 kernel：
> ```bash
> ./indigo_star --kernel sgemm-custom --kernel sgemm-naive --size 256 --verify
> ```

### 3. 纯跑一次（不 Benchmark、不 Verify）

仅执行一次 kernel，输出 `C[0]` 和 checksum。

```bash
./indigo_star --kernel sgemm-external-nodb --size 256
```

> 不指定 `--kernel` 时，默认跑 `sgemm-custom`。

---

## 命令行参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `bench` / `--bench` / `--benchmark` | 开启 benchmark 模式 | `bench` |
| `--kernel <type>` | 指定要运行的 kernel（可多次指定） | `--kernel sgemm-naive --kernel sgemm-cublas` |
| `--naive` | 等价于 `--kernel sgemm-naive`（向后兼容） | `--naive` |
| `--cublas` | 等价于 `--kernel sgemm-cublas`（向后兼容） | `--cublas` |
| `--external-db` | 等价于 `--kernel sgemm-external-db`（向后兼容） | `--external-db` |
| `--external-nodb` | 等价于 `--kernel sgemm-external-nodb`（向后兼容） | `--external-nodb` |
| `--size <N>` | 同时设置 M=N=K | `--size 1024` |
| `--m <M>` | 设置 M 维度 | `--m 1024` |
| `--n <N>` | 设置 N 维度 | `--n 1024` |
| `--k <K>` | 设置 K 维度 | `--k 1024` |
| `--warmup <N>` | benchmark 前 warmup 次数 | `--warmup 10` |
| `--iters <N>` / `--iterations <N>` | benchmark 迭代次数 | `--iters 100` |
| `--seed <N>` | 随机数种子 | `--seed 42` |
| `--verify` | 和 CPU 参考结果对比正确性 | `--verify` |
| `-h` / `--help` | 显示帮助 | `--help` |

---

## 可用 Kernel

| Kernel | 层次 | 架构 | 说明 |
|--------|------|------|------|
| `sgemm-naive` | 手写 | 通用 | 最简全局内存 SGEMM，功能对照 |
| `sgemm-custom` | 手写 | SM80 | `sgemm_128x128x32`，共享内存 + float4 向量化 |
| `sgemm-external-db` | 手写 | SM80 | 外部 128x128x16 SGEMM，带共享内存双缓冲 |
| `sgemm-external-nodb` | 手写 | SM80 | 外部 128x128x16 SGEMM，不带共享内存双缓冲 |
| `sgemm-cutlass-like-s5` | 手写模仿 | SM80 | 手写 CUTLASS-like 128x128x8 stage-5 SGEMM |
| `sgemm-cutlass-like-s5-1cta` | 手写模仿 | SM80 | 同上 + 1 CTA/SM extra smem |
| `sgemm-cutlass-like-s5-warporder` | 手写模仿 | SM80 | 同上 + CUTLASS warp tile order |
| `sgemm-cutlass-like-s5-schedule` | 手写模仿 | SM80 | 同上 + CUTLASS-like copy schedule |
| `sgemm-cutlass-like-s5-copyorder` | 手写模仿 | SM80 | 同上 + copy order only |
| `sgemm-cutlass-like-s5-mmaorder` | 手写模仿 | SM80 | 同上 + SM80 thread-level FFMA order |
| `sgemm-cutlass-ref-s5` | CUTLASS 库 | SM80 | CUTLASS SM80 SIMT 128x128x8 stage-5 |
| `hgemm-cute` | 手写 | SM80 | CuTe 手写 128x128x64 HGEMM |
| `hgemm-cute-noreg` | 手写 | SM80 | CuTe HGEMM，不做 register prefetch |
| `hgemm-cutlass-sm80` | CUTLASS 库 | SM80 | CUTLASS SM80 TensorOp HGEMM |
| `hgemm-sm90-pingpong` | 手写 | SM90 | SM90 TMA + GMMA persistent pingpong HGEMM |
| `hgemm-cutlass-sm90-pp` | CUTLASS 库 | SM90 | CUTLASS SM90 pingpong schedule |
| `hgemm-cutlass-sm90-coop` | CUTLASS 库 | SM90 | CUTLASS SM90 cooperative schedule |
| `sgemm-cublas` | 基线 | 通用 | cuBLAS SGEMM 参考 |
| `hgemm-cublas-fp16acc` | 基线 | 通用 | cuBLAS HGEMM 参考（fp16 累加器） |
| `hgemm-cublas-fp32acc` | 基线 | 通用 | cuBLAS HGEMM 参考（fp32 累加器） |

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
