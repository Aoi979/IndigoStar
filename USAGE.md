# LearnCUDA 使用文档

SGEMM (Single-precision General Matrix Multiply) 的 CUDA 手写 kernel 集合，支持 benchmark、正确性验证和 kernel 横向对比。

## 构建

```bash
cd build
cmake ..
cmake --build .
```

可执行文件位于 `build/learn_cuda`。

---

## 三种典型使用方式

### 1. Benchmark 横向对比多个 Kernel

同时跑多个 kernel，统计各自的 TFLOP/s，最后以 **cuBLAS** 为 baseline 输出相对加速比（如果没有 cuBLAS，则以第一个 kernel 为 baseline）。

```bash
# 对比 custom、naive、cublas
./learn_cuda bench --kernel custom --kernel naive --kernel cublas --size 1024

# 旧式写法（向后兼容）
./learn_cuda bench --naive --cublas --size 1024
```

输出示例：

```
custom       M=1024 N=1024 K=1024 warmup=10 iters=100 avg_ms=1.2345 TFLOP/s=1.7300
naive        M=1024 N=1024 K=1024 warmup=10 iters=100 avg_ms=5.6789 TFLOP/s=0.3760
cublas       M=1024 N=1024 K=1024 warmup=10 iters=100 avg_ms=0.5678 TFLOP/s=3.7620

--- Relative speedup vs cublas ---
custom      0.46x
naive       0.10x
cublas      1.00x
```

### 2. 正确性验证（不开 Benchmark）

只跑一次 kernel，把结果拷贝回 Host，和 CPU 端双精度参考结果逐元素对比。适用于调试新 kernel 或改完代码后快速回归测试。

```bash
./learn_cuda --kernel custom --size 256 --verify
```

输出示例：

```
custom C[0]=0.01285375 checksum=-1.42908808
verify max_abs_error=5.9604645e-08 max_rel_error=5.9604645e-08
```

> 也可以一次验证多个 kernel：
> ```bash
> ./learn_cuda --kernel custom --kernel naive --size 256 --verify
> ```

### 3. 纯跑一次（不 Benchmark、不 Verify）

仅执行一次 kernel，输出 `C[0]` 和 checksum。适用于快速确认 kernel 能正常启动、不挂死。

```bash
./learn_cuda --kernel double-buffer --size 256
```

输出示例：

```
double_buffer C[0]=0.01285375 checksum=-1.42908808
```

> 不指定 `--kernel` 时，默认跑 `custom`。

---

## 命令行参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `bench` / `--bench` / `--benchmark` | 开启 benchmark 模式 | `bench` |
| `--kernel <type>` | 指定要运行的 kernel（可多次指定） | `--kernel naive --kernel cublas` |
| `--naive` | 等价于 `--kernel naive`（向后兼容） | `--naive` |
| `--cublas` | 等价于 `--kernel cublas`（向后兼容） | `--cublas` |
| `--double-buffer` | 等价于 `--kernel double-buffer`（向后兼容） | `--double-buffer` |
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

| Kernel | 说明 |
|--------|------|
| `custom` | `sgemm_128x128x32`，手写共享内存 + float4 向量化 |
| `naive` | 最简全局内存实现，用于功能对照 |
| `cublas` | NVIDIA cuBLAS 参考实现 |
| `double-buffer` | `sgemm_128x128x32` 的双缓冲版本，使用动态共享内存 |

---

## 如何添加新 Kernel

1. **在 `src/sgemm.cuh` 中实现 kernel**，函数签名建议统一为：
   ```cuda
   __global__ void my_sgemm(int M, int N, int K,
                            float const *__restrict__ A,
                            float const *__restrict__ B,
                            float *__restrict__ C);
   ```

2. **在 `src/main.cu` 中注册**：
   - 在 `enum class KernelType` 末尾添加新枚举值，例如 `MyKernel`
   - 在 `kernel_name()` 中添加对应的字符串名称
   - 编写 `launch_mykernel()` 包装函数
   - 在 `select_launcher()` 中加入 switch case

3. **重新编译**即可通过 `--kernel mykernel` 调用。

> 新 kernel 只要实现正确，无需修改 benchmark、verify 或 CLI 逻辑即可自动享受全套基础设施。
