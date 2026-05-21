# SGEMM Double Buffer 实验笔记

## 背景

这次实验对比 `ref/sgemm.txt` 里的两个外部 SGEMM kernel：

- `external::double_buffer::sgemm`: shared memory 双缓冲版本。
- `external::no_double_buffer::sgemm`: shared memory 单缓冲版本。

两个 kernel 都是 128x128x16 tiling：

```text
BM = 128
BN = 128
bK = 16
threads_per_block = 256
```

工程里新增了两个命令行入口：

```bash
./build/learn_cuda --external-db
./build/learn_cuda --external-nodb
```

也可以用：

```bash
./build/learn_cuda --kernel external-db
./build/learn_cuda --kernel external-nodb
```

## 实验问题

直觉上，双缓冲应该隐藏 global memory load latency：

```text
加载下一块 K tile
    和
计算当前 K tile
```

发生重叠，从而提升性能。

但实际 bench 发现提升很有限，有时候甚至因为运行顺序、boost、温度出现反转。因此用 Nsight Compute 检查它到底隐藏了什么 latency，又付出了什么代价。

## 资源使用

ptxas 资源：

| Kernel | Registers | Static smem |
|---|---:|---:|
| `external::double_buffer::sgemm` | 118 | 32768 bytes |
| `external::no_double_buffer::sgemm` | 128 | 16384 bytes |

解释：

- 双缓冲版本用两份 A/B shared memory stage，所以 smem 是 32 KiB。
- 单缓冲版本只用一份 A/B shared memory stage，所以 smem 是 16 KiB。
- 双缓冲版本 register 反而略低，可能是编译器调度和变量生命周期不同导致。

虽然 smem 不同，但在当前 GPU 上两者 occupancy 一样：

```text
2 blocks/SM
16 warps/SM
theoretical occupancy = 33.33%
```

也就是说，双缓冲多用的 shared memory 没有降低 occupancy；单缓冲少用的 shared memory 也没有提高 occupancy，因为两者最终都被 register 限制在 2 blocks/SM。

## Benchmark 结果

普通 CUDA event benchmark 有明显波动，尤其是长时间跑 4096 时，boost/温度/运行顺序会影响几百分点。

一些有效观测：

```text
size 1024:
external_db     0.3157 ms
external_nodb   0.3181 ms

size 2048:
external_db     2.3507 ms
external_nodb   2.3083 ms

size 4096:
external_db     19.2393 ms
external_nodb   20.1908 ms
```

结论：

**普通 bench 看不出稳定的大幅提升。双缓冲的收益大概只有几个百分点，很容易被 GPU boost 和运行顺序噪声淹没。**

## Nsight Compute 结果

使用命令：

```bash
/usr/local/NVIDIA-Nsight-Compute-2025.3/ncu \
  --section LaunchStats \
  --section Occupancy \
  --section SpeedOfLight \
  --section SchedulerStats \
  --section WarpStateStats \
  --section ComputeWorkloadAnalysis \
  --section MemoryWorkloadAnalysis \
  --section InstructionStats \
  --print-details all \
  -c 1 \
  ./build/learn_cuda --kernel external-db --size 2048

/usr/local/NVIDIA-Nsight-Compute-2025.3/ncu \
  --section LaunchStats \
  --section Occupancy \
  --section SpeedOfLight \
  --section SchedulerStats \
  --section WarpStateStats \
  --section ComputeWorkloadAnalysis \
  --section MemoryWorkloadAnalysis \
  --section InstructionStats \
  --print-details all \
  -c 1 \
  ./build/learn_cuda --kernel external-nodb --size 2048
```

### size 2048

| Metric | double buffer | no double buffer |
|---|---:|---:|
| Duration | 2.39 ms | 2.44 ms |
| Registers/thread | 118 | 128 |
| Static smem/block | 32.77 KiB | 16.38 KiB |
| Theoretical occupancy | 33.33% | 33.33% |
| Achieved occupancy | 32.13% | 32.14% |
| Issue Slots Busy | 68.26% | 66.65% |
| Issued Warp Per Scheduler | 0.70 | 0.69 |
| Active Warps Per Scheduler | 3.86 | 3.90 |
| Eligible Warps Per Scheduler | 2.17 | 2.07 |
| Instructions Executed | 299.99M | 298.96M |
| DRAM Throughput | 8.39% | 9.20% |
| Compute Throughput | 68.26% | 66.65% |

Warp stall：

| Stall reason | double buffer | no double buffer |
|---|---:|---:|
| Not Selected | 2.07 | 2.01 |
| Short Scoreboard | 0.83 | 0.56 |
| Dispatch Stall | 0.46 | 0.53 |
| Long Scoreboard | 0.32 | 0.45 |
| Barrier | 0.27 | 0.48 |
| MIO Throttle | 0.27 | 0.34 |

### size 4096

轻量 ncu section 下：

| Metric | double buffer | no double buffer |
|---|---:|---:|
| Duration | 18.48 ms | 18.82 ms |
| Issue Slots Busy | 70.44% | 68.94% |
| Issued Warp Per Scheduler | 0.71 | 0.69 |
| Eligible Warps Per Scheduler | 2.21 | 2.13 |
| DRAM Throughput | 47.43% | 47.45% |
| Compute Throughput | 70.44% | 68.94% |

Warp stall：

| Stall reason | double buffer | no double buffer |
|---|---:|---:|
| Not Selected | 2.11 | 2.07 |
| Short Scoreboard | 0.83 | 0.56 |
| Dispatch Stall | 0.47 | 0.55 |
| Long Scoreboard | 0.42 | 0.53 |
| Barrier | 0.28 | 0.43 |
| MIO Throttle | 0.25 | 0.31 |

## 双缓冲确实生效了吗

生效了。

最直接证据是双缓冲版本降低了：

```text
Long Scoreboard
Barrier
MIO Throttle
Dispatch Stall
```

以 size 2048 为例：

```text
Long Scoreboard: 0.45 -> 0.32
Barrier:         0.48 -> 0.27
MIO Throttle:    0.34 -> 0.27
Dispatch Stall:  0.53 -> 0.46
```

这说明双缓冲确实把一部分 global memory 等待和同步等待藏进了计算阶段。

代码结构上也对应：

- 双缓冲版在主循环开头先把下一块 tile load 到 register:

  [sgemm_external_128x128x16.cuh](/home/aoi211/LearnCUDA/src/kernels/sgemm_external_128x128x16.cuh:86)

  ```cpp
  for (int k = 1; k < k_iter; ++k) {
      // load next tile to ldg_a / ldg_b
      ...
      // compute current tile
      ...
      // write loaded tile to next smem stage
      ...
  }
  ```

- 单缓冲版每个 K tile 都是 load 到 smem，sync，然后 compute:

  [sgemm_external_128x128x16.cuh](/home/aoi211/LearnCUDA/src/kernels/sgemm_external_128x128x16.cuh:291)

  ```cpp
  for (int k = 0; k < k_iter; ++k) {
      // load current tile to smem
      __syncthreads();
      // compute current tile
      __syncthreads();
  }
  ```

## 为什么收益仍然很有限

### 1. 原本就不是严重 global memory latency bound

每个 128x128x16 tile 的 global load 量：

```text
A: 128 * 16 floats
B: 16 * 128 floats
total = 4096 floats = 16 KiB
```

每个 tile 的计算量：

```text
128 * 128 * 16 FMA
```

算术密度很高。

size 2048 时，ncu 里 DRAM throughput 只有：

```text
double buffer:    8.39%
no double buffer: 9.20%
```

这说明当前规模下 global memory bandwidth/latency 不是压倒性的瓶颈。双缓冲能隐藏 global latency，但能隐藏的部分本来就不大。

size 4096 时 DRAM throughput 上升到约 47%，但 compute throughput 仍然更高：

```text
double buffer:
  DRAM Throughput:    47.43%
  Compute Throughput: 70.44%

no double buffer:
  DRAM Throughput:    47.45%
  Compute Throughput: 68.94%
```

因此大尺寸下也不是纯 memory-bound。

### 2. 两者 occupancy 一样

双缓冲没有带来更多 resident warps：

```text
double buffer:
  118 registers/thread
  32.77 KiB smem/block
  occupancy 33.33%

no double buffer:
  128 registers/thread
  16.38 KiB smem/block
  occupancy 33.33%
```

所以双缓冲的收益只能来自更好的 pipeline overlap，而不是 occupancy 提升。

### 3. 双缓冲增加了 short scoreboard

双缓冲版本减少了 long scoreboard，但 short scoreboard 变高：

```text
size 2048:
Short Scoreboard: 0.56 -> 0.83

size 4096:
Short Scoreboard: 0.56 -> 0.83
```

这通常对应较短 latency 的 shared memory / register dependency。

也就是说，双缓冲把一部分 global memory 等待藏掉了，但引入了更多 smem/register staging 的短依赖。收益被抵消了一部分。

### 4. 指令数略增

size 2048：

```text
double buffer:    299.99M instructions
no double buffer: 298.96M instructions
```

双缓冲多了约 1M 条 warp-level 指令。这个差距不大，但当总体收益只有几个百分点时，它也会吃掉一点提升。

## 最终结论

这次实验说明：

**双缓冲不是没有生效。它确实降低了 Long Scoreboard 和 Barrier stall，让 Issue Slots Busy 从约 66.65% 提到约 68.26%。但这个 kernel 本身算术密度高、occupancy 已经相同、global memory latency 不是主要瓶颈；同时双缓冲引入了更多 Short Scoreboard 和少量额外指令。因此最终净收益只有约 2%，普通 bench 中很容易被 GPU boost、温度和运行顺序噪声淹没。**

一个更准确的心智模型：

```text
双缓冲
=> 隐藏一部分 global load / barrier latency
=> 但增加 smem/register staging 依赖和指令
=> 如果原 kernel 不是 memory-latency-bound
=> 净收益会很小
```

