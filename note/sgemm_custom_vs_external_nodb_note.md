# SGEMM Custom vs External No-DB 实验笔记

## 背景

这次实验对比两个非双缓冲 SGEMM kernel：

- `--custom`: `src/kernels/sgemm_128x128x32.cuh`
- `--external-nodb`: `src/kernels/sgemm_external_128x128x16.cuh` 中的 `external::no_double_buffer::sgemm`

表面差异：

| Kernel | Block tile | K tile | MMA 内部 K 展开 | Static smem |
|---|---:|---:|---:|---:|
| `custom` | 128x128 | 32 | 8 | 32.77 KiB |
| `external-nodb` | 128x128 | 16 | 1 | 16.38 KiB |

两个 kernel 都是 256 threads/block，都是每个线程累计 8 个 `float4` 的 8x8 输出片段。

## 先修正同步问题

`custom` 原先在每个 K tile 里有：

```text
load global -> smem
__syncthreads()
compute from smem
next K tile overwrites smem
```

但 compute 结束后缺少一个 tile 间同步。这样下一轮 K tile 的部分线程可能先写 smem，覆盖其他线程还在读取的上一轮 smem 数据。

已经在 compute loop 结束后补了：

```cpp
__syncthreads();
```

位置：

- [sgemm_128x128x32.cuh](/home/aoi211/LearnCUDA/src/kernels/sgemm_128x128x32.cuh:199)
- [sgemm_128x128x32_trial.cuh](/home/aoi211/LearnCUDA/src/kernels/sgemm_128x128x32_trial.cuh:207)

修正后验证：

```bash
./build/learn_cuda --kernel custom --kernel external-nodb --kernel trial \
  --verify --size 128 --iters 1 --warmup 1
```

三个 kernel 都通过，最大误差约为 `4.4703484e-08`。

## Benchmark

正式比较要串行跑。同一张 GPU 上并行启动两个 bench 会互相抢资源，4096 的时间会被明显拉高。

命令：

```bash
./build/learn_cuda bench --kernel custom --kernel external-nodb \
  --size 4096 --warmup 20 --iters 100

./build/learn_cuda bench --kernel external-nodb --kernel custom \
  --size 4096 --warmup 20 --iters 100
```

结果：

| Order | Kernel | Avg ms | TFLOP/s |
|---|---|---:|---:|
| custom first | `custom` | 21.6071 | 6.3608 |
| custom first | `external-nodb` | 20.9042 | 6.5747 |
| external first | `external-nodb` | 19.9967 | 6.8731 |
| external first | `custom` | 21.6026 | 6.3622 |

观察：

- `external-nodb` 在普通 bench 中稳定更快，幅度约 3% 到 8%。
- `custom` 补上同步后不再有之前那种模糊优势。
- 运行顺序仍然会影响几百分点，所以最终原因要看 ncu。

## Nsight Compute

命令：

```bash
/usr/local/NVIDIA-Nsight-Compute-2025.3/ncu \
  --section LaunchStats \
  --section Occupancy \
  --section SpeedOfLight \
  --section SchedulerStats \
  --section WarpStateStats \
  --print-details all \
  -c 1 \
  ./build/learn_cuda --kernel custom --size 4096

/usr/local/NVIDIA-Nsight-Compute-2025.3/ncu \
  --section LaunchStats \
  --section Occupancy \
  --section SpeedOfLight \
  --section SchedulerStats \
  --section WarpStateStats \
  --print-details all \
  -c 1 \
  ./build/learn_cuda --kernel external-nodb --size 4096
```

注意：ncu 也要串行跑，不能两个 profile 同时打到同一张 GPU 上。

## ncu 结果

### size 4096

| Metric | `custom` | `external-nodb` |
|---|---:|---:|
| Duration | 19.71 ms | 18.81 ms |
| Waves Per SM | 21.33 | 21.33 |
| Registers/thread | 128 | 128 |
| Static smem/block | 32.77 KiB | 16.38 KiB |
| Theoretical occupancy | 33.33% | 33.33% |
| Achieved occupancy | 33.05% | 33.04% |
| Achieved active warps/SM | 15.86 | 15.86 |
| Issue Active | 65.82% | 68.95% |
| Issued Warp/Scheduler | 0.66 | 0.69 |
| Active Warps/Scheduler | 3.96 | 3.96 |
| Eligible Warps/Scheduler | 2.19 | 2.13 |
| Compute Throughput | 65.82% | 68.95% |
| DRAM Throughput | 46.52% | 47.52% |
| L1/TEX Throughput | 65.76% | 62.68% |

Warp stall：

| Stall reason | `custom` | `external-nodb` |
|---|---:|---:|
| Not Selected | 2.30 | 2.07 |
| MIO Throttle | 0.77 | 0.31 |
| Dispatch Stall | 0.63 | 0.55 |
| Barrier | 0.37 | 0.43 |
| Long Scoreboard | 0.35 | 0.53 |
| Short Scoreboard | 0.29 | 0.56 |
| Wait | 0.13 | 0.20 |

### size 2048

补同步后，2048 的 ncu 也呈现同样方向：

| Metric | `custom` | `external-nodb` |
|---|---:|---:|
| Duration | 2.51 ms | 2.44 ms |
| Registers/thread | 128 | 128 |
| Static smem/block | 32.77 KiB | 16.38 KiB |
| Theoretical occupancy | 33.33% | 33.33% |
| Achieved occupancy | 32.21% | 32.13% |
| Issue Active | 64.75% | 66.62% |
| Issued Warp/Scheduler | 0.66 | 0.69 |
| Eligible Warps/Scheduler | 2.17 | 2.07 |

Warp stall：

| Stall reason | `custom` | `external-nodb` |
|---|---:|---:|
| Not Selected | 2.26 | 2.01 |
| MIO Throttle | 0.74 | 0.34 |
| Dispatch Stall | 0.63 | 0.53 |
| Barrier | 0.37 | 0.48 |
| Long Scoreboard | 0.26 | 0.45 |
| Short Scoreboard | 0.28 | 0.56 |
| Wait | 0.13 | 0.20 |

## 结论

`custom` 的 K=8 展开确实在隐藏一部分依赖等待。

证据是它的 scoreboard stall 明显低：

```text
size 4096:
Long Scoreboard:  0.53 -> 0.35
Short Scoreboard: 0.56 -> 0.29
```

这符合代码结构：`custom` 一次从 smem 取 8 个 K slice 到寄存器，再连续做 8 层计算；比 `external-nodb` 每次围绕一个 K slice 计算更容易摊掉 smem load 到 FMA 使用之间的依赖。

但这不是免费收益。`custom` 同时把 shared-memory / MIO 路径压力推高了：

```text
size 4096:
MIO Throttle:    0.31 -> 0.77
Dispatch Stall:  0.55 -> 0.63
Not Selected:    2.07 -> 2.30
Issue Active:    68.95% -> 65.82%
```

也就是说，K=8 的展开把“等数据依赖”的问题换成了“smem/MIO 管线和调度发射更拥挤”的问题。最后 `custom` 虽然 scoreboard 更好，但发射效率更差，总时间反而更慢。

## 为什么不是 occupancy 问题

这组对比里 occupancy 完全一样：

```text
2 blocks/SM
16 warps/SM
theoretical occupancy = 33.33%
achieved occupancy ~= 33%
```

两个 kernel 都是 `128 registers/thread`，最终都被寄存器限制在 2 blocks/SM。`custom` 用 32 KiB smem，`external-nodb` 用 16 KiB smem，但这个差异没有改变 occupancy。

所以这次性能差异不该从 occupancy 解释，而应该从单个 resident block 内部的执行形态解释：

- `custom`: 少一些 scoreboard stall，但更多 MIO throttle / dispatch stall。
- `external-nodb`: scoreboard stall 更高，但 MIO 压力低，整体 issue 更顺。

## 对之前结果的解释

补同步前，`custom` 少了一次 K tile 间 barrier，这会让它看起来更快一点，但那不是合法优化。补上正确同步后：

- barrier stall 上升；
- 原先 K=8 展开带来的优势被 MIO/dispatch 压力抵消；
- `external-nodb` 在 2048 和 4096 的 ncu 里都更快。

当前结论：

**这个 `custom` 的 K=8 方案并不是单调更优。它减少了 smem load 使用依赖，却制造了更大的 MIO/调度压力；补齐必要同步后，外部非双缓冲版整体更均衡。**

## 下一步可以做的实验

如果要继续追，可以做一个中间版本：

- 保持 `custom` 的 128x128x32 大 tile；
- 把 `MMA_K` 从 8 改成 2 或 4；
- 保留正确的 tile-end `__syncthreads()`；
- 对比 scoreboard、MIO throttle、Issue Active。

理想目标不是盲目最大化 K 展开，而是找一个平衡点：scoreboard 降下来，但 MIO throttle 不爆。
