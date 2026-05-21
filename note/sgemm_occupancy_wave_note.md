# SGEMM Occupancy 和 Waves 实验笔记

## 背景

这次实验对比两个 128x128x32 SGEMM kernel：

- `--custom`: 原始 `sgemm_128x128x32`，每个 block 使用约 32 KiB static shared memory。
- `--trial`: 逻辑基本相同，但 launch 时申请 64 KiB dynamic shared memory，只使用前 32 KiB，剩余作为 padding。

实验目的：

- 故意让 `--trial` 每个 SM 只能常驻 1 个 block。
- 观察 occupancy 降低一半后，运行时间是否接近翻倍。

结论先写在前面：

**`--trial` 的 occupancy 确实比 `--custom` 低一半，但运行时间只慢约 7%-10%。原因是 occupancy 不是吞吐率本身；它只是 resident warps 的数量上限。这个 kernel 的 1 block/SM 已经能提供不少 ILP 和可调度 warp，第二个 block/SM 主要只是减少 scheduler 空泡，而不是让计算资源翻倍。**

## Benchmark 结果

机器：

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Compute Capability: 8.9
- Driver: 590.48.01

普通 CUDA event benchmark：

| Size | custom avg_ms | trial avg_ms | trial 慢多少 |
|---:|---:|---:|---:|
| 1024 | 0.3112 | 0.3441 | +10.6% |
| 2048 | 2.39 左右 | 2.55 左右 | +7%-9% |
| 4096 | 20.49 | 22.50 | +9.8% |

参考命令：

```bash
./build/learn_cuda bench --kernel custom --kernel trial --size 1024 --warmup 20 --iters 200
./build/learn_cuda bench --kernel custom --kernel trial --size 2048 --warmup 20 --iters 100
./build/learn_cuda bench --kernel custom --kernel trial --size 4096 --warmup 10 --iters 50
```

注意：不要并行跑两个 bench 进程，否则两个进程会抢同一块 GPU，计时会被污染。

## Nsight Compute 结果

使用的 `ncu` 路径：

```bash
/usr/local/NVIDIA-Nsight-Compute-2025.3/ncu
```

参考命令：

```bash
/usr/local/NVIDIA-Nsight-Compute-2025.3/ncu \
  --section LaunchStats \
  --section Occupancy \
  --section SpeedOfLight \
  --section SchedulerStats \
  -c 1 \
  ./build/learn_cuda --kernel custom --size 2048

/usr/local/NVIDIA-Nsight-Compute-2025.3/ncu \
  --section LaunchStats \
  --section Occupancy \
  --section SpeedOfLight \
  --section SchedulerStats \
  -c 1 \
  ./build/learn_cuda --kernel trial --size 2048
```

### `--custom`

关键数据：

```text
Static Shared Memory Per Block: 32.77 KB
Dynamic Shared Memory Per Block: 0
Registers Per Thread: 128

Block Limit Registers: 2
Block Limit Shared Mem: 3
Block Limit Warps: 6

Theoretical Active Warps per SM: 16
Theoretical Occupancy: 33.33%
Achieved Occupancy: 32.25%

Waves Per SM: 5.33

Issue Slots Busy: ~66%
Issued Warp Per Scheduler: ~0.68
Active Warps Per Scheduler: ~3.86
Eligible Warps Per Scheduler: ~2.29
```

解释：

- `--custom` 每个 block 256 threads，即 8 warps。
- 因为每 thread 128 registers，register 限制最多 2 blocks/SM。
- 所以常驻约 16 warps/SM。
- theoretical occupancy 是 `16 / 48 = 33.33%`。

### `--trial`

关键数据：

```text
Static Shared Memory Per Block: 0
Dynamic Shared Memory Per Block: 65.54 KB
Registers Per Thread: 128

Block Limit Registers: 2
Block Limit Shared Mem: 1
Block Limit Warps: 6

Theoretical Active Warps per SM: 8
Theoretical Occupancy: 16.67%
Achieved Occupancy: 16.67%

Waves Per SM: 10.67

Issue Slots Busy: ~57%
Issued Warp Per Scheduler: ~0.59
Active Warps Per Scheduler: ~2.00
Eligible Warps Per Scheduler: ~1.07
```

解释：

- `--trial` 每个 block 仍然是 256 threads，即 8 warps。
- 但因为每 block 申请约 64 KiB dynamic shared memory，shared memory 限制最多 1 block/SM。
- 所以常驻约 8 warps/SM。
- theoretical occupancy 是 `8 / 48 = 16.67%`。

## 为什么 waves 差一倍，时间没有差一倍

`ncu` 中的 `Waves Per SM` 确实差一倍：

```text
custom: 5.33 waves/SM
trial : 10.67 waves/SM
```

这个数字容易让人直觉上觉得 `trial` 应该慢 2 倍。但这里有个关键点：

```text
custom 的一个 wave:
  每个 SM 同时放 2 个 block

trial 的一个 wave:
  每个 SM 同时放 1 个 block
```

所以 `custom` 的 wave 数少一半，是因为它把两个 block 塞进了同一波里。但这两个 block 并不是在两套独立硬件上执行，它们仍然共享同一个 SM 的资源：

- warp scheduler
- FMA pipeline
- load/store pipeline
- shared memory pipeline
- L1/TEX

因此，`custom` 的一个 wave 虽然包含两倍 block 工作量，但不会免费获得两倍执行资源。第二个 block 的主要作用是提供更多 resident warps，让 scheduler 更容易在某些 warp 等待时找到另一个 ready warp。

换句话说：

```text
waves 决定 block 分几批进 SM
occupancy 决定每批里有多少 resident warp
真正决定时间的是每周期实际发出了多少有用指令
```

## 真正的差异在 scheduler 利用率

最关键的 `ncu` 数据是：

```text
trial  Issued Warp Per Scheduler: ~0.59
custom Issued Warp Per Scheduler: ~0.68
```

occupancy 从 16.67% 到 33.33%，是 2 倍。

但 scheduler 实际发射效率只是：

```text
0.68 / 0.59 = 1.15x
```

也就是说，多出来的第二个 block/SM 没有把 SM 吞吐翻倍，只是把 issue slot busy 从约 57% 提升到约 66%。

这和 benchmark 结果一致：`custom` 比 `trial` 快一小截，但不是 2 倍。

## 为什么 1 block/SM 也能跑得不差

这个 kernel 有几个特点，让它对 occupancy 没有线性敏感：

1. 每个 block 已经有 8 个 warps。

   `--trial` 虽然只有 1 block/SM，但仍然有 8 warps/SM。对很多 compute-heavy kernel 来说，这已经不是“完全没东西可调度”的状态。

2. 每个 thread 有很多独立 accumulator。

   代码中每个 thread 有：

   ```cpp
   float tCrC[8][8] = {};
   ```

   这意味着单个 warp 内部有较强 ILP。FMA latency 不完全依赖更多 resident warps 来隐藏，同一个 warp 内也有很多独立指令可以排。

3. 这个 SGEMM tile 的算术密度高。

   每个 128x128x32 tile 做大量 FMA，而 global memory load 相对不大。`ncu` 中 DRAM throughput 大约只有 8%-11%，说明它不是典型“高 occupancy 用来隐藏 global memory latency”的 kernel。

4. 第二个 block/SM 主要改善 latency hiding，不是增加硬件算力。

   `--custom` 的第二个 block 让 active warps 增多：

   ```text
   trial  Active Warps Per Scheduler: ~2.00
   custom Active Warps Per Scheduler: ~3.86
   ```

   但 eligible warps 和 issued warps 没有同比例增加：

   ```text
   trial  Eligible Warps Per Scheduler: ~1.07
   custom Eligible Warps Per Scheduler: ~2.29

   trial  Issued Warp Per Scheduler: ~0.59
   custom Issued Warp Per Scheduler: ~0.68
   ```

   多出来的 warp 有帮助，但帮助是减少空泡，不是把每周期发射数翻倍。

## 一个更准确的心智模型

不要把 occupancy 理解成：

```text
occupancy 高 2 倍 => 性能高 2 倍
```

更准确的是：

```text
occupancy 高 2 倍
=> scheduler 有更多候选 warp
=> latency hiding 可能更好
=> issue slot busy 可能上升
=> 如果原来已经比较忙，收益就会很小
```

本次实验里：

```text
resident warps: 8 -> 16       2.00x
theoretical occupancy: 16.7 -> 33.3%
issued warp/scheduler: 0.59 -> 0.68  约 1.15x
benchmark runtime: 只改善约 7%-10%
```

所以 `--trial` 的低 occupancy 确实有损失，但损失主要体现为 scheduler 空泡变多一点，而不是工作量翻倍。

## 最终结论

这个实验说明：

**对这个 128x128x32 SGEMM kernel 来说，1 block/SM 的 16.67% occupancy 已经可以提供相当多的有效执行。2 blocks/SM 的 33.33% occupancy 会更好，但主要是提升 scheduler ready warp 数量，把 issue slot busy 从约 57% 拉到约 66%，所以实际时间只差约 7%-10%，不会因为 waves/SM 翻倍而让 runtime 翻倍。**

