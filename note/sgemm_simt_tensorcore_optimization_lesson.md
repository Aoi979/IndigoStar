# SIMT GEMM 优化踩坑记：为什么 "粗制滥造" 打败了精心优化

> **Date**: 2026-05-29  
> **Device**: RTX 4060 Laptop (SM89, 24 SMs)  
> **NCU Report**: `ncu_temp/temp-f.ncu-rep`（锁频 base clock，以本报告为准）  
> **Kernels**: `external::double_buffer::sgemm` vs `cutlass_like::sgemm_128x128x8stage5_kernel`  
> **Status**: 根因已找到，心结已解

---

## 1. 问题的来龙去脉

### 1.1 现象

- **"粗制滥造" 版本**: `src/kernels/sgemm_external_128x128x16.cuh`
  - 手写 SIMT double buffer，无 `cp.async`
  - `LDG.128` + `STS` 经典路径
  - `bK = 16`
  - 无 block swizzle，无软件流水线

- **"精心优化" 版本**: `src/kernels/cutlass_like_sgemm_128x128x8stage5.cuh`
  - 模仿 CUTLASS 的 multistage pipeline
  - `cp.async.ca.shared.global.L2::128B` (5-stage)
  - `kCtaK = 8`
  - block swizzle = 8
  - 多种调度策略 (warp order, mainloop schedule, mma order)

**NCU 锁频性能结果 (4096×4096×4096，RTX 4060)**：

| Kernel | Duration | Elapsed Cycles | SM Throughput | Memory Throughput |
|--------|----------|----------------|---------------|-------------------|
| external_db (粗制滥造) | **16.87 ms** | 32,380,906 | 76.76% | 69.56% |
| cutlass_stage5 (精心优化) | **17.28 ms** | 33,165,808 | **84.13%** | 67.63% |

**cutlass 慢了约 2.4%。** 在 A100 上，external_db 甚至略快或持平。

### 1.2 困惑

花了大量时间优化的版本，核心思路是：
1. 用 `cp.async` 隐藏 global memory latency
2. 5-stage pipeline 最大化内存与计算重叠
3. 模仿 CUTLASS 的 warp schedule 和指令排序

结果性能不如/持平一个连 `cp.async` 都没用的早期版本。而且 NCU 报告显示 **cutlass 的 L2 Hit Rate 高达 95.49%，external 只有 48.44%** —— 明明缓存命中率更高，为什么反而更慢？

---

## 2. NCU 报告核心数据对比（以 temp-f.ncu-rep 为准）

### 2.1 执行时间 & 吞吐

| Metric | external_db | cutlass_stage5 | 差距 |
|--------|-------------|----------------|------|
| **Duration** | **16.87 ms** | 17.28 ms | +2.4% |
| **Elapsed Cycles** | **32,380,906** | 33,165,808 | +785K (+2.4%) |
| **SM Active Cycles** | 32,126,229 | 32,886,528 | +760K |
| **Compute (SM) Throughput** | 76.76% | **84.13%** | cutlass SM 更忙 |
| **Memory Throughput** | 69.56% | 67.63% | 接近 |
| **DRAM Throughput** | **54.31%** | **10.76%** | cutlass DRAM 压力极低 |
| **Issue Slots Busy** | 76.76% | **84.13%** | cutlass issue 更满 |
| **Executed IPC (Active)** | 3.09 | **3.39** | cutlass IPC 更高 |
| **Warp Cycles / Issued Inst** | 5.12 | **4.68** | cutlass 调度更高效 |

### 2.2 指令数（关键！）

| Metric | external_db | cutlass_stage5 | 差距 |
|--------|-------------|----------------|------|
| **Executed Instructions** | **2,386,026,496** | **2,678,489,088** | **+292,462,592 (+12.3%)** |
| **Branch Instructions** | 4,186,112 | 4,218,880 | 接近 |

**FFMA 计算量两者完全相同**（checksum 一致），但 cutlass 多执行了 **2.92 亿条非 FFMA 指令**。

### 2.3 Memory 子系统

| Metric | external_db | cutlass_stage5 | 含义 |
|--------|-------------|----------------|------|
| **Memory Throughput (GB/s)** | **138.92** | **27.52** | external 直接带宽利用率高 5 倍 |
| **L1/TEX Hit Rate** | 0.01% | 0.00% | 都几乎不走 L1/TEX |
| **L2 Hit Rate** | **48.44%** | **95.49%** | ⚠️ cutlass L2 命中率 **2 倍**于 external |
| **L2 Cache Throughput** | 14.69% | 17.31% | cutlass L2 更忙 |
| **Mem Pipes Busy** | 42.14% | 57.03% | cutlass memory pipe 更忙 |
| **Max Bandwidth** | 56.14% | 57.03% | 接近 |

### 2.4 Occupancy & 资源限制

| Metric | external_db | cutlass_stage5 | 含义 |
|--------|-------------|----------------|------|
| **Registers / Thread** | 128 | 128 | 相同 |
| **Shared Memory** | 32.77 KB (static) | 40.96 KB (dynamic) | cutlass 多 8 KB |
| **Theoretical Occupancy** | 33.33% | 33.33% | 相同 |
| **Achieved Occupancy** | 33.03% | 33.06% | 几乎相同 |
| **Block Limit Registers** | 2 | 2 | 寄存器限制 block 数 |
| **Block Limit Shared Mem** | 3 | **2** | cutlass 被 smem 进一步限制 |
| **Active Warps / Scheduler** | 3.96 | 3.97 | 几乎相同 |
| **Eligible Warps / Scheduler** | **2.66** | **2.56** | external 略高 |

### 2.5 Shared Memory 问题（NCU 直接指出的优化点）

**external_db**:
```
OPT   Est. Speedup: 42.79%
      The memory access pattern for shared stores might not be optimal 
      and causes on average a 4.1-way bank conflict across all 20971520 
      shared store requests. This results in 52552692 bank conflicts, 
      which represent 61.03% of the overall 86107124 wavefronts for shared stores.
```

**cutlass_stage5**:
```
OPT   Est. Speedup: 35.41%
      This kernel has uncoalesced shared accesses resulting in a total of 
      167772160 excessive wavefronts (36% of the total 469827584 wavefronts).
```

- external：**shared store bank conflict 严重**（4.1-way, 61% wavefronts）
- cutlass：**shared load uncoalesced 更严重**（36% excessive wavefronts vs external 的 10%）

---

## 3. 根因分析：L2 命中率更高，为什么反而更慢？

这是本次踩坑的**核心悖论**。

### 3.1 定量验证：指令数是决定性的

```
cutlass 额外指令数 = 2,678,489,088 - 2,386,026,496 = 292,462,592 (+12.3%)

cutlass 额外 elapsed cycles (per SM) = 33,165,808 - 32,380,906 = 784,902 (+2.4%)
```

注意：cutlass 的 IPC 更高（3.39 vs 3.09），Warp Cycles / Instruction 更低（4.68 vs 5.12），说明它的指令发射**效率更高**。但即便如此，**总指令数多了 12.3%**，最终执行时间还是长了 2.4%。

**核心结论：L2 命中率 95.49% 确实让 memory access 更快，但 cutlass 不是 memory-bound，而是 issue-bound / compute-bound（SM Throughput 84.13%）。在 compute-bound 的场景下，memory latency 再低也救不了多出来的 3 亿条指令。**

### 3.2 L2 命中率差异的真相

| | external_db | cutlass_stage5 |
|--|-------------|----------------|
| **Memory Throughput (GB/s)** | **138.92** | **27.52** |
| **L2 Hit Rate** | 48.44% | **95.49%** |
| **DRAM Throughput** | **54.31%** | **10.76%** |

- external 的 `LDG.128` 直接以 138.92 GB/s 的速率吞吐数据，大量请求直达 DRAM（L2 miss 多），但因为带宽足够高，latency 被自然隐藏。
- cutlass 的 `cp.async` 把数据缓存在 L2（hit 95.49%），DRAM 压力降到 10.76%，但 **memory pipe 实际吞吐只有 27.52 GB/s** —— cp.async 4B 的细粒度导致 memory subsystem 的有效吞吐反而更低。

**L2 hit rate 高 ≠ 快。它只说明数据在 L2 里，但如果 kernel 的瓶颈是指令发射，memory 再快也没用。**

### 3.3 那 3 亿条额外指令从哪来？

**① cp.async 的管理指令（最大头）**

SASS 层面证据：
| 指令 | external | cutlass |
|------|----------|---------|
| LDG.E.128 | 8 | 0 |
| LDGSTS.E.LTC128B | 0 | **40** |
| IMAD/LEA/IADD3 | 少量 | **大量** |

`cp.async` 引入了：
- `cp.async.commit_group`
- `cp.async.wait_group` → 编译成 `DEPBAR.LE` + `BRA` 自旋
- `LDGDEPBAR`
- `BAR.SYNC.DEFER_BLOCKING`
- 分散在 8 个 k_block 中的条件 cp.async group issue

**② kCtaK = 8 导致外层循环翻倍**

| | external | cutlass |
|--|----------|---------|
| K per tile | 16 | 8 |
| Loop trip count | 256 | 512 |

每次 iteration 的固定开销：`__syncthreads()`、pipeline index 更新、`k_tiles_to_compute` 递减等。

**③ shared memory uncoalesced access**

NCU Source Counters 明确指出：
- cutlass 有 **167,772,160 excessive wavefronts**（36%）due to uncoalesced shared accesses
- external 只有 **50,331,648**（10%）

这些 excessive wavefronts 会导致 shared memory access 被 serialize，浪费 cycle。

---

## 4. SIMT vs TensorCore：本质差异（为什么同样的策略水土不服）

这是把 TensorCore 优化策略生搬硬套到 SIMT 上的经典案例。

### 4.1 Math Density（计算密度）

| | TensorCore (mma) | SIMT (FFMA) |
|--|------------------|-------------|
| 单条指令计算量 | 极高。`mma.sync.m16n8k8` ≈ **2048 FLOPs** | 极低。`FFMA` = **2 FLOPs** |
| 指令密度 | 极低 | 极高 |
| 典型瓶颈 | Memory latency（指令太少，SM 空转等数据） | **Issue bandwidth**（指令太多，发射不过来） |
| cp.async 收益 | **极大**。隐藏 1 cycle latency = 挽救 2048 FLOPs | **极小**。隐藏 1 cycle latency = 多发射 2 FLOPs |

### 4.2 为什么 TensorCore 需要疯狂 hide latency，SIMT 不需要？

**TensorCore 时间线**:
```
Cycle 0:  issue mma (2048 FLOPs) -> latency ~10 cycles
Cycle 1-9: SM 空等 mma 结果
           -> 必须用 cp.async 搬数据，填满这 9 个 cycle
Cycle 10: mma 完成，继续下一轮
```

TensorCore 的 mma latency / throughput ratio 很高。**Memory latency hiding 是 TensorCore 性能的决定性因素。**

**SIMT 时间线**:
```
Cycle 0:  issue FFMA (2 FLOPs)
Cycle 1:  issue FFMA (2 FLOPs)
Cycle 2:  issue FFMA (2 FLOPs)
Cycle 3:  issue FFMA (2 FLOPs)
Cycle 4:  issue FFMA (2 FLOPs)  // 前几条结果已出，FFMA latency = 4 cycles
...
```

FFMA latency 只有 4 cycles，且 A100 每个 SM 有 64 个 FFMA unit。几千条独立的 FFMA 等着被发射，SM 根本没空去 "stall 等内存"。**Natural ILP 已经把 latency bubble 填满了。**

### 4.3 指令 Mix 的致命差异

从 NCU 数据：

| | external_db | cutlass_stage5 |
|--|-------------|----------------|
| 总指令数 | 23.86 亿 | 26.78 亿 |
| FFMA 计算量 | 相同 | 相同 |
| **非 FFMA 指令占比** | **~9%** | **~20%** |

cutlass 版本有 **20% 的 issue bandwidth 被非 FFMA 指令吃掉**。

如果是 TensorCore kernel，这个比例通常只有 5~10%，因为一条 mma 就值几千 FLOPs，SM 有充足的 "时间预算" 去消化 cp.async 的管理开销。

**但在 SIMT 上，FFMA 太便宜了——便宜到 issue slot 比 memory latency 值钱得多。**

### 4.4 Roofline 视角

SIMT GEMM 的 Arithmetic Intensity：
- Tile: 128×128×K FLOPs = 32,768×K FLOPs
- Memory traffic: (128×K + 128×K) × 4 bytes = 1,024×K bytes
- **Arithmetic Intensity = 32 FLOPs/byte**

A100 HBM bandwidth ~2TB/s，FP32 peak ~19.5 TFLOPs：
- Bandwidth-bound threshold = 19.5T / 2T ≈ **9.75 FLOPs/byte**
- 32 FLOPs/byte **已经远高于转折点**，本身就是 **compute-bound**

**既然是 compute-bound，优化 memory latency 的收益天花板极低。**

---

## 5. 具体实现层面的失误

### 5.1 cp.async 粒度 = 4B（最大败笔）

```cpp
// cutlass_like 的宏
#define CP_ASYNC_CA_4B(dst, src) \
  asm("cp.async.ca.shared.global.L2::128B [%0], [%1], 4;" ...)
```

A100/SM80 的 `cp.async` 最小支持 4B，但**最优是 16B**。每次 4B cp.async 的指令发射开销、L2 tag lookup、request dispatch 和 16B 几乎一样，但数据量只有 1/4。

**结果**：
- cutlass Memory Throughput 只有 **27.52 GB/s**
- external 用 `LDG.128` 达到 **138.92 GB/s**

cp.async 虽然 async，但 4B 粒度导致 memory pipe 的有效吞吐被压垮了。

### 5.2 kCtaK = 8（继承了 TensorCore 配置，对 SIMT 有害）

| | external_db | cutlass_stage5 |
|--|-------------|----------------|
| K per tile | 16 | 8 |
| 外层 loop trip count | 256 | 512 |

kCtaK=8 导致：
1. 外层循环多跑一倍 → 2 倍的 `__syncthreads()`、`cp.async.commit_group`/`wait_group`
2. 内层循环体更短 → 编译器展开更少，ILP 更低
3. Pipeline index 更新次数翻倍 → 更多整数运算指令

TensorCore 用 kCtaK=8 是因为 mma 指令内部 consume K=8（或 16），这是 hardware constraint。SIMT 没有这种限制，`bK=16` 或更大对 SIMT 更有利。

### 5.3 Shared Memory Access 问题

**external_db**：
- **Shared store bank conflict 严重**（4.1-way, 61.03% wavefronts）
- 根源：`a_smem[smem_idx] = ldg_a.x; a_smem[smem_idx + BM] = ldg_a.y;` 的 scatter store 模式
- 但 load 端相对干净（10% excessive wavefronts）

**cutlass_stage5**：
- **Shared load uncoalesced 更严重**（36% excessive wavefronts）
- 根源：warp 级 shared memory load pattern 有 stride，导致同一 warp 内 thread 访问不连续
- Store 端无显式 STS（cp.async 直接写），但 load 端的 penalty 更大

---

## 6. 教训总结

### 6.1 核心原则

> **TensorCore kernel 优化的是 "怎么让 SM 在等 mma 结果时不空转"；**
> **SIMT kernel 优化的是 "怎么让 SM 每秒发射尽可能多的 FFMA"。**

### 6.2 SIMT GEMM 的优化 checklist

1. **最大化 FFMA issue rate**
   - 减少非 FFMA 指令数量
   - 避免过度使用 cp.async / DEPBAR / 复杂 pipeline

2. **向量化 global load（LDG.128 / LDG.64）**
   - 压缩 global load request 数量，提升 memory pipe 有效吞吐
   - 4B 的 cp.async 在 SIMT 上得不偿失

3. **足够大的 bK（≥ 16，最好能到 32）**
   - 减少外层 loop trip count
   - 让编译器生成更大的展开循环体，提高 ILP

4. **减少 shared memory bank conflict & uncoalesced access**
   - SIMT 的 shared memory bandwidth 是真正的瓶颈之一
   - Store pattern 和 load pattern 都要检查

5. **足够的 occupancy**
   - 256 threads/block 对 SIMT 通常够用
   - FFMA 的 4-cycle latency 主要靠 ILP 隐藏，不是靠多 warp

### 6.3 L2 Hit Rate 高 ≠ 更快

本次踩坑最大的认知陷阱：
- **L2 Hit Rate 95.49%** 只是说明 cp.async 成功把数据留在了 L2
- 但 cutlass 是 **compute-bound**（SM Throughput 84.13%）
- 在 compute-bound 场景下，memory latency 再低也救不了多出来的 3 亿条指令
- external 的 L2 Hit Rate 只有 48.44%，但它直接以 138.92 GB/s 吞吐数据，且总指令少 12.3%，最终更快

### 6.4 对这次优化的反思

这次踩坑的本质是**优化方向错误**：
- 问题不是 "memory latency 太高"
- 问题被错误地诊断为 "需要更强的 latency hiding"
- 实际上 baseline 的 latency 已经被 natural ILP 隐藏得差不多了
- 强加的 cp.async pipeline 反而成了累赘

**当 baseline 的 stall 已经见底时，进一步优化延迟隐藏的收益是边际递减的，但新增指令的 cost 是线性累加的。**

---

## 7. 后续可验证的改进方向

如果要改进 `cutlass_like` 版本，最直接的两个修改：

1. **cp.async 从 4B 改 16B**（或合并成 16B）
   - 预期 Memory Throughput 从 27 GB/s 提升到接近 external 的 138 GB/s
   - 指令总数减少 2~3 亿条
   - 性能应能反超 external_db

2. **kCtaK 从 8 提到 16（或 32）**
   - 外层 loop trip count 减半（或再减半）
   - 需同步增加寄存器分配或调整 tile 大小
   - 注意 occupancy 和寄存器压力的平衡

3. **修复 shared memory uncoalesced access**
   - 根据 NCU Source Counters 的提示，调整 warp 级的 shared load pattern
   - 目标是把 36% excessive wavefronts 降到 10% 以下

这两个改动能验证本文的分析是否正确。

---

## 8. 参考数据存档

### 8.1 环境信息
```
GPU: NVIDIA GeForce RTX 4060 Laptop (SM89, 24 SMs, 12 TPCs)
Driver: 590.48.01
CUDA: 13.1
NCU: NVIDIA Nsight Compute 2025.3
编译: sm_80, -O3, --ptxas-options=-v
测试规模: M=N=K=4096
```

### 8.2 寄存器使用（ptxas 输出）
| Kernel | Registers | Shared Memory |
|--------|-----------|---------------|
| external_db | 128 | 32.77 KB (static) |
| cutlass_stage5 `<false,false,false>` | 128 | 40.96 KB (dynamic) |

### 8.3 NCU 报告关键提取
```bash
# 以本报告为准
ncu --import ncu_temp/temp-f.ncu-rep --page details
```

**cutlass_stage5 关键指标**：
- Duration: 17.28 ms
- Elapsed Cycles: 33,165,808
- SM Throughput: 84.13%
- IPC (Active): 3.39
- Executed Instructions: 2,678,489,088
- Memory Throughput: 27.52 GB/s
- DRAM Throughput: 10.76%
- L2 Hit Rate: 95.49%
- Shared Uncoalesced: 167,772,160 excessive wavefronts (36%)

**external_db 关键指标**：
- Duration: 16.87 ms
- Elapsed Cycles: 32,380,906
- SM Throughput: 76.76%
- IPC (Active): 3.09
- Executed Instructions: 2,386,026,496
- Memory Throughput: 138.92 GB/s
- DRAM Throughput: 54.31%
- L2 Hit Rate: 48.44%
- Shared Store Bank Conflict: 4.1-way, 61.03% wavefronts
- Shared Uncoalesced: 50,331,648 excessive wavefronts (10%)

---

*记于 2026-05-29。本次踩坑的代价是几天几夜的困惑，但收获是对 SIMT 和 TensorCore 的本质区别、以及 "L2 hit rate 高 ≠ 更快" 有了刻骨铭心的理解。*
