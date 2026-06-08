# CUTE HGEMM Register Prefetch 短依赖实验笔记

> Date: 2026-06-04  
> Device: NVIDIA GeForce RTX 4060 Laptop GPU, CC 8.9, 24 SMs  
> Tool: `/usr/local/NVIDIA-Nsight-Compute/ncu`, Nsight Compute 2025.3.1  
> Problem: `M=N=K=4096`

## 背景

这次实验对比两个 CUTE HGEMM kernel：

- `cute_ampere_hgemm_16816`: 原始版本，有 shared memory 到 register 的预取。
- `cute_ampere_hgemm_16816_no_reg_prefetch`: 去掉 register prefetch 的版本。

no-reg 版本只去掉 shared->register 这一层的提前预取，global->shared 的 `cp.async` pipeline 保持不变。也就是说，这次实验尽量只观察：

```text
LDSM(shared -> register) 和 HMMA 之间的依赖距离
```

原版主循环的关键结构是：

```cpp
copy(s2r_atom_a, tXsA_p(_, _, k_block_next), tXrA(_, _, k_block_next));
copy(s2r_atom_b, tXsB_p(_, _, k_block_next), tXrB(_, _, k_block_next));

gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
```

它先加载下一段 `k_block_next` 到寄存器，再计算当前 `k_block`。这样 HMMA 真正消费某个 register fragment 时，对应的 LDSM 已经提前发生过。

no-reg 版本则是：

```cpp
copy(s2r_atom_a, tXsA_p(_, _, k_block), tXrA(_, _, k_block));
copy(s2r_atom_b, tXsB_p(_, _, k_block), tXrB(_, _, k_block));

gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
```

也就是 LDSM 后面马上接 HMMA。这个版本寄存器压力会下降，但更容易暴露 `LDSM -> HMMA` 的 scoreboard dependency。

## 正确性和资源

小尺寸验证：

```bash
./build/learn_cuda \
  --kernel cute-hgemm \
  --kernel cute-hgemm-noreg \
  --m 256 --n 256 --k 256 \
  --verify
```

结果：

```text
cute_hgemm       C[0]=0.012817383 checksum=-1.427152
cute_hgemm_noreg C[0]=0.012817383 checksum=-1.427152
verify max_abs_error=0.00018310547 max_rel_error=0.00018310547
```

ptxas / `KERNEL_REPORT.md` 资源：

| Kernel | Registers/thread | Spill |
|---|---:|---:|
| `cute_ampere_hgemm_16816` | 168 | 0 |
| `cute_ampere_hgemm_16816_no_reg_prefetch` | 128 | 0 |

去掉 register prefetch 后，寄存器从 168 降到 128，但没有提高 occupancy，因为这个 kernel 当前被 dynamic shared memory 限制在 1 CTA/SM。

## 普通 benchmark

普通 CUDA event benchmark：

```bash
./build/learn_cuda bench \
  --kernel cute-hgemm \
  --kernel cute-hgemm-noreg \
  --m 4096 --n 4096 --k 4096 \
  --warmup 10 --iters 50
```

结果：

| Kernel | Avg time | TFLOP/s |
|---|---:|---:|
| `cute_hgemm` | 3.2543 ms | 42.2337 |
| `cute_hgemm_noreg` | 3.6663 ms | 37.4871 |

普通 benchmark 显示 no-reg 版本慢约 11%。但这个只能说明现象，不能说明慢在哪里，所以后面用 ncu 看 stall reason。

## NCU: 重点看 short dependency

这次重点指标是：

- `smsp__warp_issue_stalled_short_scoreboard_per_warp_active.*`
  - 等 MIO/shared 相关 scoreboard dependency。
  - 对这里来说，最关心的是 HMMA 是否等前面的 LDSM 结果。
- `smsp__warp_issue_stalled_wait_per_warp_active.*`
  - 固定延迟执行依赖。
  - ncu 的 WarpStateStats 文案里常叫 fixed latency execution dependency。
- `smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.*`
  - tensor/math pipe 被打满时的等待。
  - 如果 prefetch 成功把 LDSM latency 藏掉，stall 往往会更多表现成 math pipe throttle。

使用命令：

```bash
/usr/local/NVIDIA-Nsight-Compute/ncu \
  --target-processes all \
  --kernel-name-base function \
  -k cute_ampere_hgemm_16816 \
  --launch-skip 5 \
  --launch-count 1 \
  --metrics gpu__time_duration.sum,launch__registers_per_thread,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.ratio,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.ratio,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.ratio,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.ratio,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.ratio,smsp__warps_active.avg.per_cycle_active,smsp__warps_eligible.avg.per_cycle_active,smsp__inst_executed.sum,smsp__cycles_elapsed.avg \
  --page raw \
  ./build/learn_cuda bench \
    --kernel cute-hgemm \
    --m 4096 --n 4096 --k 4096 \
    --warmup 5 --iters 1
```

no-reg 版本只换 kernel filter 和命令行 kernel：

```bash
/usr/local/NVIDIA-Nsight-Compute/ncu \
  --target-processes all \
  --kernel-name-base function \
  -k cute_ampere_hgemm_16816_no_reg_prefetch \
  --launch-skip 5 \
  --launch-count 1 \
  --metrics gpu__time_duration.sum,launch__registers_per_thread,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.ratio,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.ratio,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.ratio,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.ratio,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.ratio,smsp__warps_active.avg.per_cycle_active,smsp__warps_eligible.avg.per_cycle_active,smsp__inst_executed.sum,smsp__cycles_elapsed.avg \
  --page raw \
  ./build/learn_cuda bench \
    --kernel cute-hgemm-noreg \
    --m 4096 --n 4096 --k 4096 \
    --warmup 5 --iters 1
```

结果：

| Metric | reg-prefetch | no-reg-prefetch |
|---|---:|---:|
| Duration | 3.17 ms | 3.58 ms |
| Registers/thread | 168 | 128 |
| `short_scoreboard.pct` | 1.12% | 6.04% |
| `wait.pct` | 35.53% | 39.20% |
| `math_pipe_throttle.pct` | 46.38% | 38.27% |
| `long_scoreboard.pct` | 0.61% | 1.43% |
| `mio_throttle.pct` | 1.27% | 1.08% |
| Active warps / scheduler | 0.99 | 1.00 |
| Eligible warps / scheduler | 0.13 | 0.11 |
| Executed instructions | 73,297,920 | 73,977,856 |
| Avg elapsed cycles | 6,073,971 | 6,863,214 |

关键观察：

- no-reg 的 `short_scoreboard.pct` 从 1.12% 增加到 6.04%，约 5.4 倍。
- no-reg 的 `wait.pct` 也从 35.53% 增加到 39.20%。
- reg-prefetch 版本的 `math_pipe_throttle.pct` 更高，说明它更接近 tensor pipe 被打满，而不是 HMMA 等 LDSM 数据。
- no-reg 虽然少 40 个 register/thread，但 occupancy 没变，反而 eligible warps 从 0.13 降到 0.11。

这说明去掉 register prefetch 后，节省寄存器没有转化成并发度收益；代价则是 HMMA 更频繁等 shared->register load。

## NCU SourceCounters: HMMA 处的证据

为了确认 stall 落在什么指令附近，用 SourceCounters 只看 PC sampling 的几个指标：

```bash
/usr/local/NVIDIA-Nsight-Compute/ncu \
  --target-processes all \
  --metrics smsp__pcsamp_warps_issue_stalled_short_scoreboard,smsp__pcsamp_warps_issue_stalled_wait,smsp__pcsamp_warps_issue_stalled_math_pipe_throttle,smsp__pcsamp_sample_count \
  --kernel-name-base function \
  -k cute_ampere_hgemm_16816 \
  --launch-skip 5 \
  --launch-count 1 \
  --page source \
  --print-source cuda,sass \
  ./build/learn_cuda bench \
    --kernel cute-hgemm \
    --m 4096 --n 4096 --k 4096 \
    --warmup 5 --iters 1 \
  > ncu_reports/cute_hgemm_reg_prefetch_source_stalls_min.txt
```

no-reg 版本：

```bash
/usr/local/NVIDIA-Nsight-Compute/ncu \
  --target-processes all \
  --metrics smsp__pcsamp_warps_issue_stalled_short_scoreboard,smsp__pcsamp_warps_issue_stalled_wait,smsp__pcsamp_warps_issue_stalled_math_pipe_throttle,smsp__pcsamp_sample_count \
  --kernel-name-base function \
  -k cute_ampere_hgemm_16816_no_reg_prefetch \
  --launch-skip 5 \
  --launch-count 1 \
  --page source \
  --print-source cuda,sass \
  ./build/learn_cuda bench \
    --kernel cute-hgemm-noreg \
    --m 4096 --n 4096 --k 4096 \
    --warmup 5 --iters 1 \
  > ncu_reports/cute_hgemm_no_reg_prefetch_source_stalls_min.txt
```

source page 的列顺序是：

```text
sample_count, math_pipe_throttle, short_scoreboard, wait
```

主循环 HMMA 附近的 PC sampling：

| Source / SASS | sample_count | math_pipe_throttle | short_scoreboard | wait |
|---|---:|---:|---:|---:|
| reg-prefetch `gemm(...)` / `HMMA.16816` | 636 | 312 | 7 | 154 |
| no-reg `gemm(...)` / `HMMA.16816` | 789 | 0 | 731 | 0 |

这张表是最直接的证据：

- reg-prefetch 版本 HMMA 处几乎没有 `short_scoreboard`，只有 7 个采样。
- no-reg 版本 HMMA 处 `short_scoreboard` 有 731 个采样。
- no-reg 版本的 HMMA 正在等前面的 LDSM 结果。
- reg-prefetch 版本把这段等待藏掉后，HMMA 处更多表现为 `math_pipe_throttle`，也就是 tensor pipe 已经更接近被喂满。

SourceCounters 里还能看到 no-reg 版本的多条 `LDSM.16.M88.4` / `LDSM.16.MT88.4` 之后，HMMA 处大量出现 short scoreboard sample。这和代码结构一致：no-reg 是当前 k-block 的 LDSM 后马上 HMMA，依赖距离太短。

## 结论

这次实验说明：

**register prefetch 的核心价值，是拉开 `LDSM -> HMMA` 的距离，隐藏 shared->register load 到 tensor core 消费之间的 short scoreboard dependency。**

no-reg 版本看起来更轻：

```text
168 regs/thread -> 128 regs/thread
```

但由于 shared memory 已经把 CTA residency 限制在 1 CTA/SM，少用寄存器没有提升 occupancy。反而因为 LDSM 和 HMMA 紧挨着，HMMA 直接暴露 `short_scoreboard`：

```text
short_scoreboard.pct: 1.12% -> 6.04%
HMMA short_scoreboard samples: 7 -> 731
```

因此 no-reg 版本慢不是因为 global memory 或 cp.async，而是因为 shared->register 到 HMMA 的短依赖没有被软件流水隐藏。

一个有用的判断准则：

```text
如果去掉 register prefetch 后 register 降了，
但 occupancy 没变，
并且 HMMA 处 short_scoreboard 暴涨，
那 register prefetch 不能删。
```

## 后续可尝试方向

如果想在 register 数和 latency hiding 之间折中，可以尝试：

- 只对 A 或 B 做 register prefetch，看哪一侧 LDSM 更关键。
- 保留一段 lead distance，但缩短 fragment 生命周期，观察是否能低于 168 regs/thread。
- 改变 `K_BLOCK_MAX` 内部的 LDSM/HMMA 排布，让 LDSM 至少提前一个 HMMA slot。
- 每次改动后优先看 `short_scoreboard.pct` 和 SourceCounters 里 HMMA 的 short scoreboard samples，而不是只看 CUDA event 时间。

当前结论已经比较明确：对这个 kernel，register prefetch 是有效优化，不是无意义的寄存器浪费。
