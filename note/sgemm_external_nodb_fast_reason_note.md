# SGEMM External No-DB 为什么会很快

## 背景

这次问题来自一个反直觉现象：

```text
./learn_cuda bench --size 4096 \
  --kernel cublas \
  --kernel custom \
  --kernel external-db \
  --kernel external-nodb

cublas          20.7314 ms
custom          20.6643 ms
external_db     19.8381 ms
external_nodb   19.6467 ms
```

直觉上 `custom` 应该更强：

- `custom` 的 K tile 是 32，外部版本是 16。
- `custom` 在计算时一次展开 `MMA_K=8`。
- `custom` 已经给 A smem stride 做了 `+4` pad。

但 `external-nodb` 反而最快。核心问题是：

**为什么一个没有 K=8 展开的外部单缓冲 kernel，反而能赢？**

## 当前代码状态

当前 `custom` 已经做了 A smem pad：

```cpp
constexpr int ASmemPad = 4;
constexpr int ASmemStride = kBM + ASmemPad;  // 132
```

资源：

| Kernel | Registers/thread | Static smem |
|---|---:|---:|
| `custom` | 125 | 33280 B |
| `external-db` | 118 | 32768 B |
| `external-nodb` | 128 | 16384 B |

注意：本仓库当前只有 `./build/learn_cuda`，没有根目录 `./learn_cuda`。如果机器上跑的是另一个 `./learn_cuda`，需要确认它是不是旧二进制。

## 先说结论

`external-nodb` 快的主要原因不是“没有 bank conflict”，而是：

**它的 inner loop 把 shared load 和 FFMA 以更小颗粒交替执行，MIO 压力更平滑；`custom` 的 K=8 展开把 shared load 集中成一波 burst，虽然降低 scoreboard，但更容易造成 MIO throttle / dispatch stall。**

换句话说：

```text
custom:
  一口气 load 8 个 K slice 的 A/B fragment
  然后一口气做 8 个 K slice 的 FFMA

external-nodb:
  load 1 个 K slice 的 A/B fragment
  立刻做这个 K slice 的 FFMA
  再进入下一个 K slice
```

`custom` 的 K=8 展开并不是免费 ILP。它减少了“等 shared load 数据 ready”的时间，但把压力集中到了 MIO / shared-memory instruction issue 上。

## ncu 证据

在当前 4060，本地串行 ncu，size 4096：

| Metric | `custom` | `external-nodb` |
|---|---:|---:|
| Duration | 18.43 ms | 18.81 ms |
| Issue Active | 70.46% | 68.94% |
| Issued Warp/Scheduler | 0.71 | 0.69 |
| Registers/thread | 125 | 128 |
| Static smem/block | 33.28 KiB | 16.38 KiB |
| Theoretical occupancy | 33.33% | 33.33% |
| Achieved occupancy | 33.03% | 33.03% |
| Instructions Executed | 2.3929B | 2.3902B |

Warp stall：

| Stall reason | `custom` | `external-nodb` |
|---|---:|---:|
| MIO Throttle | 0.47 | 0.31 |
| Dispatch Stall | 0.53 | 0.55 |
| Long Scoreboard | 0.37 | 0.53 |
| Short Scoreboard | 0.26 | 0.56 |
| Barrier | 0.33 | 0.43 |
| Not Selected | 2.31 | 2.07 |

解释：

- `custom` 的 K=8 展开确实有效：scoreboard 更低。
- 但 `custom` 的 MIO throttle 仍然更高：`0.47` vs `0.31`。
- `external-nodb` scoreboard 更差，但 MIO 更轻、调度更平滑。

这就是为什么普通 benchmark 里 `external-nodb` 经常能赢：差距只有几个百分点，谁在某次运行里更少被 MIO/调度/boost 噪声影响，谁就赢。

## 普通 bench 有明显顺序效应

同一组 kernel 顺序不同，结果会翻转。

当前本地串行跑：

```bash
./build/learn_cuda bench --size 4096 \
  --kernel cublas \
  --kernel custom \
  --kernel external-db \
  --kernel external-nodb \
  --warmup 10 --iters 100
```

结果：

| Kernel | Avg ms | TFLOP/s |
|---|---:|---:|
| `cublas` | 20.9686 | 6.5545 |
| `custom` | 18.9896 | 7.2376 |
| `external-db` | 19.3253 | 7.1119 |
| `external-nodb` | 19.2085 | 7.1551 |

反向顺序：

```bash
./build/learn_cuda bench --size 4096 \
  --kernel external-nodb \
  --kernel external-db \
  --kernel custom \
  --kernel cublas \
  --warmup 10 --iters 100
```

结果：

| Kernel | Avg ms | TFLOP/s |
|---|---:|---:|
| `external-nodb` | 19.7889 | 6.9453 |
| `external-db` | 19.9920 | 6.8747 |
| `custom` | 20.2435 | 6.7893 |
| `cublas` | 21.4967 | 6.3935 |

孤立进程串行跑：

| Kernel | Avg ms | TFLOP/s |
|---|---:|---:|
| `custom` | 20.2044 | 6.8024 |
| `external-nodb` | 19.9502 | 6.8891 |

所以不能只看一次多 kernel bench 的排序。这里的真实差距很小，而且 GPU boost、温度、运行顺序会改变 1% 到数%的结果。

## `custom` 不是完全无 bank conflict

这是这次最容易误判的地方。

`custom` 的 A 写 smem 原来 stride 是 128：

```text
bank = row
```

同一个 warp 写 A 时会形成很重的冲突。加 `+4` pad 后 stride 变成 132：

```text
bank = ((load_A_col * 4 + k) * 132 + row) % 32
     = (16 * load_A_col + 4 * k + row) % 32
```

对于一个 warp：

- `load_A_col` 是 0..7。
- `row` 是 0..3。
- `16 * load_A_col` 只会在两个 bank group 间跳。

所以 `+4` pad 不是完全消除冲突，而是把原来的更重冲突降成大约 4-way 冲突。为什么不用 `+1`？因为 `ASmemStride=129` 会破坏后续 `float4` shared load 的 16B 对齐。

bank conflict counter，size 4096：

| Metric | `custom` | `external-nodb` |
|---|---:|---:|
| Shared store bank conflicts | 53,849,624 | 50,331,648 |
| Shared load bank conflicts | 115,458 | 0 |

这说明：

- `custom` 并不是“没有 bank conflict”。
- `external-nodb` 也有 A 写 smem 的 store conflict。
- 但 `external-nodb` 的 shared read conflict 是 0，而且 store conflict 也不比 `custom` 更差。

所以 bank conflict 不是 `custom` 的决定性优势。

## 为什么 no-db 没有 K 展开也不慢

`external-nodb` 的内层循环每个 `dot_product_idx` 做：

```text
load reg_a for 1 K slice
load reg_b for 1 K slice
做 2x2 个 4x4 micro tile 的 FFMA
```

也就是每个 K slice 后面跟着足够多的 FFMA。虽然它没有一次缓存 8 个 K slice，但每个 K slice 的 load 后面有 64 个 scalar FFMA，可以摊掉一部分 shared-load latency。

`custom` 的 K=8 做法是：

```text
load 8 个 A slice
load 8 个 B slice
compute 8 个 slice
```

从依赖角度看更好，所以 scoreboard 低；但从 shared-memory issue 节奏看更差，因为 load burst 太集中，所以 MIO throttle 高。

这解释了两个 ncu 现象：

```text
custom scoreboard 更低
external-nodb MIO 更低
```

谁更快取决于这两个方向谁占上风。在当前本地 pad 后，ncu 单 kernel 下 `custom` 略好；但普通 bench 里两者非常接近，顺序一变就可能翻。

## 为什么 external-db 没明显赢

双缓冲的主要收益是隐藏 global memory load latency。但 4096³ 的 128x128 SGEMM 算术强度很高，global memory load 不是唯一瓶颈。

`external-db` 还要付出：

- 更多 staging 逻辑；
- 更多寄存器生命周期管理；
- 两份 smem stage；
- 最后一块 tile 的收尾逻辑更复杂。

所以它可能降低 barrier/long scoreboard，但收益会被额外调度和指令开销吃掉。当前结果里 `external-db` 和 `external-nodb` 的差距也只有几个百分点。

## 这次的答案

`external-nodb` 之所以能最快，是因为它的执行形态更均衡：

- 它不是完全没有 bank conflict，但 bank conflict 不比 `custom` 更差。
- 它没有 K=8 展开，所以 scoreboard 更高。
- 但它没有 K=8 shared-load burst，所以 MIO throttle 更低。
- 它的 shared load/compute 颗粒更细，调度更平滑。
- 普通 bench 的差距只有几个百分点，运行顺序足以让排序翻转。

一句话：

**`custom` 赢在更强的 K 方向 ILP，`external-nodb` 赢在更平滑的 MIO/调度节奏；当前差距小到普通 bench 会受顺序影响。**

## 下一步

不要再把目标理解成“把 K 展开越大越好”。更合理的方向是：

1. 保留 `custom` 的 A pad。
2. 做 `MMA_K=2` / `MMA_K=4` 的正确版本，并补齐 tile-end sync。
3. 用 ncu 对比：
   - `MIO Throttle`
   - `Long/Short Scoreboard`
   - `Issue Active`
   - shared bank conflict
4. 找一个比 K=8 更平滑、比 K=1 scoreboard 更低的中间点。

目标不是最大化 K 展开，而是让 shared load 和 FFMA 的节奏更像 `external-nodb` 那样平滑，同时保留 `custom` 已经验证有效的 pad。
