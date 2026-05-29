# cutlass-like 128x128x8 stage5 对齐 CUTLASS 报告

这份报告总结 `src/kernels/cutlass_like_sgemm_128x128x8stage5.cuh`
为了对齐 CUTLASS `a100_128x128x8_s5` 做过的主要修改，以及每个修改为什么有效。

目标参考实现来自 `cutlass_gemm_ref.txt` 里的：

```cpp
{"a100_128x128x8_s5",
 "CUTLASS SM80 SIMT f32: TB=128x128x8 Warp=32x64x8 Stages=5 Swizzle=8",
 &run_cutlass_gemm<Sgemm128x128x8S5>}
```

也就是：

- CTA tile: `128 x 128 x 8`
- warp tile: `32 x 64 x 8`
- SIMT F32
- 5-stage multistage pipeline
- `GemmIdentityThreadblockSwizzle<8>`
- 256 threads/block

## 0. 最终结果概览

改完之后，关键 NCU 指标已经和 CUTLASS 接近很多：

| 指标 | 原始 cutlass-like | 当前 cutlass-like | CUTLASS ref |
|---|---:|---:|---:|
| grid | `(32,32,1)` | `(256,4,1)` | `(256,4,1)` |
| regs/thread | 140 | 138 | 129 |
| L2 hit rate | 约 `49%` | 约 `95.7%` | 约 `95.5%` |
| L1 tag requests global | `85,065,728` | `84,410,368` | `84,410,368` |
| shared wavefront total | `605,650,944` | `469,827,584` | `470,876,160` |
| shared wavefront ideal | `436,568,064` | `302,055,424` | `303,104,000` |
| shared wavefront excessive | `169,082,880` | `167,772,160` | `167,772,160` |
| NCU duration | 约 `21.60 ms` | 约 `20.06 ms` | 约 `19.39 ms` |
| benchmark | - | `19.82 ms / 6.93 TFLOP/s` | - |

特别注意：

- occupancy 一直是一样的，所以这次不拿 occupancy 解释性能差异。
- 真正的大头是三个东西：
  1. drain 阶段重复发 `cp.async`
  2. block swizzle 不同导致 L2 reuse 不同
  3. warp lane layout 不同导致 shared `LDS.128` wavefront 变多

## 1. 修正 `cp.async` shared address

### 改了什么

当前代码里有：

```cpp
__device__ __forceinline__ unsigned smem_addr(const void *ptr) {
  return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}
```

然后 `cp.async` 宏使用：

```cpp
"r"(::cutlass_like::smem_addr(dst))
```

### 为什么要这样

`cp.async.shared.global` 的 shared memory 目的地址不是普通 64-bit generic pointer。
它要的是 shared address space 里的 32-bit address。

如果直接把普通指针传给 inline asm，编译器不一定会按你想要的 shared address 形式生成代码。
轻则生成不理想 SASS，重则地址语义就是错的。

CUTLASS 内部也会把 shared pointer 转成 shared address 再喂给异步拷贝指令。

### 为什么有效

这个改动保证了所有：

```cpp
cp.async.ca.shared.global.L2::128B [...]
```

的目的地址都是合法的 shared-space address。

这是 correctness 和 SASS 形态的基础修复。

## 2. 增加统一的 `issue_tile_cp_async`

### 改了什么

把一个 K tile 的 A/B global-to-shared 拷贝集中到：

```cpp
issue_tile_cp_async(...)
```

里面。

每个 tile 每个 thread 发：

- A: 4 条 `cp.async`，每条 4B
- B: 4 条 `cp.async`，每条 4B

也就是一个 CTA 合起来搬：

- A tile: `128 x 8` floats
- B tile: `8 x 128` floats

### 为什么要这样

这个函数本身不是性能优化的核心，它主要是为了让 prologue 和 mainloop 使用完全一致的 load 逻辑。

以前在多个地方手写同样的 `cp.async`，很容易出现：

- prologue 和 steady-state 行为不一致
- drain 阶段多发
- 写错 stage
- 修改一处漏掉另一处

集中之后，后面的 pipeline 修复才比较清楚。

## 3. 修正 5-stage pipeline 的 drain 行为

这是第一个非常关键的性能修复。

### 原来的问题

旧逻辑类似：

```cpp
while (k_tile_count > -(K_PIPE_MAX - 1)) {
  ...
  k_tile_next = min(k_tile_next + 1, K_TILE_MAX - 1);
  issue cp.async for k_tile_next;
  ...
}
```

也就是说，当真实 K tile 已经发完以后，`k_tile_next` 会卡在最后一个合法 tile。

然后 drain 阶段还会继续发：

```cpp
cp.async(last_tile)
```

这些重复 load 后面通常不会再被计算使用，所以结果可能还是对的。
但是它浪费 global load、L1 tag request、L2 traffic。

### 现在怎么改

现在把两个概念拆开：

```cpp
int k_tiles_to_issue = K_TILE_MAX;
int k_tiles_to_compute = K_TILE_MAX;
int k_tile_next = 0;
```

含义分别是：

- `k_tiles_to_issue`: 还有多少真实 K tile 需要从 global 发到 shared
- `k_tiles_to_compute`: 还有多少真实 K tile 需要被计算
- `k_tile_next`: 下一个要发的真实 tile id

mainloop 里：

```cpp
if (k_tiles_to_issue > 0) {
  issue_tile_cp_async(...);
  --k_tiles_to_issue;
  ++k_tile_next;
}
CP_ASYNC_COMMIT_GROUP();
```

### 一个很容易踩坑的细节

注意：即使 `k_tiles_to_issue == 0`，现在仍然执行：

```cpp
CP_ASYNC_COMMIT_GROUP();
```

这很重要。

不能简单地写成：

```cpp
if (k_tiles_to_issue > 0) {
  issue_tile_cp_async(...);
  CP_ASYNC_COMMIT_GROUP();
}
```

原因是 `wait_group 3` 看的不是“你心里有几个 tile”，而是硬件 cp.async group 队列。

在 5-stage pipeline 中，mainloop 每算完一个 tile，都应该推进一次 group。
真实 tile 发完以后，drain 阶段虽然不再发新的 `LDGSTS`，但仍然需要提交空 group，
这样后面的 `cp.async.wait_group 3` 才能看到正确的 pipeline 进度。

这就是 CUTLASS 里 predicated iterator / fence 的思想：

- 对无效 tile：不发真实 load
- 但 pipeline group 语义继续推进

### 为什么有效

修复前：

```text
L1 Tag Requests Global = 85,065,728
```

修复后：

```text
L1 Tag Requests Global = 84,410,368
```

CUTLASS ref：

```text
L1 Tag Requests Global = 84,410,368
```

这说明 drain 阶段重复发的 global load 已经消失。

## 4. 对齐 CUTLASS `GemmIdentityThreadblockSwizzle<8>`

这是第二个非常关键的性能修复。

### CUTLASS 怎么做

CUTLASS ref 使用：

```cpp
GemmIdentityThreadblockSwizzle<8>
```

其核心 grid/mapping 是：

```cpp
grid.x = tiled_shape.m() * 8;
grid.y = ceil(tiled_shape.n() / 8);

tile_m = blockIdx.x / 8;
tile_n = blockIdx.y * 8 + blockIdx.x % 8;
```

4096x4096 时，M/N tile 数都是 32，所以：

```text
grid = (32 * 8, ceil(32 / 8), 1)
     = (256, 4, 1)
```

### 我们现在怎么改

kernel 内：

```cpp
int tile_n_count = N / kCtaN;
int tile_m = blockIdx.x / kBlockSwizzle;
int tile_n = blockIdx.y * kBlockSwizzle + blockIdx.x % kBlockSwizzle;
if (tile_n >= tile_n_count) {
  return;
}
```

launch 内：

```cpp
dim3 grid(tile_m_count * kBlockSwizzle,
          (tile_n_count + kBlockSwizzle - 1) / kBlockSwizzle);
```

### 为什么有效

这个改动的核心是改变 CTA 的执行顺序。

旧 grid 是普通 `(N_tiles, M_tiles)`。
CUTLASS swizzle 会把 N 方向按 8 个 tile 分组，然后在这个 N band 内扫 M。

缓存角度可以粗略理解为：

- 普通 grid 更像按完整 N 行扫过去
- swizzle<8> 更像固定一小段 N band，然后扫 M
- 对 B tile 的 L2 reuse 更友好
- 同时和 CUTLASS 的调度顺序一致

实际 NCU 结果非常明显。

旧 cutlass-like：

```text
L2 Hit Rate = 约 49%
DRAM Throughput = 约 43%
grid = (32,32,1)
```

改后：

```text
L2 Hit Rate = 约 95%
DRAM Throughput = 约 9%
grid = (256,4,1)
```

CUTLASS ref：

```text
L2 Hit Rate = 约 95%
DRAM Throughput = 约 9%
grid = (256,4,1)
```

所以 L2/DRAM 这块基本已经对齐。

## 5. 对齐 CUTLASS 的 warp lane layout: `RowMajorInterleaved<2>`

这是第三个，也是最容易误解的性能修复。

### 原来我们为什么 shared wavefront 多

之前 NCU 显示：

```text
我们的 shared wavefront total 约 604M
CUTLASS shared wavefront total 约 471M
```

一开始容易以为是：

- shared load 指令更多？
- bank conflict 更多？
- global-to-shared 的 `LDGSTS` 更多？

但 NCU source page 拆开以后发现不是。

关键证据：

| 指标 | 原来我们 | CUTLASS |
|---|---:|---:|
| `LDGSTS` wavefront | `201,326,592` | `201,326,592` |
| `LDGSTS` excessive | `167,772,160` | `167,772,160` |
| `LDS.128` 执行次数 | `134,250,496` | `134,250,496` |
| `LDS.128` wavefront | `402,751,488` | `268,500,992` |

也就是说：

- global-to-shared 没差
- shared bank excessive 基本没差
- `LDS.128` 指令执行次数也没差
- 差的是：同样一条 `LDS.128`，我们的地址模式让硬件拆成更多 wavefront

### 什么是 wavefront

可以把 shared memory 的一次 warp load 想成：

```text
32 个 lane 同时各自读 16B
```

这是一条 SASS 指令，比如：

```text
LDS.128
```

但硬件内部不一定一次就能把这 32 个 lane 的地址全部服务完。
如果地址分布比较好，可能拆成 2 个 wavefront。
如果地址分布比较散，可能拆成 4 个 wavefront。

注意：

```text
SASS 指令数量没变，但是内部 transaction 数变了。
```

这就是为什么 SASS 行数看起来差不多，但 NCU 的 MIO/shared 压力不同。

### CUTLASS lane layout 是什么

CUTLASS 在 `default_mma_core_sm80.h` 中对 SIMT warp policy 做了这个选择：

```cpp
static const int WarpNumThreadsM = 4;
static const int WarpNumThreadsN = 8;
static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM; // 32 / 4 = 8
static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN; // 64 / 8 = 8
static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;

using Policy = cutlass::gemm::warp::MmaSimtPolicy<
    cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,
    cutlass::layout::RowMajorInterleaved<LaneLayout>,
    LaneMmaShape
>;
```

对于 `WarpShape=32x64x8`：

```text
ThreadTileM = 8
ThreadTileN = 8
LaneLayout = 2
```

所以 CUTLASS 用的是：

```cpp
RowMajorInterleaved<2>
```

`RowMajorInterleaved<2>` 的 inverse 逻辑是：

```cpp
row_major = offset / stride
residual = offset % stride
column = residual / 2
row_minor = residual % 2
row = row_major * 2 + row_minor
```

对于一个 warp 的 32 个 lane，可以简化理解为：

```cpp
lC_row = (lane_id / 16) * 2 + (lane_id % 2);
lC_col = (lane_id / 2) % 8;
```

### 原来我们的 lane layout

原来我们是普通 row-major：

```cpp
lC_row = lane_id / 8;
lC_col = lane_id % 8;
```

这个 mapping 本身能算对，但它和 CUTLASS 的 shared load iterator 不一样。

特别是 B operand 的 shared load：

```cpp
tCrB[...] = FETCH_FLOAT4(
    stage_B_p[(warp_col * kWarpN + ... + lC_col * 4) +
              k_block_next * kSmemStrideB]);
```

B 的地址主要跟 `lC_col` 有关。

普通 row-major 下，`lC_col` 的分布是：

```text
lane:   0 1 2 3 4 5 6 7 8 9 ...
col:    0 1 2 3 4 5 6 7 0 1 ...
```

CUTLASS interleaved<2> 下，`lC_col` 的分布是：

```text
lane:   0 1 2 3 4 5 6 7 8 9 ...
col:    0 0 1 1 2 2 3 3 4 4 ...
```

后者让同一条 `LDS.128` 的 lane 地址组合更适合 shared memory coalescing。

### 现在怎么改

当前代码：

```cpp
int lane_id = tid % 32;
int lC_row = (lane_id >> 4) * kLaneLayoutInterleave +
             (lane_id & (kLaneLayoutInterleave - 1));
int lC_col = (lane_id / kLaneLayoutInterleave) & (kWarpThreadsN - 1);
```

其中：

```cpp
kLaneLayoutInterleave = 2
```

所以它就是：

```cpp
lC_row = (lane_id / 16) * 2 + (lane_id % 2);
lC_col = (lane_id / 2) % 8;
```

### 为什么有效

NCU source page 直接显示：

改前 `LDS.128`：

```text
18 条静态 LDS.128 是 2 wavefront/execution
18 条静态 LDS.128 是 4 wavefront/execution
平均约 3 wavefront/execution
```

改后 `LDS.128`：

```text
36 条静态 LDS.128 全部是 2 wavefront/execution
```

于是：

```text
LDS.128 wavefront: 402,751,488 -> 268,500,992
```

shared 总 wavefront：

```text
604,078,080 -> 469,827,584
```

这就基本对上 CUTLASS 的：

```text
470,876,160
```

所以这个修改的本质不是“少发了 shared load”，而是：

```text
同样数量的 LDS.128，用更好的 lane 地址排列，让每条指令拆成更少 wavefront。
```

## 6. 增加 CUTLASS-like launch 函数

### 改了什么

在 kernel 文件里增加：

```cpp
inline void launch_sgemm_128x128x8stage5(float *A, float *B, float *C,
                                         int M, int N, int K,
                                         cudaStream_t stream = 0)
```

它固定：

```cpp
dim3 block(kThreads); // 256
dim3 grid(tile_m_count * kBlockSwizzle,
          (tile_n_count + kBlockSwizzle - 1) / kBlockSwizzle);
```

并传入：

```cpp
kSharedStorageBytes
```

作为 dynamic shared memory 大小。

### 为什么有效

这样 host 侧不会再手写错误的 grid。

因为 swizzle 之后，grid 不再是简单的：

```cpp
dim3(N / 128, M / 128)
```

而是 CUTLASS 那种：

```cpp
dim3(M_tiles * 8, ceil(N_tiles / 8))
```

如果 host launch 还用旧 grid，kernel 内 swizzle mapping 就会错。

## 7. host 侧增加边界检查

### 改了什么

`src/main.cu` 的 `launch_cutlass_like_stage5` 里检查：

```cpp
M % 128 == 0
N % 128 == 0
K % 8 == 0
K >= 32
```

### 为什么要 `K >= 32`

这个 kernel 是 5-stage pipeline。

prologue 会先 issue：

```cpp
K_PIPE_MAX - 1 = 4
```

个 K tile。

每个 K tile 是 `kCtaK=8`，所以至少要：

```text
4 * 8 = 32
```

如果 K 小于 32，目前 prologue 会不安全。

CUTLASS 的 predicated iterator 可以处理更多边界情况；我们的手写 kernel 目前为了简单和性能，只支持 benchmark 当前关心的整 tile shape。

## 8. 当前还剩哪些差异

虽然主要 memory 指标已经对齐，但还有一些差异：

| 指标 | 当前 cutlass-like | CUTLASS ref |
|---|---:|---:|
| regs/thread | 138 | 129 |
| NCU duration | 约 `20.06 ms` | 约 `19.39 ms` |
| MIO stall | 约 `185,561` | 约 `120,243` |
| Compute throughput | 约 `68.8%` | 约 `74.3%` |

当前 shared wavefront、L2 hit、global request 已经基本对齐，所以剩下差异更可能来自：

- CUTLASS 的 mainloop 指令调度更细
- CUTLASS 的 iterator 指针更新/寄存器生命周期更优
- CUTLASS 的 epilogue 和 accumulator fragment 组织不同
- 我们的手写 C++ 数组/`float4` 形式让 ptxas 保留了更多寄存器

换句话说，现在最明显的结构性问题已经修掉了。
后面再追性能，就不是“大 bug”，而是更细粒度的调度和寄存器分配问题。

## 9. 这几次修改解决的问题对应表

| 修改 | 解决的问题 | NCU 证据 |
|---|---|---|
| `smem_addr()` | `cp.async` shared 地址语义正确 | SASS 能稳定生成 `LDGSTS` |
| `issue_tile_cp_async()` | prologue/mainloop load 行为统一 | 后续 pipeline 修复更清楚 |
| `k_tiles_to_issue` / `k_tiles_to_compute` 分离 | drain 不再重复 load 最后 tile | global tag requests 对齐 ref |
| drain 阶段仍然 `commit_group` | 保持 `wait_group` pipeline 语义 | correctness 稳定，性能恢复 |
| `GemmIdentityThreadblockSwizzle<8>` | 对齐 CUTLASS CTA 调度和 L2 reuse | L2 hit 约 49% -> 约 95% |
| `RowMajorInterleaved<2>` lane layout | 降低 `LDS.128` wavefront | shared wavefront 约 604M -> 约 470M |
| launch 函数 | host 侧不会用错 grid/smem | grid `(256,4,1)` 对齐 ref |
| host shape guard | 避免当前手写 kernel 不支持的边界 | 小 case verify 稳定 |

## 10. 最重要的直觉总结

可以把这次对齐分成三句话：

1. `cp.async` drain 修复：  
   不该 load 的 tile 不要 load，但 pipeline group 还要继续推进。

2. swizzle 修复：  
   CTA 执行顺序会影响 L2 cache reuse；和 CUTLASS 同样的 tile shape，不等于同样的 cache 行为。

3. lane layout 修复：  
   shared load 指令数量一样，不代表 shared 压力一样；lane 地址排列不同，会让同一条 `LDS.128` 被硬件拆成 2 个或 4 个 wavefront。

这就是为什么最后一个 `RowMajorInterleaved<2>` 改动看起来只改了两行 lane id 计算，
但 shared wavefront 直接少了约 134M。

## 11. 2026-05-29: 为什么还比 CUTLASS 慢最后一点

更新后的 `sass.asm` 已经对应当前代码。
这一版的 memory 侧大指标基本对齐了，但 CUTLASS 仍然快一点：

| 指标 | 当前 cutlass-like | CUTLASS ref |
|---|---:|---:|
| Duration | `20.07 ms` | `19.39 ms` |
| Compute throughput / issue busy | `68.8%` | `74.3%` |
| Issued warp per scheduler | `0.70` | `0.75` |
| Eligible warps per scheduler | `1.16` | `1.24` |
| Registers/thread | 138 | 129 |
| L2 hit rate | `95.6%` | `95.5%` |
| L1 tag requests global | `84,410,368` | `84,410,368` |
| Shared wavefront total | `469,827,584` | `470,876,160` |

这说明最后差距已经不是“多读 global”、
也不是“shared wavefront 总量多”。
真正还差在 instruction scheduling：

| stall | 当前 cutlass-like | CUTLASS ref |
|---|---:|---:|
| `stall_mio` | `185k` | `120k` |
| `stall_lg` | `9.6k` | `0.37k` |
| `stall_short_sb` | `39.9k` | `54.7k` |
| `stall_long_sb` | 几乎没有 | `31k` |

看起来 CUTLASS 反而有更多 long scoreboard 和 short scoreboard，
但它的 MIO/LG throttle 少很多，所以整体 issue 更满。

### 11.1 关键证据：同样的访存量，不同的拥挤程度

NCU source page 按 SASS 指令归因后：

| 指令类别 | 当前 cutlass-like | CUTLASS ref |
|---|---:|---:|
| `LDS.128` executed | `134,250,496` | `134,250,496` |
| `LDS.128` shared wavefront | `268,500,992` | `268,500,992` |
| `LDS.128` stall_mio | `135k` | `61k` |
| `LDGSTS` executed | `33,816,576` | `33,816,576` |
| `LDGSTS` shared wavefront | `201,326,592` | `201,326,592` |
| `LDGSTS` stall_mio | `36.7k` | `25.2k` |
| `LDGSTS` stall_lg | `8.0k` | `0.3k` |

这张表的意思很重要：

```text
访存指令数量一样
shared wavefront 数量也一样
但是我们的 MIO/LG throttle 更高
```

所以剩下的问题不是“量”，而是“时间分布”。

可以把它理解成：

- 两个人搬同样多的箱子
- 走同一条路
- 每个箱子也一样重
- 但我们把箱子集中在几秒内一起塞到门口
- CUTLASS 把箱子更均匀地穿插在计算中

门口总通过量一样，但拥堵程度不一样。

### 11.2 我们现在的 cp.async 发射位置还是太集中

当前手写 kernel 在 mainloop 里是：

```cpp
if (k_block == 0) {
  if (k_tiles_to_issue > 0) {
    issue_tile_cp_async(...);
  }
  CP_ASYNC_COMMIT_GROUP();
  smem_pipe_write = smem_pipe_read;
  smem_pipe_read = (smem_pipe_read + 1) % K_PIPE_MAX;
}
```

也就是说，逻辑上下一 stage 的 A/B global-to-shared copy 都挂在
`k_block == 0` 这一个位置。

ptxas 会帮我们把部分 `LDGSTS` 和 FFMA 交错起来，
但它能做的有限。
从 `sass.asm` 看，我们的 steady-state 里 `LDGSTS` 的间隔仍然比较密。
例如一段主循环里，多个 `LDGSTS` 在几十条指令内连续出现。

这会让：

- `LDGSTS` 本身更容易遇到 MIO throttle
- global-to-shared 路径更容易遇到 LG throttle
- 后面的 `LDS.128` 也更容易碰到同一段 MIO 压力

### 11.3 CUTLASS 是怎么分散 cp.async 的

CUTLASS 的 mainloop 在
`cutlass/include/cutlass/gemm/threadblock/mma_multistage.h`。

核心结构是：

```cpp
for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {
  warp_tile_iterator_A_.load(...);
  warp_tile_iterator_B_.load(...);

  warp_mma_(...);

  if (warp_mma_k < Base::kWarpGemmIterations - 1) {
    group_start_iteration_A = warp_mma_k * Detail::kAccessesPerGroupA;
    group_start_iteration_B = warp_mma_k * Detail::kAccessesPerGroupB;
    copy_tiles_and_advance(iterator_A, iterator_B,
                           group_start_iteration_A,
                           group_start_iteration_B);
  }

  if (warp_mma_k + 2 == Base::kWarpGemmIterations) {
    group_start_iteration_A = (warp_mma_k + 1) * Detail::kAccessesPerGroupA;
    group_start_iteration_B = (warp_mma_k + 1) * Detail::kAccessesPerGroupB;
    copy_tiles_and_advance(...);
    cp_async_fence();
    gmem_wait();
  }
}
```

注意这里的关键点：

```text
copy_tiles_and_advance() 不是只在 k=0 做一次。
它随着 warp_mma_k 分组穿插在整个 K=8 的计算过程中。
```

对这个 kernel，`WarpGemmIterations = 8`。
CUTLASS 会把一个 stage 的 cp.async work 切成小组，
塞进多个 `warp_mma_k` 之间。

这就是为什么 CUTLASS 的 SASS 更长：

- 它有更多 pointer/predicate/update 指令
- 也有更多 false-predicated `LDS RZ, [RZ]` 之类的调度填充
- 静态 SASS 行数更多

但它的 issue 反而更好，因为 MIO/LG 压力被摊平了。

### 11.4 为什么 CUTLASS 指令更多却更快

从 NCU 看：

| 指标 | 当前 cutlass-like | CUTLASS ref |
|---|---:|---:|
| Instructions executed | `2.544B` | `2.655B` |
| Issue slots busy | `68.8%` | `74.3%` |
| Duration | `20.07 ms` | `19.39 ms` |

CUTLASS 动态指令数还多一些，但它每个 cycle 能发出更多 warp instruction。

可以理解为：

```text
我们指令少一点，但等待多一点。
CUTLASS 指令多一点，但流水更顺。
```

所以这最后一点不是简单靠“减少 SASS 行数”解决。
事实上，CUTLASS 的 SASS 更长，性能却更好。

### 11.5 下一步如果还想追

下一步最有价值的方向不是继续看 occupancy，
而是把我们的 mainloop 改成更像 CUTLASS：

1. 把 `issue_tile_cp_async()` 拆成多个小组。
2. 不要只在 `k_block == 0` 发完整 stage。
3. 按 `k_block` 分散发 A/B 的若干 `cp.async`。
4. 在倒数第二个 `k_block` 附近做 `commit_group + wait_group + syncthreads + advance stage`。
5. 让 shared load、FFMA、cp.async 三者更均匀交错。

这会让代码复杂很多，因为要维护：

- 当前 stage
- 当前 A/B copy group
- `k_tiles_to_issue`
- drain 空 group
- shared read/write stage rotate
- register prefetch 的先后关系

但方向很明确：

```text
我们现在已经对齐了“访问量”和“地址形态”；
剩下要对齐的是“发射节奏”。
```

