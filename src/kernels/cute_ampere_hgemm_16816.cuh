#pragma once

#include <cute/tensor.hpp>

template <class ElementA,
          class ElementB,
          class SmemLayoutA,
          class SmemLayoutB>
struct CuteHgemmSharedStorage
{
  cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
};

template <class ProblemShape, class CtaTiler, class TA, class AStride,
          class ASmemLayout, class TiledCopyA, class S2RAtomA, class TB,
          class BStride, class BSmemLayout, class TiledCopyB, class S2RAtomB,
          class TC, class CStride, class CSmemLayout, class TiledMma,
          class Alpha, class Beta>
__global__ static __launch_bounds__(decltype(size(
    TiledMma{}))::value) void cute_ampere_hgemm_16816(ProblemShape shape_MNK,
                                          CtaTiler cta_tiler, TA const *A,
                                          AStride dA, ASmemLayout sA_layout,
                                          TiledCopyA copy_a,
                                          S2RAtomA s2r_atom_a, TB const *B,
                                          BStride dB, BSmemLayout sB_layout,
                                          TiledCopyB copy_b,
                                          S2RAtomB s2r_atom_b, TC *C,
                                          CStride dC, CSmemLayout sC_layout, TiledMma mma,
                                          Alpha alpha, Beta beta,
                                          int kBlockSwizzle) {
  using namespace cute;

  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{}); // (M, N, K)
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{}); // (BLK_M, BLK_N, BLK_K)

  CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma)); // NumThreads
  CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma)); // NumThreads

  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);
  static_assert(is_static<CSmemLayout>::value);
  static_assert(cosize_v<CSmemLayout> <= cosize_v<ASmemLayout>);

  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler)); // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler)); // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler)); // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler)); // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler)); // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler)); // BLK_K

  CUTE_STATIC_ASSERT_V(
      congruent(select<0, 2>(shape_MNK), dA)); // dA strides for shape MK
  CUTE_STATIC_ASSERT_V(
      congruent(select<1, 2>(shape_MNK), dB)); // dB strides for shape NK
  CUTE_STATIC_ASSERT_V(
      congruent(select<0, 1>(shape_MNK), dC)); // dC strides for shape MN

  //
  // Full and Tiled Tensors
  //

  // Represent the full tensors
  Tensor mA =
      make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA); // (M,K)
  Tensor mB =
      make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB); // (N,K)
  Tensor mC =
      make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC); // (M,N)

  // Block swizzle: reorder CTA execution for better L2 cache locality.
  // Positive kBlockSwizzle = row-major swizzle (reuse A rows).
  // Negative kBlockSwizzle = column-major swizzle (reuse B cols).
  int const tile_m_max = size(ceil_div(get<0>(shape_MNK), size<0>(cta_tiler)));
  int const tile_n_max = size(ceil_div(get<1>(shape_MNK), size<1>(cta_tiler)));
  int tile_m, tile_n;
  if (kBlockSwizzle > 0) {
    // Row-major: grid.x = tile_m_count * swizzle
    tile_m = blockIdx.x / kBlockSwizzle;
    tile_n = blockIdx.y * kBlockSwizzle + blockIdx.x % kBlockSwizzle;
    if (tile_n >= tile_n_max) return;
  } else {
    // Column-major: grid.y = tile_n_count * swizzle
    int const swiz = -kBlockSwizzle;
    tile_n = blockIdx.y / swiz;
    tile_m = blockIdx.x * swiz + blockIdx.y % swiz;
    if (tile_m >= tile_m_max) return;
  }

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(tile_m, tile_n, _); // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord,
                         Step<_1, X, _1>{}); // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord,
                         Step<X, _1, _1>{}); // (BLK_N,BLK_K,k)
  Tensor gC =
      local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (BLK_M,BLK_N)

  // Shared memory buffers
  extern __shared__ char shared_memory[];
  using MainloopSharedStorage =
      CuteHgemmSharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
  MainloopSharedStorage &smem =
      *reinterpret_cast<MainloopSharedStorage *>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()),
                          sA_layout); // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()),
                          sB_layout); // (BLK_N,BLK_K,PIPE)

  //
  // Partition the copying of A and B tiles across the threads
  //

  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY,CPY_M,CPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA); // (CPY,CPY_M,CPY_K,PIPE)

  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB); // (CPY,CPY_N,CPY_K,k)
  Tensor tBsB = thr_copy_b.partition_D(sB); // (CPY,CPY_N,CPY_K,PIPE)

  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA)); // CPY_M
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA)); // CPY_K
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB)); // CPY_N
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB)); // CPY_K

  //
  // PREFETCH
  //

  auto K_PIPE_MAX = size<3>(tAsA);

  // Total count of tiles
  int K_TILE_MAX = size<3>(tAgA);
  int k_tiles_to_issue = K_TILE_MAX;
  int k_tiles_to_compute = K_TILE_MAX;
  // Current tile index in gmem to read from
  int k_tile_next = 0;

  // Start async loads for all pipes but the last
  CUTE_UNROLL
  for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe) {
    if (k_tiles_to_issue > 0) {
      copy(copy_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, k_pipe));
      copy(copy_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, k_pipe));
      cp_async_fence();
      --k_tiles_to_issue;
      ++k_tile_next;
    }
  }


  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCgC = thr_mma.partition_C(gC); // (MMA,MMA_M,MMA_N)

  // Allocate registers for pipelining
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0)); // (MMA,MMA_M,MMA_K)
  Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0)); // (MMA,MMA_N,MMA_K)
  // Allocate the accumulators -- same size as the projected data
  Tensor tCrC = thr_mma.make_fragment_C(tCgC); // (MMA,MMA_M,MMA_N)

  CUTE_STATIC_ASSERT_V(
      (shape(tCrC) == take<0, 3>(shape(tCgC))));          // (MMA,MMA_M,MMA_N)
  CUTE_STATIC_ASSERT_V((size<1>(tCgC) == size<1>(tCrA))); // MMA_M
  CUTE_STATIC_ASSERT_V((size<2>(tCgC) == size<1>(tCrB))); // MMA_N

  // Clear the accumulators
  clear(tCrC);

  //
  // Copy Atom retiling
  //

  TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
  ThrCopy s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
  Tensor tXsA = s2r_thr_copy_a.partition_S(sA); // (CPY,MMA_M,MMA_K,PIPE)
  Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);  // (CPY,MMA_M,MMA_K)

  TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
  ThrCopy s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
  Tensor tXsB = s2r_thr_copy_b.partition_S(sB); // (CPY,MMA_N,MMA_K,PIPE)
  Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);  // (CPY,MMA_N,MMA_K)

  // Current pipe index in smem to read from
  int smem_pipe_read = 0;
  // Current pipe index in smem to write to
  int smem_pipe_write = K_PIPE_MAX - 1;

  // Pipe slice
  Tensor tXsA_p = tXsA(_, _, _, smem_pipe_read);
  Tensor tXsB_p = tXsB(_, _, _, smem_pipe_read);

  // Size of the register pipeline
  auto K_BLOCK_MAX = size<2>(tCrA);
  CUTE_STATIC_ASSERT_V(K_BLOCK_MAX == size<2>(tXrA));

  // PREFETCH register pipeline
  if (K_BLOCK_MAX > 1) {
    // Wait until our first prefetched tile is loaded in
    cp_async_wait<K_PIPE_MAX - 2>();
    __syncthreads();

    // Prefetch the first rmem from the first k-block
    copy(s2r_atom_a, tXsA_p(_, _, Int<0>{}), tXrA(_, _, Int<0>{}));
    copy(s2r_atom_b, tXsB_p(_, _, Int<0>{}), tXrB(_, _, Int<0>{}));
  }

  CUTE_NO_UNROLL
  while (k_tiles_to_compute > 0) {
    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
      if (k_block == K_BLOCK_MAX - 1) {
        // Slice the smem_pipe_read smem
        tXsA_p = tXsA(_, _, _, smem_pipe_read);
        tXsB_p = tXsB(_, _, _, smem_pipe_read);

        // Wait for the next tile to land in smem
        cp_async_wait<K_PIPE_MAX - 2>();
        __syncthreads();
      }

      // Load A, B shmem->regs for k_block+1
      auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX; // static
      copy(s2r_atom_a, tXsA_p(_, _, k_block_next), tXrA(_, _, k_block_next));
      copy(s2r_atom_b, tXsB_p(_, _, k_block_next), tXrB(_, _, k_block_next));

      // Spread cp.async across k-blocks to lower ldgsts density.
      // A and B are issued in separate k-blocks; fence + pipe advance
      // are unconditional so the pipeline drains correctly even after
      // the last real tile has been issued.
      if (k_block == 0) {
        if (k_tiles_to_issue > 0) {
          copy(copy_a, tAgA(_, _, _, k_tile_next),
               tAsA(_, _, _, smem_pipe_write));
        }
      }
      if (k_block == 1) {
        if (k_tiles_to_issue > 0) {
          copy(copy_b, tBgB(_, _, _, k_tile_next),
               tBsB(_, _, _, smem_pipe_write));
        }
        cp_async_fence();
        if (k_tiles_to_issue > 0) {
          --k_tiles_to_issue;
          ++k_tile_next;
        }
      }
      if (k_block == K_BLOCK_MAX - 2) {
        smem_pipe_write = smem_pipe_read;
        smem_pipe_read =
            (smem_pipe_read == K_PIPE_MAX - 1) ? 0 : smem_pipe_read + 1;
      }
      // Thread-level register gemm for k_block
      gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
    }
    --k_tiles_to_compute;
  }

  //
  // Epilogue
  //

  cp_async_wait<0>();
  __syncthreads();

  (void)alpha;
  (void)beta;

  // Reuse the mainloop A smem buffer: it is large enough for the 128x128
  // half output tile and the mainloop no longer needs it.
  Tensor sC = make_tensor(make_smem_ptr(reinterpret_cast<TC *>(smem.A.begin())),
                          sC_layout); // (BLK_M,BLK_N)

  auto r2s_copy = make_tiled_copy_C(cute::Copy_Atom<DefaultCopy, TC>{}, mma);
  auto r2s_thr_copy = r2s_copy.get_slice(threadIdx.x);
  Tensor tCrC_r2s = r2s_thr_copy.retile_S(tCrC);
  Tensor tCsC_r2s = r2s_thr_copy.partition_D(sC);
  copy(r2s_copy, tCrC_r2s, tCsC_r2s);

  __syncthreads();

  constexpr int kBlockM = decltype(size<0>(CSmemLayout{}))::value;
  constexpr int kBlockN = decltype(size<1>(CSmemLayout{}))::value;
  constexpr int kVecElems = 8; // 8 half values = 16 B
  int const problem_n = int(get<1>(shape_MNK));
  int const smem_stride_c = int(stride<0>(sC_layout));
  // Swizzled tile indices already computed above; convert to pixel coords.
  int const gmem_m = tile_m * kBlockM;
  int const gmem_n = tile_n * kBlockN;
  TC *sC_ptr = reinterpret_cast<TC *>(smem.A.begin());

  CUTE_NO_UNROLL
  for (int vec = int(threadIdx.x); vec < kBlockM * kBlockN / kVecElems;
       vec += int(blockDim.x)) {
    int const row = vec / (kBlockN / kVecElems);
    int const col = (vec % (kBlockN / kVecElems)) * kVecElems;
    uint4 const *src =
        reinterpret_cast<uint4 const *>(&sC_ptr[row * smem_stride_c + col]);
    uint4 *dst =
        reinterpret_cast<uint4 *>(&C[(gmem_m + row) * problem_n + gmem_n + col]);
    *dst = *src;
  }
}
