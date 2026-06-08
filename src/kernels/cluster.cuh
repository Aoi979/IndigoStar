#pragma once

__device__ void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" : :);
}

__device__ void cluster_arrive() {
  asm volatile("barrier.cluster.arrive.aligned;\n" : :);
}

__device__ void cluster_wait() {
  asm volatile("barrier.cluster.wait.aligned;\n" : :);
}

__device__ void cluster_sync() {
  cluster_arrive();
  cluster_wait();
}

// Returns the dim3 grid size in terms of number of clusters.
__device__ dim3 cluster_grid_dims() {
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%nclusterid.x;\n" : "=r"(x) :);
  asm volatile("mov.u32 %0, %%nclusterid.y;\n" : "=r"(y) :);
  asm volatile("mov.u32 %0, %%nclusterid.z;\n" : "=r"(z) :);
  return {x, y, z};
}

// Returns the dim3 cluster rank in the grid.
__device__ dim3 cluster_id_in_grid() {
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%clusterid.x;\n" : "=r"(x) :);
  asm volatile("mov.u32 %0, %%clusterid.y;\n" : "=r"(y) :);
  asm volatile("mov.u32 %0, %%clusterid.z;\n" : "=r"(z) :);
  return {x, y, z};
}

// Returns the relative dim3 block rank local to the cluster.
__device__ dim3 block_id_in_cluster() {
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%cluster_ctaid.x;\n" : "=r"(x) :);
  asm volatile("mov.u32 %0, %%cluster_ctaid.y;\n" : "=r"(y) :);
  asm volatile("mov.u32 %0, %%cluster_ctaid.z;\n" : "=r"(z) :);
  return {x, y, z};
}

// Returns the dim3 cluster shape.
__device__ dim3 cluster_shape() {
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%cluster_nctaid.x;\n" : "=r"(x) :);
  asm volatile("mov.u32 %0, %%cluster_nctaid.y;\n" : "=r"(y) :);
  asm volatile("mov.u32 %0, %%cluster_nctaid.z;\n" : "=r"(z) :);
  return {x, y, z};
}

// Get 1D ctaid in a cluster.
__device__ uint32_t block_rank_in_cluster() {
  uint32_t rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank) :);
  return rank;
}

// Set the destination block-ID in cluster for a given SMEM Address
__device__ uint32_t set_block_rank(uint32_t smemAddr, uint32_t rank) {
  uint32_t result;
  asm volatile("mapa.shared::cluster.u32  %0, %1, %2;\n"
               : "=r"(result)
               : "r"(smemAddr), "r"(rank));
  return result;
}

// Elect one thread in the warp. The elected thread gets its predicate set to
// true, all others obtain false.
__device__ uint32_t elect_one_sync() {
  uint32_t pred = 0;
  uint32_t laneid = 0;
  asm volatile("{\n"
               ".reg .b32 %%rx;\n"
               ".reg .pred %%px;\n"
               "     elect.sync %%rx|%%px, %2;\n"
               "@%%px mov.s32 %1, 1;\n"
               "     mov.s32 %0, %%rx;\n"
               "}\n"
               : "+r"(laneid), "+r"(pred)
               : "r"(0xFFFFFFFF));
  return pred;
}

struct ElectOneLaneIdReturnType {
  uint32_t is_leader;
  uint32_t leader_lane_id;
};

__device__ ElectOneLaneIdReturnType elect_one_leader_sync() {
  uint32_t pred = 0;
  uint32_t laneid = 0;
  asm volatile("{\n"
               ".reg .b32 %%rx;\n"
               ".reg .pred %%px;\n"
               "     elect.sync %%rx|%%px, %2;\n"
               "@%%px mov.s32 %1, 1;\n"
               "     mov.s32 %0, %%rx;\n"
               "}\n"
               : "+r"(laneid), "+r"(pred)
               : "r"(0xFFFFFFFF));
  return {pred, laneid};
}

// Store value to remote shared memory in the cluster
__device__ void store_shared_remote(uint32_t value, uint32_t smem_addr,
                                    uint32_t mbarrier_addr,
                                    uint32_t dst_cta_rank) {
  uint32_t dsmem_addr = set_block_rank(smem_addr, dst_cta_rank);
  uint32_t remote_barrier_addr = set_block_rank(mbarrier_addr, dst_cta_rank);
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.u32 "
               "[%0], %1, [%2];"
               :
               : "r"(dsmem_addr), "r"(value), "r"(remote_barrier_addr));
}
