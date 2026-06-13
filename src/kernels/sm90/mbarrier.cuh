
__device__ __forceinline__ void init_barrier(uint64_t *bar, int arrive_count) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar_ptr),
               "r"(arrive_count)
               : "memory");
}

__device__ __forceinline__ void wait_barrier(uint64_t *bar, int phase) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("{\n"
               ".reg .pred p;\n"
               "wait_again:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
               "@p bra.uni wait_done;\n"
               "bra.uni wait_again;\n"
               "wait_done:\n"
               "}\n" ::"r"(bar_ptr),
               "r"(phase));
}

__device__ __forceinline__ void arrive_barrier(uint64_t *bar, int count = 1) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n" ::"r"(
          bar_ptr),
      "r"(count)
      : "memory");
}

__device__ __forceinline__ void expect_tma_bytes(uint64_t *bar,
                                                 uint32_t bytes) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar_ptr),
      "r"(bytes)
      : "memory");
}

__device__ __forceinline__ void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void arrive_barrier_remote(uint64_t *bar,
                                                      uint32_t dst_cta_rank) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("{\n"
               ".reg .b32 remote_bar;\n"
               "mapa.shared::cluster.u32 remote_bar, %0, %1;\n"
               "mbarrier.arrive.shared::cluster.b64 _, [remote_bar];\n"
               "}\n" ::"r"(bar_ptr),
               "r"(dst_cta_rank)
               : "memory");
}
