# CUDA Kernel Resource Usage Report

Generated automatically after each build.

| Kernel | Arch | REG | Shared Mem | Local Mem | Constant Mem | Stack |
|--------|------|-----|------------|-----------|--------------|-------|
| `_ZN7cutlass13device_kernelINS_4gemm6kernel13GemmUniversalIN4cute5tupleIJiiiiEEENS1_10collective13CollectiveMmaINS1_34MainloopSm90TmaGmmaWarpSpecializedILi6ENS5_IJNS4_1CILi2EEENSA_ILi1EEESC_EEENS1_32KernelTmaWarpSpecializedPingpongEEENS5_IJNSA_ILi128EEESG_NSA_ILi64EEEEEENS_6half_tENS5_IJlSC_lEEESJ_SK_NS4_8TiledMMAINS4_8MMA_AtomIJNS4_4SM904GMMA26MMA_64x128x16_F32F16F16_SSILNSO_5MajorE0ELSQ_0ELNSO_7ScaleInE1ELSR_1EEEEEENS4_6LayoutINS5_IJSC_SC_SC_EEENS5_IJNSA_ILi0EEESW_SW_EEEEENS5_IJNS4_10UnderscoreESZ_SZ_EEEEENS4_13SM90_TMA_LOADENS4_14ComposedLayoutINS4_7SwizzleILi3ELi4ELi3EEENS4_18smem_ptr_flag_bitsILi16EEENSU_INS5_IJNSA_ILi8EEESH_EEENS5_IJSH_SC_EEEEEEEvNS4_8identityENS4_23SM90_TMA_LOAD_MULTICASTES1C_vS1D_EENS_8epilogue10collective18CollectiveEpilogueINS1G_22Sm90TmaWarpSpecializedILi4ELi2ELi16ELb0ELb1EEEJSI_NS5_IJSH_NSA_ILi32EEEEEEvSK_SJ_SK_NS1G_6fusion15FusionCallbacksIS1K_NS1N_17LinearCombinationISJ_fvfLNS_15FloatRoundStyleE2EEESI_S1M_JEEES12_NS13_INS14_ILi2ELi4ELi3EEES17_NSU_INS5_IJS18_S1L_EEENS5_IJS1L_SC_EEEEEEENS4_17SM75_U32x4_LDSM_NENS4_14SM90_TMA_STOREES1X_NS4_17SM90_U32x4_STSM_NENS4_9Copy_AtomIJS20_SJ_EEEvEEENS1_19PersistentSchedulerEvEEEEvNT_6ParamsE` | sm_90 | 168 | 1024 | 0 | 2432 | 0 |
| `_ZN7cutlass13device_kernelINS_4gemm6kernel13GemmUniversalIN4cute5tupleIJiiiiEEENS1_10collective13CollectiveMmaINS1_34MainloopSm90TmaGmmaWarpSpecializedILi6ENS5_IJNS4_1CILi2EEENSA_ILi1EEESC_EEENS1_35KernelTmaWarpSpecializedCooperativeEEENS5_IJNSA_ILi128EEESG_NSA_ILi64EEEEEENS_6half_tENS5_IJlSC_lEEESJ_SK_NS4_8TiledMMAINS4_8MMA_AtomIJNS4_4SM904GMMA26MMA_64x128x16_F32F16F16_SSILNSO_5MajorE0ELSQ_0ELNSO_7ScaleInE1ELSR_1EEEEEENS4_6LayoutISD_NS5_IJSC_NSA_ILi0EEESV_EEEEENS5_IJNS4_10UnderscoreESY_SY_EEEEENS4_13SM90_TMA_LOADENS4_14ComposedLayoutINS4_7SwizzleILi3ELi4ELi3EEENS4_18smem_ptr_flag_bitsILi16EEENSU_INS5_IJNSA_ILi8EEESH_EEENS5_IJSH_SC_EEEEEEEvNS4_8identityENS4_23SM90_TMA_LOAD_MULTICASTES1B_vS1C_EENS_8epilogue10collective18CollectiveEpilogueINS1F_22Sm90TmaWarpSpecializedILi4ELi2ELi16ELb0ELb1EEEJSI_NS5_IJSG_NSA_ILi32EEEEEEvSK_SJ_SK_NS1F_6fusion15FusionCallbacksIS1J_NS1M_17LinearCombinationISJ_fvfLNS_15FloatRoundStyleE2EEESI_S1L_JEEES11_NS12_INS13_ILi2ELi4ELi3EEES16_NSU_INS5_IJS17_S1K_EEENS5_IJS1K_SC_EEEEEEENS4_17SM75_U32x4_LDSM_NENS4_14SM90_TMA_STOREES1W_NS4_17SM90_U32x4_STSM_NENS4_9Copy_AtomIJS1Z_SJ_EEEvEEENS1_19PersistentSchedulerEvEEEEvNT_6ParamsE` | sm_90 | 168 | 1024 | 0 | 2432 | 0 |
| `sgemm_naive(float*, float*, float*, int, int, int)` | sm_90 | 32 | 1024 | 0 | 564 | 0 |
| `sm90_hgemm_pingpong::hgemm_pingpong_kernel(CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, int, int, int, sm90_hgemm_pingpong::PersistentTileSchedulerSm90Params)` | sm_90 | 168 | 1152 | 0 | 1096 | 0 |

## Legend

- **REG**: Number of registers used per thread
- **Shared Mem**: Static shared memory usage in bytes
- **Local Mem**: Local memory usage in bytes
- **Constant Mem**: Constant memory usage in bytes
- **Stack**: Stack memory usage in bytes
