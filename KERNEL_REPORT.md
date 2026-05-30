# CUDA Kernel Resource Usage Report

Generated automatically after each build.

| Kernel | Arch | REG | Shared Mem | Local Mem | Constant Mem | Stack |
|--------|------|-----|------------|-----------|--------------|-------|
| `_Z23cute_ampere_hgemm_16816IN4cute5tupleIJiiiEEENS1_IJNS0_1CILi128EEES4_NS3_ILi64EEEEEEN7cutlass6half_tENS1_IJiNS3_ILi1EEEEEENS0_14ComposedLayoutINS0_7SwizzleILi3ELi3ELi3EEENS3_ILi0EEENS0_6LayoutINS1_IJNS1_IJNS3_ILi8EEENS3_ILi16EEEEEENS1_IJNS1_IJSG_SG_EEES9_EEENS1_IJS9_NS3_ILi3EEEEEEEEENS1_IJNS1_IJSG_NS3_ILi512EEEEEENS1_IJNS1_IJS9_S5_EEESE_EEENS1_IJSE_NS3_ILi8192EEEEEEEEEEEEENS0_9TiledCopyINS0_9Copy_AtomIJNS0_25SM80_CP_ASYNC_CACHEALWAYSINS7_9uint128_tES10_EES8_EEENSF_INS1_IJSI_SG_EEENS1_IJNS1_IJS4_S9_EEESH_EEEEENS1_IJSH_S5_EEEEENSY_IJNS0_17SM75_U32x4_LDSM_NES8_EEES8_NS1_IJS9_iEEENSB_ISD_SE_NSF_INS1_IJNS1_IJSI_S9_EEESJ_SM_EEENS1_IJSR_NS1_IJSG_NS3_ILi1024EEEEEEST_EEEEEEENSX_IS12_NSF_INS1_IJS4_SG_EEENS1_IJSG_S9_EEEEES1J_EENSY_IJNS0_17SM75_U16x8_LDSM_TES8_EEES8_SA_NSF_INS1_IJS4_S4_EEENS1_IJNS3_ILi136EEES9_EEEEENS0_8TiledMMAINS0_8MMA_AtomIJNS0_28SM80_16x8x16_F16F16F16F16_TNEEEENSF_INS1_IJNS3_ILi2EEES1X_S9_EEENS1_IJS9_S1X_SE_EEEEENS1_IJNS3_ILi32EEES21_SH_EEEEEffEvT_T0_PKT1_T2_T3_T4_T5_PKT6_T7_T8_T9_T10_PT11_T12_T13_T14_T15_T16_i` | sm_89 | 168 | 0 | 0 | 428 | 0 |
| `_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock13MmaMultistageINS1_9GemmShapeILi128ELi128ELi8EEENS_9transform11threadblock28PredicatedTileAccessIteratorINS_11MatrixShapeILi128ELi8EEEfNS_6layout8RowMajorELi1ENS8_30PitchLinearStripminedThreadMapINS_16PitchLinearShapeILi8ELi128EEELi256ELi1EEENS_5ArrayIfLi1ELb1EEELb0ENSD_9NoPermuteEEENS9_25RegularTileAccessIteratorISC_fNSD_11ColumnMajorELi0ENS8_33TransposePitchLinearThreadMapSimtISI_EELi4EEELNS_4arch14CacheOperation4KindE0ENSA_INSB_ILi8ELi128EEEfSE_Li0ENSF_INSG_ILi128ELi8EEELi256ELi1EEESK_Lb0ESL_EENSN_ISV_fSE_Li1ESX_Li4EEELSU_0EfSE_NS4_9MmaPolicyINS1_4warp7MmaSimtINS6_ILi32ELi64ELi8EEEfSO_fSE_fSE_NS11_13MmaSimtPolicyINSB_ILi4ELi8EEENSD_19RowMajorInterleavedILi2EEENS6_ILi4ELi4ELi1EEEEELi1ELNS_16ComplexTransformE0ELS1A_0EbEENSB_ILi0ELi0EEES1C_Li1EEELi5ELNS1_23SharedMemoryClearOptionE0EbEENS_8epilogue11threadblock8EpilogueIS7_S1B_Li1ENS1H_22PredicatedTileIteratorINS1H_26OutputTileOptimalThreadMapINS1H_15OutputTileShapeILi128ELi1ELi4ELi4ELi1EEENS1L_ILi1ELi4ELi2ELi1ELi8EEELi256ELi1ELi32EEEfLb0ESL_Lb0EEENS1G_4warp20FragmentIteratorSimtIS13_NS1_6thread3MmaINS6_ILi8ELi8ELi1EEEfSO_fSE_fSE_NSS_13OpMultiplyAddEbEESE_S19_EENS1Q_16TileIteratorSimtIS13_S1W_fSE_S19_EENS1H_18SharedLoadIteratorINS1O_18CompactedThreadMapEfLi4EEENS1G_6thread17LinearCombinationIfLi1EffLNS23_9ScaleType4KindE0ELNS_15FloatRoundStyleE2EfEENSB_ILi0ELi17EEELi1ELi1EEENS4_30GemmIdentityThreadblockSwizzleILi8EEELb0EEEEEvNT_6ParamsE` | sm_89 | 129 | 0 | 0 | 720 | 0 |
| `dev::sgemm_128x128x32_double_buffer(int, int, int, float const*, float const*, float*)` | sm_89 | 130 | 0 | 0 | 392 | 0 |
| `external::double_buffer::sgemm(int, int, int, float const*, float const*, float*)` | sm_89 | 128 | 32768 | 0 | 392 | 0 |
| `external::no_double_buffer::sgemm(int, int, int, float const*, float const*, float*)` | sm_89 | 128 | 16384 | 0 | 392 | 0 |
| `sgemm_128x128x32(int, int, int, float const*, float const*, float*)` | sm_89 | 128 | 32768 | 0 | 392 | 0 |
| `sgemm_naive(float*, float*, float*, int, int, int)` | sm_89 | 40 | 0 | 0 | 388 | 0 |
| `void cutlass_like::sgemm_128x128x8stage5_kernel<false, false, false>(float*, float*, float*, int, int, int)` | sm_89 | 128 | 0 | 0 | 388 | 0 |
| `void cutlass_like::sgemm_128x128x8stage5_kernel<false, false, true>(float*, float*, float*, int, int, int)` | sm_89 | 140 | 0 | 0 | 388 | 0 |
| `void cutlass_like::sgemm_128x128x8stage5_kernel<false, true, false>(float*, float*, float*, int, int, int)` | sm_89 | 134 | 0 | 0 | 388 | 0 |
| `void cutlass_like::sgemm_128x128x8stage5_kernel<true, false, false>(float*, float*, float*, int, int, int)` | sm_89 | 140 | 0 | 0 | 388 | 0 |
| `void cutlass_like::sgemm_128x128x8stage5_kernel<true, true, false>(float*, float*, float*, int, int, int)` | sm_89 | 134 | 0 | 0 | 388 | 0 |

## Legend

- **REG**: Number of registers used per thread
- **Shared Mem**: Static shared memory usage in bytes
- **Local Mem**: Local memory usage in bytes
- **Constant Mem**: Constant memory usage in bytes
- **Stack**: Stack memory usage in bytes
