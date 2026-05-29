#pragma once

#include "kernels/sgemm_common.cuh"
#include "kernels/sgemm_naive.cuh"
#include "kernels/sgemm_128x128x32.cuh"
#include "kernels/sgemm_128x128x32_double_buffer_dev.cuh"
#include "kernels/sgemm_external_128x128x16.cuh"
#include "kernels/cutlass_like_sgemm_128x128x8stage5.cuh"
#include "kernels/cutlass_ref_sgemm_128x128x8stage5.cuh"
