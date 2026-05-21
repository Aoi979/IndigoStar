#pragma once

#include "kernels/sgemm_common.cuh"
#include "kernels/sgemm_naive.cuh"
#include "kernels/sgemm_128x128x32.cuh"
#include "kernels/sgemm_128x128x32_trial.cuh"
#include "kernels/sgemm_128x128x32_double_buffer_low_occupancy.cuh"
#include "kernels/sgemm_128x128x32_double_buffer_dev.cuh"
#include "kernels/sgemm_128x128x32_double_buffer_kimi.cuh"
#include "kernels/sgemm_128x128x32_K4.cuh"
#include "kernels/sgemm_128x128x32_K2.cuh"
#include "kernels/sgemm_external_128x128x16.cuh"
