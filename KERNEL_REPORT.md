# CUDA Kernel Resource Usage Report

Generated automatically after each build.

| Kernel | Arch | REG | Shared Mem | Local Mem | Constant Mem | Stack |
|--------|------|-----|------------|-----------|--------------|-------|
| `custom::trial::sgemm_128x128x32_trial(int, int, int, float const*, float const*, float*)` | sm_89 | 128 | 0 | 0 | 392 | 0 |
| `dev::sgemm_128x128x32_double_buffer(int, int, int, float const*, float const*, float*)` | sm_89 | 151 | 0 | 0 | 392 | 0 |
| `external::double_buffer::sgemm(int, int, int, float const*, float const*, float*)` | sm_89 | 118 | 32768 | 0 | 392 | 0 |
| `external::no_double_buffer::sgemm(int, int, int, float const*, float const*, float*)` | sm_89 | 128 | 16384 | 0 | 392 | 0 |
| `kimi::sgemm_128x128x32_double_buffer(int, int, int, float const*, float const*, float*)` | sm_89 | 120 | 0 | 0 | 392 | 0 |
| `low_occupancy::sgemm_128x128x32_double_buffer(int, int, int, float const*, float const*, float*)` | sm_89 | 219 | 0 | 0 | 392 | 0 |
| `sgemm_128x128x32(int, int, int, float const*, float const*, float*)` | sm_89 | 128 | 32768 | 0 | 392 | 0 |
| `sgemm_128x128x32_K2(int, int, int, float const*, float const*, float*)` | sm_89 | 128 | 32768 | 0 | 392 | 0 |
| `sgemm_128x128x32_K4(int, int, int, float const*, float const*, float*)` | sm_89 | 127 | 32768 | 0 | 392 | 0 |
| `sgemm_naive(float*, float*, float*, int, int, int)` | sm_89 | 40 | 0 | 0 | 388 | 0 |

## Legend

- **REG**: Number of registers used per thread
- **Shared Mem**: Static shared memory usage in bytes
- **Local Mem**: Local memory usage in bytes
- **Constant Mem**: Constant memory usage in bytes
- **Stack**: Stack memory usage in bytes
