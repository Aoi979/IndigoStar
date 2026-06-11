#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstddef>
#include <iostream>

inline bool cuda_check(cudaError_t error, const char *call) {
  if (error == cudaSuccess) return true;
  std::cerr << call << " failed: " << cudaGetErrorString(error) << '\n';
  return false;
}

#define CUDA_CHECK(expr) cuda_check((expr), #expr)

template <typename T>
class DeviceBuffer {
public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }

  bool allocate(std::size_t elements, const char *name) {
    return cuda_check(cudaMalloc(&ptr_, elements * sizeof(T)), name);
  }
  T *get() const { return ptr_; }
private:
  T *ptr_ = nullptr;
};

class CudaEventTimer {
public:
  CudaEventTimer() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }
  ~CudaEventTimer() {
    if (start_) cudaEventDestroy(start_);
    if (stop_) cudaEventDestroy(stop_);
  }
  CudaEventTimer(const CudaEventTimer &) = delete;
  CudaEventTimer &operator=(const CudaEventTimer &) = delete;

  bool record_start() { return CUDA_CHECK(cudaEventRecord(start_)); }
  bool record_stop_and_sync() {
    return CUDA_CHECK(cudaEventRecord(stop_)) &&
           CUDA_CHECK(cudaEventSynchronize(stop_));
  }
  bool elapsed_ms(float *out) const {
    return CUDA_CHECK(cudaEventElapsedTime(out, start_, stop_));
  }
private:
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

class CuBlasHandle {
public:
  CuBlasHandle() { cublasCreate(&handle_); }
  ~CuBlasHandle() { if (handle_) cublasDestroy(handle_); }
  CuBlasHandle(const CuBlasHandle &) = delete;
  CuBlasHandle &operator=(const CuBlasHandle &) = delete;
  bool ok() const { return handle_ != nullptr; }
  cublasHandle_t get() const { return handle_; }
private:
  cublasHandle_t handle_ = nullptr;
};

inline bool require_sm90_device(std::string_view name) {
  int device = 0;
  cudaDeviceProp props{};
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) {
    std::cerr << name << " cudaGetDevice failed: "
              << cudaGetErrorString(err) << '\n';
    return false;
  }
  err = cudaGetDeviceProperties(&props, device);
  if (err != cudaSuccess) {
    std::cerr << name << " cudaGetDeviceProperties failed: "
              << cudaGetErrorString(err) << '\n';
    return false;
  }
  if (props.major < 9) {
    std::cerr << name << " requires an SM90+ GPU. Current device is "
              << props.major << "." << props.minor << ".\n";
    return false;
  }
  return true;
}
