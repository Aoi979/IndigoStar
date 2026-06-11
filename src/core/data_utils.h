#pragma once

#include <cute/numeric/numeric_types.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

inline std::size_t matrix_elements(int rows, int cols) {
  return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
}

inline void fill_random(std::vector<float> *values, int fan_in, std::mt19937 *rng) {
  const float limit = 1.0F / std::sqrt(static_cast<float>(fan_in));
  std::uniform_real_distribution<float> dist(-limit, limit);
  for (float &value : *values) value = dist(*rng);
}

inline double checksum(const std::vector<float> &values) {
  double sum = 0.0;
  for (float v : values) sum += static_cast<double>(v);
  return sum;
}

inline std::vector<float> compute_reference(int m, int n, int k,
                                            const std::vector<float> &A,
                                            const std::vector<float> &B) {
  std::vector<float> C(matrix_elements(m, n), 0.0F);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      double acc = 0.0;
      for (int l = 0; l < k; ++l) {
        acc += static_cast<double>(A[i * k + l]) *
               static_cast<double>(B[l * n + j]);
      }
      C[i * n + j] = static_cast<float>(acc);
    }
  }
  return C;
}

inline double checksum_half(const std::vector<cute::half_t> &values) {
  double sum = 0.0;
  for (cute::half_t v : values) sum += static_cast<double>(v);
  return sum;
}

inline std::vector<cute::half_t> compute_reference_half(
    int m, int n, int k,
    const std::vector<cute::half_t> &A,
    const std::vector<cute::half_t> &B) {
  std::vector<cute::half_t> C(matrix_elements(m, n), cute::half_t(0.0F));
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      double acc = 0.0;
      for (int l = 0; l < k; ++l) {
        acc += static_cast<double>(A[i * k + l]) *
               static_cast<double>(B[l * n + j]);
      }
      C[i * n + j] = cute::half_t(static_cast<float>(acc));
    }
  }
  return C;
}
