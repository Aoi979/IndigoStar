#pragma once

#include <cute/numeric/numeric_types.hpp>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

inline bool verify_against_reference(const std::vector<float> &host_C,
                                     const std::vector<float> &ref_C) {
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  for (std::size_t i = 0; i < host_C.size(); ++i) {
    const double actual   = static_cast<double>(host_C[i]);
    const double expected = static_cast<double>(ref_C[i]);
    const double abs_error = std::abs(actual - expected);
    const double rel_error = abs_error / std::max(1.0, std::abs(expected));
    max_abs_error = std::max(max_abs_error, abs_error);
    max_rel_error = std::max(max_rel_error, rel_error);
  }
  std::cout << "verify max_abs_error=" << max_abs_error
            << " max_rel_error=" << max_rel_error << '\n';
  return max_abs_error < 1.0e-3 || max_rel_error < 1.0e-3;
}

inline bool verify_half_against_reference(const std::vector<cute::half_t> &host_C,
                                          const std::vector<cute::half_t> &ref_C,
                                          int columns) {
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  std::size_t max_error_index = 0;
  int printed_large_errors = 0;
  for (std::size_t i = 0; i < host_C.size(); ++i) {
    const double actual   = static_cast<double>(host_C[i]);
    const double expected = static_cast<double>(ref_C[i]);
    const double abs_error = std::abs(actual - expected);
    const double rel_error = abs_error / std::max(1.0, std::abs(expected));
    if (abs_error > max_abs_error) {
      max_error_index = i;
    }
    max_abs_error = std::max(max_abs_error, abs_error);
    max_rel_error = std::max(max_rel_error, rel_error);
    if (abs_error > 1.0e-2 && printed_large_errors < 8) {
      std::cout << "verify large_error index=" << i
                << " row=" << (columns > 0 ? i / columns : 0)
                << " col=" << (columns > 0 ? i % columns : i)
                << " actual=" << static_cast<float>(host_C[i])
                << " expected=" << static_cast<float>(ref_C[i])
                << " abs=" << abs_error << '\n';
      ++printed_large_errors;
    }
  }
  std::cout << "verify max_abs_error=" << max_abs_error
            << " max_rel_error=" << max_rel_error
            << " max_index=" << max_error_index
            << " actual=" << static_cast<float>(host_C[max_error_index])
            << " expected=" << static_cast<float>(ref_C[max_error_index])
            << '\n';
  return max_abs_error < 1.0e-3 && max_rel_error < 1.0e-3;
}
