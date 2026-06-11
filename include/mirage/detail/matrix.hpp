#pragma once

#include <algorithm>
#include <array>
#include <eve/arch/cpu/wide.hpp>
#include <eve/conditional.hpp>
#include <eve/module/core.hpp>
#include <eve/module/core/regular/fnma.hpp>
#include <vector>

#include "utils.hpp"

namespace mirage::detail {
namespace matrix {

enum class Variant : uint8_t { FMA = 0, FNMA, EMA, SQ_EMA, NEG, SCALE };

template <typename T>
struct Dims {
  static constexpr int x_height = UNROLL_FACTOR;
  static constexpr int y_height = std::max(1, UNROLL_FACTOR / 2);
  static constexpr int vec_size = eve::wide<T>::size();
  static constexpr int arr_size = x_height * y_height;
};

template <typename T>
inline eve::wide<T> mload(const T* ptr, int valid) {
  constexpr int width = eve::wide<T>::size();
  if (valid >= width) return eve::wide<T>(ptr);

  alignas(eve::wide<T>) T buf[width] = {};
  for (int k = 0; k < valid; ++k) buf[k] = ptr[k];
  return eve::wide<T>(&buf[0]);
}

template <typename T, typename F>
inline void tile_loop(int M, int N, int x_chunk, int y_chunk, int x_off, int y_off, F&& body) {
  using D = Dims<T>;
  x_chunk = std::min(x_chunk, M - x_off);
  y_chunk = std::min(y_chunk, N - y_off);
  for (int i = 0; i < x_chunk; i += D::x_height) {
    int i_rem = std::min(D::x_height, x_chunk - i);
    for (int j = 0; j < y_chunk; j += D::y_height * D::vec_size) {
      int j_rem = std::min(D::y_height * D::vec_size, y_chunk - j);
      body(i, j, i_rem, j_rem);
    }
  }
}

template <typename T, typename F>
inline void for_each_elem(
  int i, int j, int i_rem, int j_rem, int stride, int x_off, int y_off, F&& f
) {
  using D = Dims<T>;
  unroll<D::arr_size>([&]<int e>() {
    constexpr int row = e % D::x_height;
    constexpr int col = e / D::x_height;
    if (row < i_rem && col * D::vec_size < j_rem) {
      auto valid = std::min(D::vec_size, j_rem - col * D::vec_size);
      f(e, (x_off + i + row) * stride + y_off + j + col * D::vec_size, valid);
    }
  });
}

template <typename T, typename RowLoad, typename ColLoad>
inline auto matmul_acc(int K, RowLoad&& row_load, ColLoad&& col_load)
  -> std::array<eve::wide<T>, Dims<T>::arr_size> {
  using D = Dims<T>;
  std::array<eve::wide<T>, D::arr_size> acc;
  std::ranges::fill(acc, eve::wide<T>(T(0)));
  for (int k = 0; k < K; ++k) {
    std::array<eve::wide<T>, D::x_height> row_tile;
    std::array<eve::wide<T>, D::y_height> col_tile;
    unroll<D::x_height>([&]<int idx>() { row_tile[idx] = row_load.template operator()<idx>(k); });
    unroll<D::y_height>([&]<int idx>() { col_tile[idx] = col_load.template operator()<idx>(k); });
    unroll<D::arr_size>([&]<int idx>() {
      constexpr int row = idx % D::x_height;
      constexpr int col = idx / D::x_height;
      acc[idx] = eve::fma(row_tile[row], col_tile[col], acc[idx]);
    });
  }
  return acc;
}

template <bool take_sign, typename T, typename F>
inline void pair(
  std::span<const T> A,
  std::span<const T> B,
  std::span<T> out,
  int M,
  int K,
  int N,
  int x_chunk,
  int y_chunk,
  int x_off,
  int y_off,
  F&& func
) {
  using D = Dims<T>;
  tile_loop<T>(M, N, x_chunk, y_chunk, x_off, y_off, [&](int i, int j, int i_rem, int j_rem) {
    auto acc = matmul_acc<T>(
      K,
      [&]<int idx>(int k) -> eve::wide<T> {
        return (idx < i_rem) ? eve::wide<T>(A[(x_off + i + idx) * K + k]) : eve::wide<T>(T(0));
      },
      [&]<int idx>(int k) -> eve::wide<T> {
        if (idx * D::vec_size >= j_rem) return eve::wide<T>(T(0));
        int valid = std::min(D::vec_size, j_rem - idx * D::vec_size);
        auto v = mload(&B[k * N + y_off + j + idx * D::vec_size], valid);
        if constexpr (take_sign) v = eve::sign(v);
        return v;
      }
    );
    for_each_elem<T>(i, j, i_rem, j_rem, N, x_off, y_off, [&](int e, int base, int valid) {
      eve::store[eve::keep_first(valid)](func(acc[e]), &out[base]);
    });
  });
}

template <typename T, Variant var>
inline void matrix_accum_internal(
  std::span<const T> X,
  std::span<T> Y,
  int M,
  int N,
  int x_chunk,
  int y_chunk,
  int x_off,
  int y_off,
  float scalar
) {
  tile_loop<T>(M, N, x_chunk, y_chunk, x_off, y_off, [&](int i, int j, int i_rem, int j_rem) {
    for_each_elem<T>(i, j, i_rem, j_rem, N, x_off, y_off, [&](int, int base, int valid) {
      eve::wide<T> data = mload(&X[base], valid);
      eve::wide<T> res;

      if constexpr (var == Variant::NEG) {
        res = -data;
      } else if constexpr (var == Variant::SCALE) {
        res = eve::mul(data, eve::wide<T>(scalar));
      } else {
        res = mload(&Y[base], valid);
        eve::wide<T> mul(scalar);
        if constexpr (var == Variant::FMA) {
          res = eve::fma(mul, res, data);
        } else if constexpr (var == Variant::FNMA) {
          res = eve::fnma(mul, res, data);
        } else {
          if constexpr (var == Variant::SQ_EMA) data = eve::mul(data, data);
          res = eve::fma(mul, res, data);
          res = eve::fnma(mul, data, res);
        }
      }

      eve::store[eve::keep_first(valid)](res, &Y[base]);
    });
  });
}

template <typename T, bool compute_ema>
inline void symmetrized_internal(
  std::span<const T> X_og,
  std::span<const T> X_tp,
  std::span<T> E,
  int M,
  int N,
  int x_chunk,
  int y_chunk,
  int x_off,
  int y_off,
  float ema_rate
) {
  using D = Dims<T>;
  tile_loop<T>(M, N, x_chunk, y_chunk, x_off, y_off, [&](int i, int j, int i_rem, int j_rem) {
    auto acc = matmul_acc<T>(
      N,
      [&]<int idx>(int k) -> eve::wide<T> {
        return (idx < i_rem) ? eve::wide<T>(X_og[(x_off + i + idx) * N + k]) : eve::wide<T>(T(0));
      },
      [&]<int idx>(int k) -> eve::wide<T> {
        if (idx * D::vec_size >= j_rem) return eve::wide<T>(T(0));
        int valid = std::min(D::vec_size, j_rem - idx * D::vec_size);
        return mload(&X_tp[k * M + y_off + j + idx * D::vec_size], valid);
      }
    );
    for_each_elem<T>(i, j, i_rem, j_rem, M, x_off, y_off, [&](int e, int base, int valid) {
      auto* ptr = &E[base];
      eve::wide<T> ema = mload(ptr, valid);
      if constexpr (compute_ema) {
        eve::wide<T> wt(ema_rate);
        ema = eve::fma(wt, ema, acc[e]);
        ema = eve::fnma(wt, acc[e], ema);
      } else {
        ema = acc[e];
      }
      eve::store[eve::keep_first(valid)](ema, ptr);
    });
  });
}

}  // namespace matrix

template <typename T>
void pair_tile(
  std::span<const T> A,
  std::span<const T> B,
  std::span<T> out,
  int M,
  int K,
  int N,
  int x_chunk,
  int y_chunk,
  int x_off,
  int y_off
) {
  matrix::pair<false, T>(A, B, out, M, K, N, x_chunk, y_chunk, x_off, y_off, [](auto& reg) {
    return reg;
  });
}

template <typename T>
void sign_before_pair_tile(
  std::span<const T> A,
  std::span<const T> B,
  std::span<T> out,
  int M,
  int K,
  int N,
  int x_chunk,
  int y_chunk,
  int x_off,
  int y_off
) {
  matrix::pair<true, T>(A, B, out, M, K, N, x_chunk, y_chunk, x_off, y_off, [](auto& reg) {
    return reg;
  });
}

template <typename T>
void sign_after_pair_tile(
  std::span<const T> A,
  std::span<const T> B,
  std::span<T> out,
  int M,
  int K,
  int N,
  int x_chunk,
  int y_chunk,
  int x_off,
  int y_off,
  bool maximize
) {
  matrix::pair<false, T>(A, B, out, M, K, N, x_chunk, y_chunk, x_off, y_off, [&](auto& reg) {
    return ((maximize) ? 1 : -1) * eve::sign(reg);
  });
}

template <typename T>
void norm_pair_tile(
  std::span<const T> A,
  std::span<const T> B,
  std::span<T> out,
  int M,
  int K,
  int N,
  int x_chunk,
  int y_chunk,
  int x_off,
  int y_off
) {
  matrix::pair<false, T>(A, B, out, M, K, N, x_chunk, y_chunk, x_off, y_off, [&](auto& reg) {
    eve::wide<T> m_reg(M);
    eve::wide<T> n_reg(N);
    eve::wide<T> twos(T(2));
    return reg * twos / (m_reg + n_reg);
  });
}

template <typename T>
void symmetrized_ema_tile(
  std::span<const T> X_og,
  std::span<const T> X_tp,
  std::span<T> E,
  int M,
  int N,
  int x_chunk,
  int y_chunk,
  int x_off,
  int y_off,
  float ema_rate
) {
  matrix::symmetrized_internal<T, true>(
    X_og, X_tp, E, M, N, x_chunk, y_chunk, x_off, y_off, ema_rate
  );
}

template <typename T>
void symmetrized_tile(
  std::span<const T> X_og,
  std::span<const T> X_tp,
  std::span<T> E,
  int M,
  int N,
  int x_chunk,
  int y_chunk,
  int x_off,
  int y_off
) {
  matrix::symmetrized_internal<T, false>(X_og, X_tp, E, M, N, x_chunk, y_chunk, x_off, y_off, 0);
}

template <typename T>
void quadratic_tile(
  std::span<const T> A,
  std::span<T> out,
  float b,
  float c,
  int N,
  int x_chunk,
  int y_chunk,
  int x_off,
  int y_off
) {
  using D = matrix::Dims<T>;
  matrix::tile_loop<T>(
    N, N, x_chunk, y_chunk, x_off, y_off, [&](int i, int j, int i_rem, int j_rem) {
      auto acc = matrix::matmul_acc<T>(
        N,
        [&]<int idx>(int k) -> eve::wide<T> {
          return (idx < i_rem) ? eve::wide<T>(A[(x_off + i + idx) * N + k]) : eve::wide<T>(T(0));
        },
        [&]<int idx>(int k) -> eve::wide<T> {
          if (idx * D::vec_size >= j_rem) return eve::wide<T>(T(0));
          int valid = std::min(D::vec_size, j_rem - idx * D::vec_size);
          return matrix::mload(&A[k * N + y_off + j + idx * D::vec_size], valid);
        }
      );
      matrix::for_each_elem<T>(
        i, j, i_rem, j_rem, N, x_off, y_off, [&](int e, int base, int valid) {
          eve::wide<T> lin = matrix::mload(&A[base], valid);
          auto res = eve::add(eve::mul(eve::wide<T>(b), lin), eve::mul(eve::wide<T>(c), acc[e]));
          eve::store[eve::keep_first(valid)](res, &out[base]);
        }
      );
    }
  );
}

template <typename T>
void ns_final_tile(
  std::span<const T> B,
  std::span<const T> X,
  std::span<T> out,
  float a,
  int M,
  int N,
  int x_chunk,
  int y_chunk,
  int x_off,
  int y_off
) {
  using D = matrix::Dims<T>;
  matrix::tile_loop<T>(
    M, N, x_chunk, y_chunk, x_off, y_off, [&](int i, int j, int i_rem, int j_rem) {
      auto acc = matrix::matmul_acc<T>(
        M,
        [&]<int idx>(int k) -> eve::wide<T> {
          return (idx < i_rem) ? eve::wide<T>(B[(x_off + i + idx) * M + k]) : eve::wide<T>(T(0));
        },
        [&]<int idx>(int k) -> eve::wide<T> {
          if (idx * D::vec_size >= j_rem) return eve::wide<T>(T(0));
          int valid = std::min(D::vec_size, j_rem - idx * D::vec_size);
          return matrix::mload(&X[k * N + y_off + j + idx * D::vec_size], valid);
        }
      );
      matrix::for_each_elem<T>(
        i, j, i_rem, j_rem, N, x_off, y_off, [&](int e, int base, int valid) {
          eve::wide<T> x_vec = matrix::mload(&X[base], valid);
          auto res = eve::add(eve::mul(eve::wide<T>(a), x_vec), acc[e]);
          eve::store[eve::keep_first(valid)](res, &out[base]);
        }
      );
    }
  );
}

template <typename T>
void fma_tile(
  std::span<const T> X,
  std::span<T> Y,
  int M,
  int N,
  int x_chunk,
  int y_chunk,
  int x_off,
  int y_off,
  float fma_mul
) {
  matrix::matrix_accum_internal<T, matrix::Variant::FMA>(
    X, Y, M, N, x_chunk, y_chunk, x_off, y_off, fma_mul
  );
}

template <typename T>
void fnma_tile(
  std::span<const T> X,
  std::span<T> Y,
  int M,
  int N,
  int x_chunk,
  int y_chunk,
  int x_off,
  int y_off,
  float fma_mul
) {
  matrix::matrix_accum_internal<T, matrix::Variant::FNMA>(
    X, Y, M, N, x_chunk, y_chunk, x_off, y_off, fma_mul
  );
}

template <typename T>
void ema_tile(
  std::span<const T> X,
  std::span<T> E,
  int M,
  int N,
  int x_chunk,
  int y_chunk,
  int x_off,
  int y_off,
  float ema_rate
) {
  matrix::matrix_accum_internal<T, matrix::Variant::EMA>(
    X, E, M, N, x_chunk, y_chunk, x_off, y_off, ema_rate
  );
}

template <typename T>
void squared_ema_tile(
  std::span<const T> X,
  std::span<T> E,
  int M,
  int N,
  int x_chunk,
  int y_chunk,
  int x_off,
  int y_off,
  float ema_rate
) {
  matrix::matrix_accum_internal<T, matrix::Variant::SQ_EMA>(
    X, E, M, N, x_chunk, y_chunk, x_off, y_off, ema_rate
  );
}

template <typename T>
void normalize(
  std::span<T> X, float norm, int M, int N, int x_chunk, int y_chunk, int x_off, int y_off
) {
  matrix::matrix_accum_internal<T, matrix::Variant::SCALE>(
    X, X, M, N, x_chunk, y_chunk, x_off, y_off, 1.0f / norm
  );
}

template <typename T>
void transpose(std::span<const T> X, std::span<T> out, int M, int N) {
  std::vector<int> indices(M * N);

  std::for_each(indices.begin(), indices.end(), [&](auto& val) {
    auto i = &val - indices.data();

    int out_row = i / M;
    int out_col = i % M;

    val = out_col * N + out_row;
  });

  collect(std::span<const T>(X), std::span<T>(out), std::span<const int>(indices), M * N);
}

template <typename T>
void negate_tile(
  std::span<const T> X, std::span<T> Y, int M, int N, int x_chunk, int y_chunk, int x_off, int y_off
) {
  matrix::matrix_accum_internal<T, matrix::Variant::NEG>(
    X, Y, M, N, x_chunk, y_chunk, x_off, y_off, 0.0f
  );
}

template <typename T>
void adam_tile(
  std::span<const T> X,
  std::span<const T> Y,
  std::span<T> Z,
  int M,
  int N,
  int x_chunk,
  int y_chunk,
  int x_off,
  int y_off,
  float epsilon
) {
  matrix::tile_loop<T>(
    M, N, x_chunk, y_chunk, x_off, y_off, [&](int i, int j, int i_rem, int j_rem) {
      matrix::for_each_elem<T>(i, j, i_rem, j_rem, N, x_off, y_off, [&](int, int base, int valid) {
        eve::wide<T> mom = matrix::mload(&X[base], valid);
        eve::wide<T> vel = matrix::mload(&Y[base], valid);
        auto res = mom / (eve::sqrt(vel) + eve::wide<T>(T(epsilon)));
        eve::store[eve::keep_first(valid)](res, &Z[base]);
      });
    }
  );
}

}  // namespace mirage::detail
