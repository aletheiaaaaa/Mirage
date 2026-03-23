#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <eve/arch/cpu/wide.hpp>
#include <eve/conditional.hpp>
#include <eve/module/core.hpp>
#include <eve/module/core/regular/if_else.hpp>
#include <eve/module/core/regular/store.hpp>
#include <vector>

#include "utils.hpp"

namespace mirage::detail {
namespace matrix {
template <typename T, typename F>
  requires std::invocable<F, eve::wide<T>>
inline void triple(
  const std::vector<T>& A,
  const std::vector<T>& B,
  const std::vector<T>& C,
  std::vector<T>& out,
  size_t M,
  size_t K,
  size_t N,
  size_t P,
  size_t x_chunk,
  size_t y_chunk,
  size_t x_off,
  size_t y_off,
  F&& func
) {
  constexpr size_t x_height = UNROLL_FACTOR;
  constexpr size_t y_height = std::max(1, UNROLL_FACTOR / 2);
  constexpr size_t vec_size = eve::wide<T>::size();
  constexpr size_t arr_size = x_height * y_height;

  for (size_t i = 0; i < x_chunk; i += x_height) {
    size_t i_rem = std::min(x_height, x_chunk - i);

    for (size_t j = 0; j < N; j += y_height * vec_size) {
      size_t j_rem = std::min(y_height * vec_size, N - j);

      std::array<eve::wide<T>, arr_size> acc0;
      std::ranges::fill(acc0, eve::wide<T>(T(0)));

      for (size_t k = 0; k < K; ++k) {
        std::array<eve::wide<T>, x_height> a_tile;
        std::array<eve::wide<T>, y_height> b_tile;

        unroll<x_height>([&]<size_t idx>() {
          a_tile[idx] = (idx < i_rem)
                          ? eve::wide<T>(A[(x_chunk + i + idx) * K + k])
                          : eve::wide<T>(T(0));
        });

        unroll<y_height>([&]<size_t idx>() {
          size_t valid = std::min(vec_size, j_rem - idx * vec_size);

          b_tile[idx] = (idx * vec_size < j_rem)
                          ? eve::if_else(
                              eve::keep_first(valid),
                              eve::wide<T>(&B[k * N + j + vec_size * idx]),
                              eve::zero
                            )
                          : eve::wide<T>(T(0));
        });

        unroll<arr_size>([&]<size_t idx>() {
          constexpr size_t row = idx % x_height;
          constexpr size_t col = idx / x_height;

          acc0[idx] = eve::fma(a_tile[row], b_tile[col], acc0[idx]);
        });
      }

      std::array<T, x_height * y_height * vec_size> temp;
      unroll<arr_size>([&]<size_t idx>() {
        eve::store(acc0[idx], &temp[idx * vec_size]);
      });

      for (size_t k = 0; k < y_chunk; k += y_height * vec_size) {
        size_t k_rem = std::min(y_height * vec_size, y_chunk - k);

        std::array<eve::wide<T>, arr_size> acc1;
        std::ranges::fill(acc1, eve::wide<T>(T(0)));

        for (size_t l = 0; l < j_rem; ++l) {
          std::array<eve::wide<T>, x_height> t_tile;
          std::array<eve::wide<T>, y_height> c_tile;

          unroll<x_height>([&]<size_t idx>() {
            size_t row = l % vec_size;
            size_t col = l / vec_size;

            t_tile[idx] =
              eve::wide<T>(temp[(col * x_height + idx) * vec_size + row]);
          });

          unroll<y_height>([&]<size_t idx>() {
            size_t valid = std::min(vec_size, k_rem - idx * vec_size);

            c_tile[idx] = (idx * vec_size < k_rem)
                            ? eve::if_else(
                                eve::keep_first(valid),
                                eve::wide<T>(
                                  &C[(j + l) * P + y_chunk + k + idx * vec_size]
                                ),
                                eve::zero
                              )
                            : eve::wide<T>(T(0));
          });

          unroll<arr_size>([&]<size_t idx>() {
            constexpr size_t row = idx % x_height;
            constexpr size_t col = idx / x_height;

            acc1[idx] = eve::fma(t_tile[row], c_tile[col], acc1[idx]);
          });
        }

        unroll<arr_size>([&]<size_t idx>() {
          constexpr size_t row = idx % x_height;
          constexpr size_t col = idx / x_height;

          if (row < i_rem && col * vec_size < k_rem) {
            auto mask =
              eve::keep_first(std::min(vec_size, k_rem - col * vec_size));

            eve::wide<T> prev = eve::if_else(
              mask,
              eve::wide<T>(
                &out[(x_chunk + i + row) * P + y_chunk + k + col * vec_size]
              ),
              eve::zero
            );
            eve::store[mask](
              prev + acc1[idx],
              &out[(x_chunk + i + row) * P + y_chunk + k + col * vec_size]
            );
          }
        });
      }
    }

    for (size_t j = 0; j < y_chunk; j += y_height * vec_size) {
      size_t j_rem = std::min(y_height * vec_size, y_chunk - j);
      unroll<arr_size>([&]<size_t idx>() {
        constexpr size_t row = idx % x_height;
        constexpr size_t col = idx / x_height;

        if (row < i_rem && col * vec_size < j_rem) {
          auto mask =
            eve::keep_first(std::min(vec_size, j_rem - col * vec_size));

          eve::wide<T> val = eve::if_else(
            mask,
            eve::wide<T>(
              &out[(x_chunk + i + row) * P + y_chunk + j + col * vec_size]
            ),
            eve::zero
          );
          eve::store[mask](
            func(val),
            &out[(x_chunk + i + row) * P + y_chunk + j + col * vec_size]
          );
        }
      });
    }
  }
}

template <typename T, bool compute_ema>
inline void matrix_fma(
  const std::vector<T>& X,
  std::vector<T>& Y,
  size_t M,
  size_t N,
  size_t x_chunk,
  size_t y_chunk,
  size_t x_off,
  size_t y_off,
  float fma_mul
) {
  constexpr size_t x_height = UNROLL_FACTOR;
  constexpr size_t y_height = std::max(1, UNROLL_FACTOR / 2);
  constexpr size_t vec_size = eve::wide<T>::size();
  constexpr size_t arr_size = x_height * y_height;

  for (size_t i = 0; i < x_chunk; i += x_height) {
    size_t i_rem = std::min(x_height, x_chunk - i);

    for (size_t j = 0; j < y_chunk; j += y_height * vec_size) {
      size_t j_rem = std::min(y_height * vec_size, y_chunk - j);

      unroll<arr_size>([&]<size_t idx>() {
        constexpr size_t row = idx % x_height;
        constexpr size_t col = idx / x_height;

        if (row < i_rem && col * vec_size < j_rem) {
          auto valid = std::min(vec_size, j_rem - col * vec_size);

          eve::wide<T> res = eve::if_else(
            eve::keep_first(valid),
            eve::wide<T>(
              &Y[(x_off + i + row) * M + y_off + j + col * vec_size]
            ),
            eve::zero
          );
          eve::wide<T> data = eve::if_else(
            eve::keep_first(valid),
            eve::wide<T>(
              &X[(x_off + i + row) * M + y_off + j + col * vec_size]
            ),
            eve::zero
          );
          eve::wide<T> mul(fma_mul);

          res = eve::fma(mul, res, data);
          if (compute_ema) res = eve::fnma(mul, data, res);

          eve::store[eve::keep_first(valid)](
            res, &Y[(x_off + i + row) * M + y_off + j + col * vec_size]
          );
        }
      });
    }
  }
}
}  // namespace matrix

template <typename T>
void triple_matmul_tile(
  const std::vector<T>& A,
  const std::vector<T>& B,
  const std::vector<T>& C,
  std::vector<T>& out,
  size_t M,
  size_t K,
  size_t N,
  size_t P,
  size_t x_chunk,
  size_t y_chunk,
  size_t x_off,
  size_t y_off
) {
  matrix::triple(
    A,
    B,
    C,
    out,
    M,
    K,
    N,
    P,
    x_chunk,
    y_chunk,
    x_off,
    y_off,
    [&](eve::wide<T>& reg) {}
  );
}

template <typename T>
void triple_matmul_sign(
  const std::vector<T>& A,
  const std::vector<T>& B,
  const std::vector<T>& C,
  std::vector<T>& out,
  size_t M,
  size_t K,
  size_t N,
  size_t P,
  size_t x_chunk,
  size_t y_chunk,
  size_t x_off,
  size_t y_off
) {
  matrix::triple(
    A,
    B,
    C,
    out,
    M,
    K,
    N,
    P,
    x_chunk,
    y_chunk,
    x_off,
    y_off,
    [&](eve::wide<T>& reg) { return eve::sign(reg); }
  );
}

template <typename T>
void symmetrized_ema_tile(
  const std::vector<T>& X_og,
  const std::vector<T>& X_tp,
  std::vector<T>& E,
  size_t M,
  size_t N,
  size_t x_chunk,
  size_t y_chunk,
  size_t x_off,
  size_t y_off,
  float ema_rate
) {
  constexpr size_t x_height = UNROLL_FACTOR;
  constexpr size_t y_height = std::max(1, UNROLL_FACTOR / 2);
  constexpr size_t vec_size = eve::wide<T>::size();
  constexpr size_t arr_size = x_height * y_height;

  for (size_t i = 0; i < x_chunk; i += x_height) {
    size_t i_rem = std::min(x_height, x_chunk - i);

    for (size_t j = 0; j < y_chunk; j += y_height * vec_size) {
      size_t j_rem = std::min(y_height * vec_size, y_chunk - j);

      std::array<eve::wide<T>, arr_size> acc;
      std::ranges::fill(acc, eve::wide<T>(T(0)));

      for (size_t k = 0; k < N; ++k) {
        std::array<eve::wide<T>, x_height> og_tile;
        std::array<eve::wide<T>, y_height> tp_tile;

        unroll<x_height>([&]<size_t idx>() {
          og_tile[idx] = (idx < i_rem)
                           ? eve::wide<T>(X_og[(x_off + i + idx) * N + k])
                           : eve::wide<T>(T(0));
        });

        unroll<y_height>([&]<size_t idx>() {
          size_t valid = std::min(vec_size, j_rem - idx * vec_size);

          tp_tile[idx] =
            (idx * vec_size < j_rem)
              ? eve::if_else(
                  eve::keep_first(valid),
                  eve::wide<T>(&X_tp[k * M + y_off + j + idx * vec_size]),
                  eve::zero
                )
              : eve::wide<T>(T(0));
        });

        unroll<arr_size>([&]<size_t idx>() {
          constexpr size_t row = idx % x_height;
          constexpr size_t col = idx / x_height;

          acc[idx] = eve::fma(og_tile[row], tp_tile[col], acc[idx]);
        });
      }

      unroll<arr_size>([&]<size_t idx>() {
        constexpr size_t row = idx % x_height;
        constexpr size_t col = idx / x_height;

        if (row < i_rem && col * vec_size < j_rem) {
          auto valid = std::min(vec_size, j_rem - col * vec_size);

          eve::wide<T> ema = eve::if_else(
            eve::keep_first(valid),
            eve::wide<T>(
              &E[(x_off + i + row) * M + y_off + j + col * vec_size]
            ),
            eve::zero
          );
          eve::wide<T> wt(ema_rate);

          ema = eve::fma(wt, ema, acc[idx]);
          ema = eve::fnma(wt, acc[idx], ema);

          eve::store[eve::keep_first(valid)](
            ema, &E[(x_off + i + row) * M + y_off + j + col * vec_size]
          );
        }
      });
    }
  }
}

template <typename T>
void fma_tile(
  const std::vector<T>& X,
  std::vector<T>& Y,
  size_t M,
  size_t N,
  size_t x_chunk,
  size_t y_chunk,
  size_t x_off,
  size_t y_off,
  float fma_mul
) {
  matrix::matrix_fma<false>(
    X, Y, M, N, x_chunk, y_chunk, x_off, y_off, fma_mul
  );
}

template <typename T>
void ema_tile(
  const std::vector<T>& X,
  std::vector<T>& E,
  size_t M,
  size_t N,
  size_t x_chunk,
  size_t y_chunk,
  size_t x_off,
  size_t y_off,
  float ema_rate
) {
  matrix::matrix_fma<true>(
    X, E, M, N, x_chunk, y_chunk, x_off, y_off, ema_rate
  );
}

template <typename T>
std::vector<T> transpose(const std::vector<T>& X, size_t M, size_t N) {
  std::vector<T> out(M * N);
  std::vector<T> indices(M * N);

  std::for_each(indices.begin(), indices.end(), [&](auto& val) {
    auto i = &val - indices.data();

    size_t row = i % N;
    size_t col = i / N;

    val = M * col + row;
  });

  collect(X, out, indices);

  return out;
}
}  // namespace mirage::detail