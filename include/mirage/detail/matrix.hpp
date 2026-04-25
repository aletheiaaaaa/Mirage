#pragma once

#include <algorithm>
#include <array>
#include <eve/arch/cpu/wide.hpp>
#include <eve/conditional.hpp>
#include <eve/module/core.hpp>
#include <vector>

#include "utils.hpp"

namespace mirage::detail {
namespace matrix {
template <bool take_sign, typename T, typename F>
inline void triple(
  std::span<const T> A,
  std::span<const T> B,
  std::span<const T> C,
  std::span<T> out,
  int M,
  int K,
  int N,
  int P,
  int x_chunk,
  int y_chunk,
  int x_off,
  int y_off,
  F&& func
) {
  constexpr int x_height = UNROLL_FACTOR;
  constexpr int y_height = std::max(1, UNROLL_FACTOR / 2);
  constexpr int vec_size = eve::wide<T>::size();
  constexpr int arr_size = x_height * y_height;

  x_chunk = std::min(x_chunk, M - x_off);
  y_chunk = std::min(y_chunk, P - y_off);

  for (int i = 0; i < x_chunk; i += x_height) {
    int i_rem = std::min(x_height, x_chunk - i);

    for (int j = 0; j < N; j += y_height * vec_size) {
      int j_rem = std::min(y_height * vec_size, N - j);

      std::array<eve::wide<T>, arr_size> acc0;
      std::ranges::fill(acc0, eve::wide<T>(T(0)));

      for (int k = 0; k < K; ++k) {
        std::array<eve::wide<T>, x_height> a_tile;
        std::array<eve::wide<T>, y_height> b_tile;

        unroll<x_height>([&]<int idx>() {
          a_tile[idx] =
            (idx < i_rem) ? eve::wide<T>(A[(x_off + i + idx) * K + k]) : eve::wide<T>(T(0));
        });

        unroll<y_height>([&]<int idx>() {
          int valid = std::min(vec_size, j_rem - idx * vec_size);

          b_tile[idx] =
            (idx * vec_size < j_rem)
              ? eve::if_else(
                  eve::keep_first(valid), eve::wide<T>(&B[k * N + j + vec_size * idx]), eve::zero
                )
              : eve::wide<T>(T(0));

          if (take_sign) b_tile[idx] = eve::sign(b_tile[idx]);
        });

        unroll<arr_size>([&]<int idx>() {
          constexpr int row = idx % x_height;
          constexpr int col = idx / x_height;

          acc0[idx] = eve::fma(a_tile[row], b_tile[col], acc0[idx]);
        });
      }

      std::array<T, x_height * y_height * vec_size> temp;
      unroll<arr_size>([&]<int idx>() { eve::store(acc0[idx], &temp[idx * vec_size]); });

      for (int k = 0; k < y_chunk; k += y_height * vec_size) {
        int k_rem = std::min(y_height * vec_size, y_chunk - k);

        std::array<eve::wide<T>, arr_size> acc1;
        std::ranges::fill(acc1, eve::wide<T>(T(0)));

        for (int l = 0; l < j_rem; ++l) {
          std::array<eve::wide<T>, x_height> t_tile;
          std::array<eve::wide<T>, y_height> c_tile;

          unroll<x_height>([&]<int idx>() {
            int row = l % vec_size;
            int col = l / vec_size;

            t_tile[idx] = eve::wide<T>(temp[(col * x_height + idx) * vec_size + row]);
          });

          unroll<y_height>([&]<int idx>() {
            int valid = std::min(vec_size, k_rem - idx * vec_size);

            c_tile[idx] = (idx * vec_size < k_rem)
                            ? eve::if_else(
                                eve::keep_first(valid),
                                eve::wide<T>(&C[(j + l) * P + y_off + k + idx * vec_size]),
                                eve::zero
                              )
                            : eve::wide<T>(T(0));
          });

          unroll<arr_size>([&]<int idx>() {
            constexpr int row = idx % x_height;
            constexpr int col = idx / x_height;

            acc1[idx] = eve::fma(t_tile[row], c_tile[col], acc1[idx]);
          });
        }

        unroll<arr_size>([&]<int idx>() {
          constexpr int row = idx % x_height;
          constexpr int col = idx / x_height;

          if (row < i_rem && col * vec_size < k_rem) {
            auto mask = eve::keep_first(std::min(vec_size, k_rem - col * vec_size));

            eve::wide<T> prev = eve::if_else(
              mask,
              eve::wide<T>(&out[(x_off + i + row) * P + y_off + k + col * vec_size]),
              eve::zero
            );
            eve::store[mask](

              prev + acc1[idx], &out[(x_off + i + row) * P + y_off + k + col * vec_size]
            );
          }
        });
      }
    }

    for (int j = 0; j < y_chunk; j += y_height * vec_size) {
      int j_rem = std::min(y_height * vec_size, y_chunk - j);
      unroll<arr_size>([&]<int idx>() {
        constexpr int row = idx % x_height;
        constexpr int col = idx / x_height;

        if (row < i_rem && col * vec_size < j_rem) {
          auto mask = eve::keep_first(std::min(vec_size, j_rem - col * vec_size));

          eve::wide<T> val = eve::if_else(
            mask, eve::wide<T>(&out[(x_off + i + row) * P + y_off + j + col * vec_size]), eve::zero
          );
          eve::store[mask](func(val), &out[(x_off + i + row) * P + y_off + j + col * vec_size]);
        }
      });
    }
  }
}

template <typename T, bool compute_ema, bool squared_fma>
inline void matrix_accum_internal(
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
  constexpr int x_height = UNROLL_FACTOR;
  constexpr int y_height = std::max(1, UNROLL_FACTOR / 2);
  constexpr int vec_size = eve::wide<T>::size();
  constexpr int arr_size = x_height * y_height;

  x_chunk = std::min(x_chunk, M - x_off);
  y_chunk = std::min(y_chunk, N - y_off);

  for (int i = 0; i < x_chunk; i += x_height) {
    int i_rem = std::min(x_height, x_chunk - i);

    for (int j = 0; j < y_chunk; j += y_height * vec_size) {
      int j_rem = std::min(y_height * vec_size, y_chunk - j);

      unroll<arr_size>([&]<int idx>() {
        constexpr int row = idx % x_height;
        constexpr int col = idx / x_height;

        if (row < i_rem && col * vec_size < j_rem) {
          auto valid = std::min(vec_size, j_rem - col * vec_size);

          eve::wide<T> res = eve::if_else(
            eve::keep_first(valid),
            eve::wide<T>(&Y[(x_off + i + row) * N + y_off + j + col * vec_size]),
            eve::zero
          );
          eve::wide<T> data = eve::if_else(
            eve::keep_first(valid),
            eve::wide<T>(&X[(x_off + i + row) * N + y_off + j + col * vec_size]),
            eve::zero
          );
          eve::wide<T> mul(fma_mul);

          if (squared_fma) data = data * data;
          res = eve::fma(mul, res, data);
          if (compute_ema) res = eve::fnma(mul, data, res);

          eve::store[eve::keep_first(valid)](
            res, &Y[(x_off + i + row) * N + y_off + j + col * vec_size]
          );
        }
      });
    }
  }
}
}  // namespace matrix

template <typename T>
void triple_tile(
  std::span<const T> A,
  std::span<const T> B,
  std::span<const T> C,
  std::span<T> out,
  int M,
  int K,
  int N,
  int P,
  int x_chunk,
  int y_chunk,
  int x_off,
  int y_off,
  bool maximize
) {
  matrix::triple<false, T>(
    A, B, C, out, M, K, N, P, x_chunk, y_chunk, x_off, y_off, [&](auto& reg) {
      return ((maximize) ? 1 : -1) * eve::sign(reg);
    }
  );
}

template <typename T>
void norm_triple_sign_tile(
  std::span<const T> A,
  std::span<const T> B,
  std::span<const T> C,
  std::span<T> out,
  int M,
  int K,
  int N,
  int P,
  int x_chunk,
  int y_chunk,
  int x_off,
  int y_off
) {
  matrix::triple<true, T>(A, B, C, out, M, K, N, P, x_chunk, y_chunk, x_off, y_off, [&](auto& reg) {
    eve::wide<T> m_reg(M);
    eve::wide<T> n_reg(N);
    eve::wide<T> twos(T(2));

    return reg * twos / (m_reg + n_reg);
  });
}

namespace matrix {
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
  constexpr int x_height = UNROLL_FACTOR;
  constexpr int y_height = std::max(1, UNROLL_FACTOR / 2);
  constexpr int vec_size = eve::wide<T>::size();
  constexpr int arr_size = x_height * y_height;

  x_chunk = std::min(x_chunk, M - x_off);
  y_chunk = std::min(y_chunk, N - y_off);

  for (int i = 0; i < x_chunk; i += x_height) {
    int i_rem = std::min(x_height, x_chunk - i);

    for (int j = 0; j < y_chunk; j += y_height * vec_size) {
      int j_rem = std::min(y_height * vec_size, y_chunk - j);

      std::array<eve::wide<T>, arr_size> acc;
      std::ranges::fill(acc, eve::wide<T>(T(0)));

      for (int k = 0; k < N; ++k) {
        std::array<eve::wide<T>, x_height> og_tile;
        std::array<eve::wide<T>, y_height> tp_tile;

        unroll<x_height>([&]<int idx>() {
          og_tile[idx] =
            (idx < i_rem) ? eve::wide<T>(X_og[(x_off + i + idx) * N + k]) : eve::wide<T>(T(0));
        });

        unroll<y_height>([&]<int idx>() {
          int valid = std::min(vec_size, j_rem - idx * vec_size);

          tp_tile[idx] = (idx * vec_size < j_rem)
                           ? eve::if_else(
                               eve::keep_first(valid),
                               eve::wide<T>(&X_tp[k * M + y_off + j + idx * vec_size]),
                               eve::zero
                             )
                           : eve::wide<T>(T(0));
        });

        unroll<arr_size>([&]<int idx>() {
          constexpr int row = idx % x_height;
          constexpr int col = idx / x_height;

          acc[idx] = eve::fma(og_tile[row], tp_tile[col], acc[idx]);
        });
      }

      unroll<arr_size>([&]<int idx>() {
        constexpr int row = idx % x_height;
        constexpr int col = idx / x_height;

        if (row < i_rem && col * vec_size < j_rem) {
          auto valid = std::min(vec_size, j_rem - col * vec_size);

          eve::wide<T> ema = eve::if_else(
            eve::keep_first(valid),
            eve::wide<T>(&E[(x_off + i + row) * M + y_off + j + col * vec_size]),
            eve::zero
          );

          if (compute_ema) {
            eve::wide<T> wt(ema_rate);

            ema = eve::fma(wt, ema, acc[idx]);
            ema = eve::fnma(wt, acc[idx], ema);
          } else {
            ema = acc[idx];
          }

          eve::store[eve::keep_first(valid)](
            ema, &E[(x_off + i + row) * M + y_off + j + col * vec_size]
          );
        }
      });
    }
  }
}
}  // namespace matrix

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
  constexpr int x_height = UNROLL_FACTOR;
  constexpr int y_height = std::max(1, UNROLL_FACTOR / 2);
  constexpr int vec_size = eve::wide<T>::size();
  constexpr int arr_size = x_height * y_height;

  x_chunk = std::min(x_chunk, N - x_off);
  y_chunk = std::min(y_chunk, N - y_off);

  for (int i = 0; i < x_chunk; i += x_height) {
    int i_rem = std::min(x_height, x_chunk - i);

    for (int j = 0; j < y_chunk; j += y_height * vec_size) {
      int j_rem = std::min(y_height * vec_size, y_chunk - j);

      std::array<eve::wide<T>, arr_size> acc;
      std::ranges::fill(acc, eve::wide<T>(T(0)));

      for (int k = 0; k < N; ++k) {
        std::array<eve::wide<T>, x_height> tile0;
        std::array<eve::wide<T>, y_height> tile1;

        unroll<x_height>([&]<int idx>() {
          tile0[idx] =
            (idx < i_rem) ? eve::wide<T>(A[(x_off + i + idx) * N + k]) : eve::wide<T>(T(0));
        });

        unroll<y_height>([&]<int idx>() {
          int valid = std::min(vec_size, j_rem - idx * vec_size);

          tile1[idx] = (idx * vec_size < j_rem)
                         ? eve::if_else(
                             eve::keep_first(valid),
                             eve::wide<T>(&A[k * N + y_off + j + idx * vec_size]),
                             eve::zero
                           )
                         : eve::wide<T>(T(0));
        });

        unroll<arr_size>([&]<int idx>() {
          constexpr int row = idx % x_height;
          constexpr int col = idx / x_height;

          acc[idx] = eve::fma(tile0[row], tile1[col], acc[idx]);
        });
      }

      unroll<arr_size>([&]<int idx>() {
        constexpr int row = idx % x_height;
        constexpr int col = idx / x_height;

        if (row < i_rem && col * vec_size < j_rem) {
          auto valid = std::min(vec_size, j_rem - col * vec_size);

          eve::wide<T> lin = eve::if_else(
            eve::keep_first(valid),
            eve::wide<T>(&A[(x_off + i + row) * N + y_off + j + col * vec_size]),
            eve::zero
          );
          eve::wide<T> b_vec(b);
          eve::wide<T> c_vec(c);

          auto res = eve::add(eve::mul(b_vec, lin), eve::mul(c_vec, acc[idx]));

          eve::store[eve::keep_first(valid)](
            res, &out[(x_off + i + row) * N + y_off + j + col * vec_size]
          );
        }
      });
    }
  }
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
  constexpr int x_height = UNROLL_FACTOR;
  constexpr int y_height = std::max(1, UNROLL_FACTOR / 2);
  constexpr int vec_size = eve::wide<T>::size();
  constexpr int arr_size = x_height * y_height;

  x_chunk = std::min(x_chunk, M - x_off);
  y_chunk = std::min(y_chunk, N - y_off);

  for (int i = 0; i < x_chunk; i += x_height) {
    int i_rem = std::min(x_height, x_chunk - i);

    for (int j = 0; j < y_chunk; j += y_height * vec_size) {
      int j_rem = std::min(y_height * vec_size, y_chunk - j);

      std::array<eve::wide<T>, arr_size> acc;
      std::ranges::fill(acc, eve::wide<T>(T(0)));

      for (int k = 0; k < M; ++k) {
        std::array<eve::wide<T>, x_height> tile0;
        std::array<eve::wide<T>, y_height> tile1;

        unroll<x_height>([&]<int idx>() {
          tile0[idx] =
            (idx < i_rem) ? eve::wide<T>(B[(x_off + i + idx) * M + k]) : eve::wide<T>(T(0));
        });

        unroll<y_height>([&]<int idx>() {
          int valid = std::min(vec_size, j_rem - idx * vec_size);

          tile1[idx] = (idx * vec_size < j_rem)
                         ? eve::if_else(
                             eve::keep_first(valid),
                             eve::wide<T>(&X[k * N + y_off + j + idx * vec_size]),
                             eve::zero
                           )
                         : eve::wide<T>(T(0));
        });

        unroll<arr_size>([&]<int idx>() {
          constexpr int row = idx % x_height;
          constexpr int col = idx / x_height;

          acc[idx] = eve::fma(tile0[row], tile1[col], acc[idx]);
        });
      }

      unroll<arr_size>([&]<int idx>() {
        constexpr int row = idx % x_height;
        constexpr int col = idx / x_height;

        if (row < i_rem && col * vec_size < j_rem) {
          auto valid = std::min(vec_size, j_rem - col * vec_size);

          eve::wide<T> x_vec = eve::if_else(
            eve::keep_first(valid),
            eve::wide<T>(&X[(x_off + i + row) * N + y_off + j + col * vec_size]),
            eve::zero
          );
          eve::wide<T> a_vec(a);

          auto res = eve::add(eve::mul(a_vec, x_vec), acc[idx]);

          eve::store[eve::keep_first(valid)](
            res, &out[(x_off + i + row) * N + y_off + j + col * vec_size]
          );
        }
      });
    }
  }
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
  matrix::matrix_accum_internal<T, false, false>(
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
  matrix::matrix_accum_internal<T, true, false>(
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
  matrix::matrix_accum_internal<T, true, true>(
    X, E, M, N, x_chunk, y_chunk, x_off, y_off, ema_rate
  );
}

template <typename T>
std::vector<T> transpose(std::span<const T> X, int M, int N) {
  std::vector<T> out(M * N);
  std::vector<int> indices(M * N);

  std::for_each(indices.begin(), indices.end(), [&](auto& val) {
    auto i = &val - indices.data();

    int row = i % N;
    int col = i / N;

    val = M * col + row;
  });

  collect(std::span<const T>(X), std::span<T>(out), std::span<const int>(indices), M * N);

  return out;
}

template <typename T>
void negate_tile(
  std::span<const T> X, std::span<T> Y, int M, int N, int x_chunk, int y_chunk, int x_off, int y_off
) {
  constexpr int x_height = UNROLL_FACTOR;
  constexpr int y_height = std::max(1, UNROLL_FACTOR / 2);
  constexpr int vec_size = eve::wide<T>::size();
  constexpr int arr_size = x_height * y_height;

  x_chunk = std::min(x_chunk, M - x_off);
  y_chunk = std::min(y_chunk, N - y_off);

  for (int i = 0; i < x_chunk; i += x_height) {
    int i_rem = std::min(x_height, x_chunk - i);

    for (int j = 0; j < y_chunk; j += y_height * vec_size) {
      int j_rem = std::min(y_height * vec_size, y_chunk - j);

      unroll<arr_size>([&]<int idx>() {
        constexpr int row = idx % x_height;
        constexpr int col = idx / x_height;

        if (row < i_rem && col * vec_size < j_rem) {
          auto valid = std::min(vec_size, j_rem - col * vec_size);

          eve::wide<T> data = eve::if_else(
            eve::keep_first(valid),
            eve::wide<T>(&X[(x_off + i + row) * N + y_off + j + col * vec_size]),
            eve::zero
          );

          data = -data;
          eve::store[eve::keep_first(valid)](
            data, &Y[(x_off + i + row) * N + y_off + j + col * vec_size]
          );
        }
      });
    }
  }
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
  constexpr int x_height = UNROLL_FACTOR;
  constexpr int y_height = std::max(1, UNROLL_FACTOR / 2);
  constexpr int vec_size = eve::wide<T>::size();
  constexpr int arr_size = x_height * y_height;

  x_chunk = std::min(x_chunk, M - x_off);
  y_chunk = std::min(y_chunk, N - y_off);

  for (int i = 0; i < x_chunk; i += x_height) {
    int i_rem = std::min(x_height, x_chunk - i);

    for (int j = 0; j < y_chunk; j += y_height * vec_size) {
      int j_rem = std::min(y_height * vec_size, y_chunk - j);

      unroll<arr_size>([&]<int idx>() {
        constexpr int row = idx % x_height;
        constexpr int col = idx / x_height;

        if (row < i_rem && col * vec_size < j_rem) {
          auto valid = std::min(vec_size, j_rem - col * vec_size);

          eve::wide<T> mom = eve::if_else(
            eve::keep_first(valid),
            eve::wide<T>(&X[(x_off + i + row) * N + y_off + j + col * vec_size]),
            eve::zero
          );
          eve::wide<T> vel = eve::if_else(
            eve::keep_first(valid),
            eve::wide<T>(&Y[(x_off + i + row) * N + y_off + j + col * vec_size]),
            eve::zero
          );

          auto eps = eve::wide<T>(T(epsilon));
          auto res = mom / (eve::sqrt(vel) + eps);

          eve::store[eve::keep_first(valid)](
            res, &Z[(x_off + i + row) * N + y_off + j + col * vec_size]
          );
        }
      });
    }
  }
}

}  // namespace mirage::detail
