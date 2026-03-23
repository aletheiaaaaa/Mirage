#pragma once

#include <cstddef>
#include <cstdint>
#include <eve/module/core.hpp>
#include <eve/wide.hpp>
#include <string>
#include <utility>
#include <vector>

#include "arch.hpp"

namespace mirage::detail {
template <typename T>
struct TypeName {
  static std::string name() { return "unknown"; }
};
template <>
struct TypeName<float> {
  static std::string name() { return "float"; }
};
template <>
struct TypeName<double> {
  static std::string name() { return "double"; }
};
template <>
struct TypeName<int16_t> {
  static std::string name() { return "int16_t"; }
};
template <>
struct TypeName<int8_t> {
  static std::string name() { return "int8_t"; }
};

template <typename F, typename T>
concept IsUpcast = eve::wide<F>::size() < eve::wide<T>::size();

constexpr int UNROLL_FACTOR =
  (CURRENT_ARCH == Arch::AVX512 || CURRENT_ARCH == Arch::NEON) ? 4
  : (CURRENT_ARCH == Arch::AVX2)                               ? 2
                                                               : 1;

constexpr int OUTER_REGS = UNROLL_FACTOR * 2;
constexpr int INNER_REGS = std::max(1, UNROLL_FACTOR / 2);

template <size_t N, typename F>
constexpr void unroll(F&& func) {
  [&]<size_t... Is>(std::index_sequence<Is...>) {
    (func.template operator()<Is>(), ...);
  }(std::make_index_sequence<N>{});
}

template <typename T>
inline void collect(
  std::vector<T>& source,
  std::vector<T>& target,
  std::vector<int>& indices,
  size_t max
) {
  constexpr size_t vec_size = eve::wide<T>::size();
  constexpr size_t unroll_factor = detail::UNROLL_FACTOR;

  size_t i = 0;
  for (; i + vec_size * unroll_factor < max; i += vec_size * unroll_factor) {
    detail::unroll<unroll_factor>([&]<size_t index>() {
      constexpr size_t offset = index * vec_size;

      eve::wide<int32_t, eve::fixed<vec_size>> gather_idxs(
        &indices[i + offset]
      );
      eve::wide<T> idxs = eve::gather(source.data(), gather_idxs);
      eve::store(idxs, &target[i + offset]);
    });
  }

  for (; i < max; ++i) {
    target[i] = source[indices[i]];
  }
}
}  // namespace mirage::detail