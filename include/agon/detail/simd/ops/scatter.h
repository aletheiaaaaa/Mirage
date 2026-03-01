#pragma once

#include "../arch.h"
#include "../types.h"
#include <cstdint>
#if defined(__AVX512F__)
  #include <immintrin.h>
#elif defined(__AVX2__)
  #include <immintrin.h>
#elif defined(__SSE4_1__)
  #include <smmintrin.h>
#endif

namespace agon::simd {
#if defined(__AVX512F__)
  template<typename T>
    requires std::is_floating_point_v<T>
  inline void scatter(T* base, const int32_t* indices, const Vec<CURRENT_ARCH, T>& v) {
    if constexpr (std::is_same_v<T, float>) {
      __m512i vindex = _mm512_loadu_si512(indices);
      _mm512_i32scatter_ps(base, vindex, v.data, 4);
    } else if constexpr (std::is_same_v<T, double>) {
      __m256i vindex = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices));
      _mm512_i32scatter_pd(base, vindex, v.data, 8);
    }
  }

#else
  template<typename T>
    requires std::is_floating_point_v<T>
  inline void scatter(T* base, const int32_t* indices, const Vec<CURRENT_ARCH, T>& v) {
    base[indices[0]] = v.data;
  }

#endif
}
