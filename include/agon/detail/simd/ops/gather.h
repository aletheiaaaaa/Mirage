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
  inline Vec<CURRENT_ARCH, T> gather(const T* base, const int32_t* indices) {
    if constexpr (std::is_same_v<T, float>) {
      __m512i vindex = _mm512_loadu_si512(indices);
      return Vec<CURRENT_ARCH, T>(_mm512_i32gather_ps(vindex, base, 4));
    } else if constexpr (std::is_same_v<T, double>) {
      __m256i vindex = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices));
      return Vec<CURRENT_ARCH, T>(_mm512_i32gather_pd(vindex, base, 8));
    }
  }

#elif defined(__AVX2__)
  template<typename T>
    requires std::is_floating_point_v<T>
  inline Vec<CURRENT_ARCH, T> gather(const T* base, const int32_t* indices) {
    if constexpr (std::is_same_v<T, float>) {
      __m256i vindex = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices));
      return Vec<CURRENT_ARCH, T>(_mm256_i32gather_ps(base, vindex, 4));
    } else if constexpr (std::is_same_v<T, double>) {
      __m128i vindex = _mm_loadu_si128(reinterpret_cast<const __m128i*>(indices));
      return Vec<CURRENT_ARCH, T>(_mm256_i32gather_pd(base, vindex, 8));
    }
  }

#else
  template<typename T>
    requires std::is_floating_point_v<T>
  inline Vec<CURRENT_ARCH, T> gather(const T* base, const int32_t* indices) {
    return Vec<CURRENT_ARCH, T>(base[indices[0]]);
  }

#endif
}
