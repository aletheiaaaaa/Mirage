#pragma once

#include "../arch.h"
#include "../types.h"
#if defined(__AVX512F__)
    #include <immintrin.h>
#elif defined(__AVX2__)
    #include <immintrin.h>
#elif defined(__SSE4_1__)
    #include <smmintrin.h>
#endif

namespace agon::simd {
#if defined(__AVX512F__)
    template<typename Vec>
        requires (std::is_same_v<Vec, VecF32<Arch::AVX512>> ||
                  std::is_same_v<Vec, VecF64<Arch::AVX512>>)
    inline Vec neg(const Vec& a) {
        if constexpr (std::is_same_v<Vec, VecF32<Arch::AVX512>>) {
            return Vec(_mm512_xor_ps(a.data, _mm512_set1_ps(-0.0f)));
        } else if constexpr (std::is_same_v<Vec, VecF64<Arch::AVX512>>) {
            return Vec(_mm512_xor_pd(a.data, _mm512_set1_pd(-0.0)));
        }
    }
#elif defined(__AVX2__)

    template<typename Vec>
        requires (std::is_same_v<Vec, VecF32<Arch::AVX2>> ||
                  std::is_same_v<Vec, VecF64<Arch::AVX2>>)
    inline Vec neg(const Vec& a) {
        if constexpr (std::is_same_v<Vec, VecF32<Arch::AVX2>>) {
            return Vec(_mm256_xor_ps(a.data, _mm256_set1_ps(-0.0f)));
        } else if constexpr (std::is_same_v<Vec, VecF64<Arch::AVX2>>) {
            return Vec(_mm256_xor_pd(a.data, _mm256_set1_pd(-0.0)));
        }
    }
#elif defined(__SSE4_1__)

    template<typename Vec>
        requires (std::is_same_v<Vec, VecF32<Arch::SSE4_1>> ||
                  std::is_same_v<Vec, VecF64<Arch::SSE4_1>>)
    inline Vec neg(const Vec& a) {
        if constexpr (std::is_same_v<Vec, VecF32<Arch::SSE4_1>>) {
            return Vec(_mm_xor_ps(a.data, _mm_set1_ps(-0.0f)));
        } else if constexpr (std::is_same_v<Vec, VecF64<Arch::SSE4_1>>) {
            return Vec(_mm_xor_pd(a.data, _mm_set1_pd(-0.0)));
        }
    }#else
    template<typename Vec>
        requires (std::is_same_v<Vec, VecF32<Arch::GENERIC>> ||
                  std::is_same_v<Vec, VecF64<Arch::GENERIC>>)
    inline Vec neg(const Vec& a) {
        return Vec(-a.data);
    }

#endif

}
