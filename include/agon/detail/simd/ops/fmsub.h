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
    inline Vec fmsub(const Vec& a, const Vec& b, const Vec& c) {
        if constexpr (std::is_same_v<Vec, VecF32<Arch::AVX512>>) {
            return Vec(_mm512_fmsub_ps(a.data, b.data, c.data));
        } else if constexpr (std::is_same_v<Vec, VecF64<Arch::AVX512>>) {
            return Vec(_mm512_fmsub_pd(a.data, b.data, c.data));
        }
    }

#elif defined(__AVX2__)
    template<typename Vec>
        requires (std::is_same_v<Vec, VecF32<Arch::AVX2>> ||
                  std::is_same_v<Vec, VecF64<Arch::AVX2>>)
    inline Vec fmsub(const Vec& a, const Vec& b, const Vec& c) {
        if constexpr (std::is_same_v<Vec, VecF32<Arch::AVX2>>) {
            return Vec(_mm256_fmsub_ps(a.data, b.data, c.data));
        } else if constexpr (std::is_same_v<Vec, VecF64<Arch::AVX2>>) {
            return Vec(_mm256_fmsub_pd(a.data, b.data, c.data));
        }
    }

#elif defined(__SSE4_1__)
    template<typename Vec>
        requires (std::is_same_v<Vec, VecF32<Arch::SSE4_1>> ||
                  std::is_same_v<Vec, VecF64<Arch::SSE4_1>>)
    inline Vec fmsub(const Vec& a, const Vec& b, const Vec& c) {
        if constexpr (std::is_same_v<Vec, VecF32<Arch::SSE4_1>>) {
            return Vec(_mm_sub_ps(_mm_mul_ps(a.data, b.data), c.data));
        } else if constexpr (std::is_same_v<Vec, VecF64<Arch::SSE4_1>>) {
            return Vec(_mm_sub_pd(_mm_mul_pd(a.data, b.data), c.data));
        }
    }#else
    template<typename Vec>
        requires (std::is_same_v<Vec, VecF32<Arch::GENERIC>> ||
                  std::is_same_v<Vec, VecF64<Arch::GENERIC>>)
    inline Vec fmsub(const Vec& a, const Vec& b, const Vec& c) {
        return Vec(a.data * b.data - c.data);
    }


#endif
}
