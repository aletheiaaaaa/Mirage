#pragma once

#include "../arch.h"
#include "../types.h"

#include <algorithm>
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
        requires (std::is_same_v<Vec, VecI8<Arch::AVX512>> ||
                  std::is_same_v<Vec, VecI16<Arch::AVX512>> ||
                  std::is_same_v<Vec, VecI32<Arch::AVX512>> ||
                  std::is_same_v<Vec, VecI64<Arch::AVX512>> ||
                  std::is_same_v<Vec, VecF32<Arch::AVX512>> ||
                  std::is_same_v<Vec, VecF64<Arch::AVX512>>)
    inline Vec max(const Vec& a, const Vec& b) {
        if constexpr (std::is_same_v<Vec, VecI8<Arch::AVX512>>) {
            return Vec(_mm512_max_epi8(a.data, b.data));
        } else if constexpr (std::is_same_v<Vec, VecI16<Arch::AVX512>>) {
            return Vec(_mm512_max_epi16(a.data, b.data));
        } else if constexpr (std::is_same_v<Vec, VecI32<Arch::AVX512>>) {
            return Vec(_mm512_max_epi32(a.data, b.data));
        } else if constexpr (std::is_same_v<Vec, VecI64<Arch::AVX512>>) {
            return Vec(_mm512_max_epi64(a.data, b.data));
        } else if constexpr (std::is_same_v<Vec, VecF32<Arch::AVX512>>) {
            return Vec(_mm512_max_ps(a.data, b.data));
        } else if constexpr (std::is_same_v<Vec, VecF64<Arch::AVX512>>) {
            return Vec(_mm512_max_pd(a.data, b.data));
        }
    }

#elif defined(__AVX2__)
    template<typename Vec>
        requires (std::is_same_v<Vec, VecI8<Arch::AVX2>> ||
                  std::is_same_v<Vec, VecI16<Arch::AVX2>> ||
                  std::is_same_v<Vec, VecI32<Arch::AVX2>> ||
                  std::is_same_v<Vec, VecF32<Arch::AVX2>> ||
                  std::is_same_v<Vec, VecF64<Arch::AVX2>>)
    inline Vec max(const Vec& a, const Vec& b) {
        if constexpr (std::is_same_v<Vec, VecI8<Arch::AVX2>>) {
            return Vec(_mm256_max_epi8(a.data, b.data));
        } else if constexpr (std::is_same_v<Vec, VecI16<Arch::AVX2>>) {
            return Vec(_mm256_max_epi16(a.data, b.data));
        } else if constexpr (std::is_same_v<Vec, VecI32<Arch::AVX2>>) {
            return Vec(_mm256_max_epi32(a.data, b.data));
        } else if constexpr (std::is_same_v<Vec, VecF32<Arch::AVX2>>) {
            return Vec(_mm256_max_ps(a.data, b.data));
        } else if constexpr (std::is_same_v<Vec, VecF64<Arch::AVX2>>) {
            return Vec(_mm256_max_pd(a.data, b.data));
        }
    }

#elif defined(__SSE4_1__)
    template<typename Vec>
        requires (std::is_same_v<Vec, VecI8<Arch::SSE4_1>> ||
                  std::is_same_v<Vec, VecI16<Arch::SSE4_1>> ||
                  std::is_same_v<Vec, VecI32<Arch::SSE4_1>> ||
                  std::is_same_v<Vec, VecF32<Arch::SSE4_1>> ||
                  std::is_same_v<Vec, VecF64<Arch::SSE4_1>>)
    inline Vec max(const Vec& a, const Vec& b) {
        if constexpr (std::is_same_v<Vec, VecI8<Arch::SSE4_1>>) {
            return Vec(_mm_max_epi8(a.data, b.data));
        } else if constexpr (std::is_same_v<Vec, VecI16<Arch::SSE4_1>>) {
            return Vec(_mm_max_epi16(a.data, b.data));
        } else if constexpr (std::is_same_v<Vec, VecI32<Arch::SSE4_1>>) {
            return Vec(_mm_max_epi32(a.data, b.data));
        } else if constexpr (std::is_same_v<Vec, VecF32<Arch::SSE4_1>>) {
            return Vec(_mm_max_ps(a.data, b.data));
        } else if constexpr (std::is_same_v<Vec, VecF64<Arch::SSE4_1>>) {
            return Vec(_mm_max_pd(a.data, b.data));
        }
    }#else
    template<typename Vec>
        requires (std::is_same_v<Vec, VecI8<Arch::GENERIC>> ||
                  std::is_same_v<Vec, VecI16<Arch::GENERIC>> ||
                  std::is_same_v<Vec, VecI32<Arch::GENERIC>> ||
                  std::is_same_v<Vec, VecI64<Arch::GENERIC>> ||
                  std::is_same_v<Vec, VecF32<Arch::GENERIC>> ||
                  std::is_same_v<Vec, VecF64<Arch::GENERIC>>)
    inline Vec max(const Vec& a, const Vec& b) {
        return Vec(std::max(a.data, b.data));
    }


#endif
}
