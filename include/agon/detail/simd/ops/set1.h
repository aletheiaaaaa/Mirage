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
        requires (std::is_same_v<Vec, VecI8<Arch::AVX512>> ||
                  std::is_same_v<Vec, VecI16<Arch::AVX512>> ||
                  std::is_same_v<Vec, VecI32<Arch::AVX512>> ||
                  std::is_same_v<Vec, VecI64<Arch::AVX512>> ||
                  std::is_same_v<Vec, VecF32<Arch::AVX512>> ||
                  std::is_same_v<Vec, VecF64<Arch::AVX512>>)
    inline Vec set1(typename Vec::scalar_type val) {
        if constexpr (std::is_same_v<Vec, VecI8<Arch::AVX512>>) {
            return Vec(_mm512_set1_epi8(val));
        } else if constexpr (std::is_same_v<Vec, VecI16<Arch::AVX512>>) {
            return Vec(_mm512_set1_epi16(val));
        } else if constexpr (std::is_same_v<Vec, VecI32<Arch::AVX512>>) {
            return Vec(_mm512_set1_epi32(val));
        } else if constexpr (std::is_same_v<Vec, VecI64<Arch::AVX512>>) {
            return Vec(_mm512_set1_epi64(val));
        } else if constexpr (std::is_same_v<Vec, VecF32<Arch::AVX512>>) {
            return Vec(_mm512_set1_ps(val));
        } else if constexpr (std::is_same_v<Vec, VecF64<Arch::AVX512>>) {
            return Vec(_mm512_set1_pd(val));
        }
    }

#elif defined(__AVX2__)
    template<typename Vec>
        requires (std::is_same_v<Vec, VecI8<Arch::AVX2>> ||
                  std::is_same_v<Vec, VecI16<Arch::AVX2>> ||
                  std::is_same_v<Vec, VecI32<Arch::AVX2>> ||
                  std::is_same_v<Vec, VecI64<Arch::AVX2>> ||
                  std::is_same_v<Vec, VecF32<Arch::AVX2>> ||
                  std::is_same_v<Vec, VecF64<Arch::AVX2>>)
    inline Vec set1(typename Vec::scalar_type val) {
        if constexpr (std::is_same_v<Vec, VecI8<Arch::AVX2>>) {
            return Vec(_mm256_set1_epi8(val));
        } else if constexpr (std::is_same_v<Vec, VecI16<Arch::AVX2>>) {
            return Vec(_mm256_set1_epi16(val));
        } else if constexpr (std::is_same_v<Vec, VecI32<Arch::AVX2>>) {
            return Vec(_mm256_set1_epi32(val));
        } else if constexpr (std::is_same_v<Vec, VecI64<Arch::AVX2>>) {
            return Vec(_mm256_set1_epi64x(val));
        } else if constexpr (std::is_same_v<Vec, VecF32<Arch::AVX2>>) {
            return Vec(_mm256_set1_ps(val));
        } else if constexpr (std::is_same_v<Vec, VecF64<Arch::AVX2>>) {
            return Vec(_mm256_set1_pd(val));
        }
    }

#elif defined(__SSE4_1__)
    template<typename Vec>
        requires (std::is_same_v<Vec, VecI8<Arch::SSE4_1>> ||
                  std::is_same_v<Vec, VecI16<Arch::SSE4_1>> ||
                  std::is_same_v<Vec, VecI32<Arch::SSE4_1>> ||
                  std::is_same_v<Vec, VecI64<Arch::SSE4_1>> ||
                  std::is_same_v<Vec, VecF32<Arch::SSE4_1>> ||
                  std::is_same_v<Vec, VecF64<Arch::SSE4_1>>)
    inline Vec set1(typename Vec::scalar_type val) {
        if constexpr (std::is_same_v<Vec, VecI8<Arch::SSE4_1>>) {
            return Vec(_mm_set1_epi8(val));
        } else if constexpr (std::is_same_v<Vec, VecI16<Arch::SSE4_1>>) {
            return Vec(_mm_set1_epi16(val));
        } else if constexpr (std::is_same_v<Vec, VecI32<Arch::SSE4_1>>) {
            return Vec(_mm_set1_epi32(val));
        } else if constexpr (std::is_same_v<Vec, VecI64<Arch::SSE4_1>>) {
            return Vec(_mm_set1_epi64x(val));
        } else if constexpr (std::is_same_v<Vec, VecF32<Arch::SSE4_1>>) {
            return Vec(_mm_set1_ps(val));
        } else if constexpr (std::is_same_v<Vec, VecF64<Arch::SSE4_1>>) {
            return Vec(_mm_set1_pd(val));
        }
    }#else
    template<typename Vec>
        requires (std::is_same_v<Vec, VecI8<Arch::GENERIC>> ||
                  std::is_same_v<Vec, VecI16<Arch::GENERIC>> ||
                  std::is_same_v<Vec, VecI32<Arch::GENERIC>> ||
                  std::is_same_v<Vec, VecI64<Arch::GENERIC>> ||
                  std::is_same_v<Vec, VecF32<Arch::GENERIC>> ||
                  std::is_same_v<Vec, VecF64<Arch::GENERIC>>)
    inline Vec set1(typename Vec::scalar_type val) {
        return Vec(val);
    }


#endif

    template<typename T>
        requires ScalarCastable<T>
    inline vec<T> set1(T val) {
        return set1<vec<T>>(val);
    }
}
