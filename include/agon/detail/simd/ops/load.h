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
    inline Vec load(const typename Vec::scalar_type* ptr) {
        if constexpr (std::is_same_v<Vec, VecI8<Arch::AVX512>>) {
            return Vec(_mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr)));
        } else if constexpr (std::is_same_v<Vec, VecI16<Arch::AVX512>>) {
            return Vec(_mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr)));
        } else if constexpr (std::is_same_v<Vec, VecI32<Arch::AVX512>>) {
            return Vec(_mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr)));
        } else if constexpr (std::is_same_v<Vec, VecI64<Arch::AVX512>>) {
            return Vec(_mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr)));
        } else if constexpr (std::is_same_v<Vec, VecF32<Arch::AVX512>>) {
            return Vec(_mm512_loadu_ps(ptr));
        } else if constexpr (std::is_same_v<Vec, VecF64<Arch::AVX512>>) {
            return Vec(_mm512_loadu_pd(ptr));
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
    inline Vec load(const typename Vec::scalar_type* ptr) {
        if constexpr (std::is_same_v<Vec, VecI8<Arch::AVX2>>) {
            return Vec(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)));
        } else if constexpr (std::is_same_v<Vec, VecI16<Arch::AVX2>>) {
            return Vec(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)));
        } else if constexpr (std::is_same_v<Vec, VecI32<Arch::AVX2>>) {
            return Vec(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)));
        } else if constexpr (std::is_same_v<Vec, VecI64<Arch::AVX2>>) {
            return Vec(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)));
        } else if constexpr (std::is_same_v<Vec, VecF32<Arch::AVX2>>) {
            return Vec(_mm256_loadu_ps(ptr));
        } else if constexpr (std::is_same_v<Vec, VecF64<Arch::AVX2>>) {
            return Vec(_mm256_loadu_pd(ptr));
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
    inline Vec load(const typename Vec::scalar_type* ptr) {
        if constexpr (std::is_same_v<Vec, VecI8<Arch::SSE4_1>>) {
            return Vec(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)));
        } else if constexpr (std::is_same_v<Vec, VecI16<Arch::SSE4_1>>) {
            return Vec(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)));
        } else if constexpr (std::is_same_v<Vec, VecI32<Arch::SSE4_1>>) {
            return Vec(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)));
        } else if constexpr (std::is_same_v<Vec, VecI64<Arch::SSE4_1>>) {
            return Vec(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)));
        } else if constexpr (std::is_same_v<Vec, VecF32<Arch::SSE4_1>>) {
            return Vec(_mm_loadu_ps(ptr));
        } else if constexpr (std::is_same_v<Vec, VecF64<Arch::SSE4_1>>) {
            return Vec(_mm_loadu_pd(ptr));
        }
    }#else
    template<typename Vec>
        requires (std::is_same_v<Vec, VecI8<Arch::GENERIC>> ||
                  std::is_same_v<Vec, VecI16<Arch::GENERIC>> ||
                  std::is_same_v<Vec, VecI32<Arch::GENERIC>> ||
                  std::is_same_v<Vec, VecI64<Arch::GENERIC>> ||
                  std::is_same_v<Vec, VecF32<Arch::GENERIC>> ||
                  std::is_same_v<Vec, VecF64<Arch::GENERIC>>)
    inline Vec load(const typename Vec::scalar_type* ptr) {
        return Vec(*ptr);
    }


#endif

    template<typename T>
        requires ScalarCastable<T>
    inline vec<T> load(const T* ptr) {
        return load<vec<T>>(ptr);
    }
}
