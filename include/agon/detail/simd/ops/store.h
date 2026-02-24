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
    inline void store(typename Vec::scalar_type* ptr, const Vec& v) {
        if constexpr (std::is_same_v<Vec, VecI8<Arch::AVX512>>) {
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), v.data);
        } else if constexpr (std::is_same_v<Vec, VecI16<Arch::AVX512>>) {
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), v.data);
        } else if constexpr (std::is_same_v<Vec, VecI32<Arch::AVX512>>) {
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), v.data);
        } else if constexpr (std::is_same_v<Vec, VecI64<Arch::AVX512>>) {
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), v.data);
        } else if constexpr (std::is_same_v<Vec, VecF32<Arch::AVX512>>) {
            _mm512_storeu_ps(ptr, v.data);
        } else if constexpr (std::is_same_v<Vec, VecF64<Arch::AVX512>>) {
            _mm512_storeu_pd(ptr, v.data);
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
    inline void store(typename Vec::scalar_type* ptr, const Vec& v) {
        if constexpr (std::is_same_v<Vec, VecI8<Arch::AVX2>>) {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), v.data);
        } else if constexpr (std::is_same_v<Vec, VecI16<Arch::AVX2>>) {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), v.data);
        } else if constexpr (std::is_same_v<Vec, VecI32<Arch::AVX2>>) {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), v.data);
        } else if constexpr (std::is_same_v<Vec, VecI64<Arch::AVX2>>) {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), v.data);
        } else if constexpr (std::is_same_v<Vec, VecF32<Arch::AVX2>>) {
            _mm256_storeu_ps(ptr, v.data);
        } else if constexpr (std::is_same_v<Vec, VecF64<Arch::AVX2>>) {
            _mm256_storeu_pd(ptr, v.data);
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
    inline void store(typename Vec::scalar_type* ptr, const Vec& v) {
        if constexpr (std::is_same_v<Vec, VecI8<Arch::SSE4_1>>) {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), v.data);
        } else if constexpr (std::is_same_v<Vec, VecI16<Arch::SSE4_1>>) {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), v.data);
        } else if constexpr (std::is_same_v<Vec, VecI32<Arch::SSE4_1>>) {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), v.data);
        } else if constexpr (std::is_same_v<Vec, VecI64<Arch::SSE4_1>>) {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), v.data);
        } else if constexpr (std::is_same_v<Vec, VecF32<Arch::SSE4_1>>) {
            _mm_storeu_ps(ptr, v.data);
        } else if constexpr (std::is_same_v<Vec, VecF64<Arch::SSE4_1>>) {
            _mm_storeu_pd(ptr, v.data);
        }
    }#else
    template<typename Vec>
        requires (std::is_same_v<Vec, VecI8<Arch::GENERIC>> ||
                  std::is_same_v<Vec, VecI16<Arch::GENERIC>> ||
                  std::is_same_v<Vec, VecI32<Arch::GENERIC>> ||
                  std::is_same_v<Vec, VecI64<Arch::GENERIC>> ||
                  std::is_same_v<Vec, VecF32<Arch::GENERIC>> ||
                  std::is_same_v<Vec, VecF64<Arch::GENERIC>>)
    inline void store(typename Vec::scalar_type* ptr, const Vec& v) {
        *ptr = v.data;
    }


#endif
}
