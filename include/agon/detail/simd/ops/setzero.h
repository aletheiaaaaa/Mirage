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
    inline Vec setzero() {
        if constexpr (std::is_same_v<Vec, VecI8<Arch::AVX512>> ||
                      std::is_same_v<Vec, VecI16<Arch::AVX512>> ||
                      std::is_same_v<Vec, VecI32<Arch::AVX512>> ||
                      std::is_same_v<Vec, VecI64<Arch::AVX512>>) {
            return Vec(_mm512_setzero_si512());
        } else if constexpr (std::is_same_v<Vec, VecF32<Arch::AVX512>>) {
            return Vec(_mm512_setzero_ps());
        } else if constexpr (std::is_same_v<Vec, VecF64<Arch::AVX512>>) {
            return Vec(_mm512_setzero_pd());
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
    inline Vec setzero() {
        if constexpr (std::is_same_v<Vec, VecI8<Arch::AVX2>> ||
                      std::is_same_v<Vec, VecI16<Arch::AVX2>> ||
                      std::is_same_v<Vec, VecI32<Arch::AVX2>> ||
                      std::is_same_v<Vec, VecI64<Arch::AVX2>>) {
            return Vec(_mm256_setzero_si256());
        } else if constexpr (std::is_same_v<Vec, VecF32<Arch::AVX2>>) {
            return Vec(_mm256_setzero_ps());
        } else if constexpr (std::is_same_v<Vec, VecF64<Arch::AVX2>>) {
            return Vec(_mm256_setzero_pd());
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
    inline Vec setzero() {
        if constexpr (std::is_same_v<Vec, VecI8<Arch::SSE4_1>> ||
                      std::is_same_v<Vec, VecI16<Arch::SSE4_1>> ||
                      std::is_same_v<Vec, VecI32<Arch::SSE4_1>> ||
                      std::is_same_v<Vec, VecI64<Arch::SSE4_1>>) {
            return Vec(_mm_setzero_si128());
        } else if constexpr (std::is_same_v<Vec, VecF32<Arch::SSE4_1>>) {
            return Vec(_mm_setzero_ps());
        } else if constexpr (std::is_same_v<Vec, VecF64<Arch::SSE4_1>>) {
            return Vec(_mm_setzero_pd());
        }
    }#else
    template<typename Vec>
        requires (std::is_same_v<Vec, VecI8<Arch::GENERIC>> ||
                  std::is_same_v<Vec, VecI16<Arch::GENERIC>> ||
                  std::is_same_v<Vec, VecI32<Arch::GENERIC>> ||
                  std::is_same_v<Vec, VecI64<Arch::GENERIC>> ||
                  std::is_same_v<Vec, VecF32<Arch::GENERIC>> ||
                  std::is_same_v<Vec, VecF64<Arch::GENERIC>>)
    inline Vec setzero() {
        if constexpr (std::is_same_v<Vec, VecI8<Arch::GENERIC>> ||
                      std::is_same_v<Vec, VecI16<Arch::GENERIC>> ||
                      std::is_same_v<Vec, VecI32<Arch::GENERIC>> ||
                      std::is_same_v<Vec, VecI64<Arch::GENERIC>>) {
            return Vec(0);
        } else if constexpr (std::is_same_v<Vec, VecF32<Arch::GENERIC>>) {
            return Vec(0.0f);
        } else if constexpr (std::is_same_v<Vec, VecF64<Arch::GENERIC>>) {
            return Vec(0.0);
        }
    }


#endif

    template<typename T>
        requires ScalarCastable<T>
    inline vec<T> setzero() {
        return setzero<vec<T>>();
    }
}
