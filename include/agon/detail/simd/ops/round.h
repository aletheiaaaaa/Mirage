#pragma once

#include "../arch.h"
#include "../types.h"

#include <cmath>
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
    inline Vec round(Vec v) {
        if constexpr (std::is_same_v<Vec, VecF32<Arch::AVX512>>) {
            return Vec(_mm512_roundscale_ps(v.data, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        } else if constexpr (std::is_same_v<Vec, VecF64<Arch::AVX512>>) {
            return Vec(_mm512_roundscale_pd(v.data, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
    }
#elif defined(__AVX2__)
    template<typename Vec>
        requires (std::is_same_v<Vec, VecF32<Arch::AVX2>> ||
                  std::is_same_v<Vec, VecF64<Arch::AVX2>>)
    inline Vec round(Vec v) {
        if constexpr (std::is_same_v<Vec, VecF32<Arch::AVX2>>) {
            return Vec(_mm256_round_ps(v.data, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        } else if constexpr (std::is_same_v<Vec, VecF64<Arch::AVX2>>) {
            return Vec(_mm256_round_pd(v.data, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
    }
#elif defined(__SSE4_1__)
    template<typename Vec>
        requires (std::is_same_v<Vec, VecF32<Arch::SSE4_1>> ||
                  std::is_same_v<Vec, VecF64<Arch::SSE4_1>>)
    inline Vec round(Vec v) {
        if constexpr (std::is_same_v<Vec, VecF32<Arch::SSE4_1>>) {
            return Vec(_mm_round_ps(v.data, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        } else if constexpr (std::is_same_v<Vec, VecF64<Arch::SSE4_1>>) {
            return Vec(_mm_round_pd(v.data, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
    }#else
    template<typename Vec>
        requires (std::is_same_v<Vec, VecF32<Arch::GENERIC>> ||
                  std::is_same_v<Vec, VecF64<Arch::GENERIC>>)
    inline Vec round(Vec v) {
        return Vec(std::round(v.data));
    }
#endif
}
