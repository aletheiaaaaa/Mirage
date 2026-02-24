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
        requires std::is_same_v<Vec, VecI16<Arch::AVX512>>
    inline Vec mulhi(const Vec& a, const Vec& b) {
        return Vec(_mm512_mulhi_epi16(a.data, b.data));
    }

#elif defined(__AVX2__)
    template<typename Vec>
        requires std::is_same_v<Vec, VecI16<Arch::AVX2>>
    inline Vec mulhi(const Vec& a, const Vec& b) {
        return Vec(_mm256_mulhi_epi16(a.data, b.data));
    }

#elif defined(__SSE4_1__)
    template<typename Vec>
        requires std::is_same_v<Vec, VecI16<Arch::SSE4_1>>
    inline Vec mulhi(const Vec& a, const Vec& b) {
        return Vec(_mm_mulhi_epi16(a.data, b.data));
    }#else
    template<typename Vec>
        requires std::is_same_v<Vec, VecI16<Arch::GENERIC>>
    inline Vec mulhi(const Vec& a, const Vec& b) {
        return Vec(static_cast<int16_t>((static_cast<int32_t>(a.data) * static_cast<int32_t>(b.data)) >> 16));
    }


#endif
}
