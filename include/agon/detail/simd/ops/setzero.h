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
    // Set all elements to zero
    template<typename Vec>
    inline Vec setzero();

#if defined(__AVX512F__)
    template<>
    inline VecI8<Arch::AVX512> setzero() {
        return VecI8<Arch::AVX512>(_mm512_setzero_si512());
    }

    template<>
    inline VecI16<Arch::AVX512> setzero() {
        return VecI16<Arch::AVX512>(_mm512_setzero_si512());
    }

    template<>
    inline VecI32<Arch::AVX512> setzero() {
        return VecI32<Arch::AVX512>(_mm512_setzero_si512());
    }

    template<>
    inline VecI64<Arch::AVX512> setzero() {
        return VecI64<Arch::AVX512>(_mm512_setzero_si512());
    }

#if defined(HAS_FLOAT16) && HAS_FLOAT16
    template<>
    inline VecF16<Arch::AVX512> setzero() {
        return VecF16<Arch::AVX512>(_mm512_setzero_ph());
    }
#endif

    template<>
    inline VecF32<Arch::AVX512> setzero() {
        return VecF32<Arch::AVX512>(_mm512_setzero_ps());
    }

    template<>
    inline VecF64<Arch::AVX512> setzero() {
        return VecF64<Arch::AVX512>(_mm512_setzero_pd());
    }
#elif defined(__AVX2__)
    template<>
    inline VecI8<Arch::AVX2> setzero() {
        return VecI8<Arch::AVX2>(_mm256_setzero_si256());
    }

    template<>
    inline VecI16<Arch::AVX2> setzero() {
        return VecI16<Arch::AVX2>(_mm256_setzero_si256());
    }

    template<>
    inline VecI32<Arch::AVX2> setzero() {
        return VecI32<Arch::AVX2>(_mm256_setzero_si256());
    }

    template<>
    inline VecI64<Arch::AVX2> setzero() {
        return VecI64<Arch::AVX2>(_mm256_setzero_si256());
    }

    template<>
    inline VecF32<Arch::AVX2> setzero() {
        return VecF32<Arch::AVX2>(_mm256_setzero_ps());
    }

    template<>
    inline VecF64<Arch::AVX2> setzero() {
        return VecF64<Arch::AVX2>(_mm256_setzero_pd());
    }
#elif defined(__SSE4_1__)
    template<>
    inline VecI8<Arch::SSE4_1> setzero() {
        return VecI8<Arch::SSE4_1>(_mm_setzero_si128());
    }

    template<>
    inline VecI16<Arch::SSE4_1> setzero() {
        return VecI16<Arch::SSE4_1>(_mm_setzero_si128());
    }

    template<>
    inline VecI32<Arch::SSE4_1> setzero() {
        return VecI32<Arch::SSE4_1>(_mm_setzero_si128());
    }

    template<>
    inline VecI64<Arch::SSE4_1> setzero() {
        return VecI64<Arch::SSE4_1>(_mm_setzero_si128());
    }

    template<>
    inline VecF32<Arch::SSE4_1> setzero() {
        return VecF32<Arch::SSE4_1>(_mm_setzero_ps());
    }

    template<>
    inline VecF64<Arch::SSE4_1> setzero() {
        return VecF64<Arch::SSE4_1>(_mm_setzero_pd());
    }
#else
    template<>
    inline VecI8<Arch::GENERIC> setzero() {
        return VecI8<Arch::GENERIC>(0);
    }

    template<>
    inline VecI16<Arch::GENERIC> setzero() {
        return VecI16<Arch::GENERIC>(0);
    }

    template<>
    inline VecI32<Arch::GENERIC> setzero() {
        return VecI32<Arch::GENERIC>(0);
    }

    template<>
    inline VecI64<Arch::GENERIC> setzero() {
        return VecI64<Arch::GENERIC>(0);
    }

    template<>
    inline VecF32<Arch::GENERIC> setzero() {
        return VecF32<Arch::GENERIC>(0.0f);
    }

    template<>
    inline VecF64<Arch::GENERIC> setzero() {
        return VecF64<Arch::GENERIC>(0.0);
    }
#endif
}
