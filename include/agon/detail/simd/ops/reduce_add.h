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
    // Horizontal sum of all elements
    template<typename Vec>
    inline typename Vec::scalar_type reduce_add(Vec v);

#if defined(__AVX512F__)
    template<>
    inline int32_t reduce_add(VecI32<Arch::AVX512> v) {
        return _mm512_reduce_add_epi32(v.data);
    }

    template<>
    inline int64_t reduce_add(VecI64<Arch::AVX512> v) {
        return _mm512_reduce_add_epi64(v.data);
    }

#if defined(HAS_FLOAT16) && HAS_FLOAT16
    template<>
    inline std::float16_t reduce_add(VecF16<Arch::AVX512> v) {
        return _mm512_reduce_add_ph(v.data);
    }
#endif

    template<>
    inline float reduce_add(VecF32<Arch::AVX512> v) {
        return _mm512_reduce_add_ps(v.data);
    }

    template<>
    inline double reduce_add(VecF64<Arch::AVX512> v) {
        return _mm512_reduce_add_pd(v.data);
    }
#elif defined(__AVX2__)
    template<>
    inline int32_t reduce_add(VecI32<Arch::AVX2> v) {
        __m128i lo = _mm256_castsi256_si128(v.data);
        __m128i hi = _mm256_extracti128_si256(v.data, 1);
        __m128i sum = _mm_add_epi32(lo, hi);
        sum = _mm_hadd_epi32(sum, sum);
        sum = _mm_hadd_epi32(sum, sum);
        return _mm_cvtsi128_si32(sum);
    }

    template<>
    inline int64_t reduce_add(VecI64<Arch::AVX2> v) {
        __m128i lo = _mm256_castsi256_si128(v.data);
        __m128i hi = _mm256_extracti128_si256(v.data, 1);
        __m128i sum = _mm_add_epi64(lo, hi);
        sum = _mm_add_epi64(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2)));
        return _mm_cvtsi128_si64(sum);
    }

    template<>
    inline float reduce_add(VecF32<Arch::AVX2> v) {
        __m128 lo = _mm256_castps256_ps128(v.data);
        __m128 hi = _mm256_extractf128_ps(v.data, 1);
        __m128 sum = _mm_add_ps(lo, hi);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        return _mm_cvtss_f32(sum);
    }

    template<>
    inline double reduce_add(VecF64<Arch::AVX2> v) {
        __m128d lo = _mm256_castpd256_pd128(v.data);
        __m128d hi = _mm256_extractf128_pd(v.data, 1);
        __m128d sum = _mm_add_pd(lo, hi);
        sum = _mm_hadd_pd(sum, sum);
        return _mm_cvtsd_f64(sum);
    }
#elif defined(__SSE4_1__)
    template<>
    inline int32_t reduce_add(VecI32<Arch::SSE4_1> v) {
        __m128i sum = _mm_hadd_epi32(v.data, v.data);
        sum = _mm_hadd_epi32(sum, sum);
        return _mm_cvtsi128_si32(sum);
    }

    template<>
    inline int64_t reduce_add(VecI64<Arch::SSE4_1> v) {
        __m128i sum = _mm_add_epi64(v.data, _mm_shuffle_epi32(v.data, _MM_SHUFFLE(1, 0, 3, 2)));
        return _mm_cvtsi128_si64(sum);
    }

    template<>
    inline float reduce_add(VecF32<Arch::SSE4_1> v) {
        __m128 sum = _mm_hadd_ps(v.data, v.data);
        sum = _mm_hadd_ps(sum, sum);
        return _mm_cvtss_f32(sum);
    }

    template<>
    inline double reduce_add(VecF64<Arch::SSE4_1> v) {
        __m128d sum = _mm_hadd_pd(v.data, v.data);
        return _mm_cvtsd_f64(sum);
    }
#else
    template<>
    inline int32_t reduce_add(VecI32<Arch::GENERIC> v) {
        return v.data;
    }

    template<>
    inline int64_t reduce_add(VecI64<Arch::GENERIC> v) {
        return v.data;
    }

    template<>
    inline float reduce_add(VecF32<Arch::GENERIC> v) {
        return v.data;
    }

    template<>
    inline double reduce_add(VecF64<Arch::GENERIC> v) {
        return v.data;
    }
#endif
}
