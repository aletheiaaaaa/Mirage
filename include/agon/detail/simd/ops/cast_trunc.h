#pragma once

#include "../arch.h"
#include "../types.h"

#include <cmath>
#include <concepts>
#include <type_traits>
#if defined(__AVX512F__)
    #include <immintrin.h>
#elif defined(__AVX2__)
    #include <immintrin.h>
#elif defined(__SSE4_1__)
    #include <smmintrin.h>
#endif

namespace agon::simd {
    template<typename T, size_t N = 0, typename F>
    inline T cast_trunc(F v);
#if defined(__AVX512F__)
    template<typename T, size_t N, typename F>
        requires IsVec<T> && IsVec<F>
    inline T cast_trunc(F v) {
        constexpr Arch arch = Arch::AVX512;

        if constexpr (std::is_same_v<F, VecI8<arch>>) {
            if constexpr (std::is_same_v<T, VecI16<arch>>) {
                __m256i chunk = _mm512_extracti64x4_epi64(v.data, N);
                return T(_mm512_cvtepi8_epi16(chunk));

            } else if constexpr (std::is_same_v<T, VecI32<arch>>) {
                __m128i chunk = _mm512_extracti32x4_epi32(v.data, N);
                return T(_mm512_cvtepi8_epi32(chunk));

            } else if constexpr (std::is_same_v<T, VecI64<arch>>) {
                __m128i lane = _mm512_extracti32x4_epi32(v.data, N / 2);
                __m128i chunk = (N % 2 == 0) ? lane : _mm_srli_si128(lane, 8);
                return T(_mm512_cvtepi8_epi64(chunk));

            } else if constexpr (std::is_same_v<T, VecF32<arch>>) {
                __m128i chunk = _mm512_extracti32x4_epi32(v.data, N);
                return T(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(chunk)));

            } else if constexpr (std::is_same_v<T, VecF64<arch>>) {
                __m128i lane = _mm512_extracti32x4_epi32(v.data, N / 2);
                __m128i chunk = (N % 2 == 0) ? lane : _mm_srli_si128(lane, 8);
                __m512i i64 = _mm512_cvtepi8_epi64(chunk);
                return T(_mm512_cvtepi64_pd(i64));

            } else {
                static_assert(std::is_same_v<T, void>, "Unsupported cast_trunc from VecI8<AVX512>");
            }
        }

        else if constexpr (std::is_same_v<F, VecI16<arch>>) {
            if constexpr (std::is_same_v<T, VecI32<arch>>) {
                __m256i chunk = _mm512_extracti64x4_epi64(v.data, N);
                return T(_mm512_cvtepi16_epi32(chunk));

            } else if constexpr (std::is_same_v<T, VecI64<arch>>) {
                __m128i chunk = _mm512_extracti32x4_epi32(v.data, N);
                return T(_mm512_cvtepi16_epi64(chunk));

            } else if constexpr (std::is_same_v<T, VecF32<arch>>) {
                __m256i chunk = _mm512_extracti64x4_epi64(v.data, N);
                return T(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(chunk)));

            } else if constexpr (std::is_same_v<T, VecF64<arch>>) {
                __m128i chunk = _mm512_extracti32x4_epi32(v.data, N);
                __m512i i64 = _mm512_cvtepi16_epi64(chunk);
                return T(_mm512_cvtepi64_pd(i64));

            } else if constexpr (std::is_same_v<T, VecI8<arch>>) {
                __m256i narrow = _mm512_cvtepi16_epi8(v.data);
                return T(_mm512_castsi256_si512(narrow));

            } else {
                static_assert(std::is_same_v<T, void>, "Unsupported cast_trunc from VecI16<AVX512>");
            }
        }

        else if constexpr (std::is_same_v<F, VecI32<arch>>) {
            if constexpr (std::is_same_v<T, VecI64<arch>>) {
                __m256i chunk = _mm512_extracti64x4_epi64(v.data, N);
                return T(_mm512_cvtepi32_epi64(chunk));

            } else if constexpr (std::is_same_v<T, VecF32<arch>>) {
                return T(_mm512_cvtepi32_ps(v.data));

            } else if constexpr (std::is_same_v<T, VecF64<arch>>) {
                __m256i chunk = _mm512_extracti64x4_epi64(v.data, N);
                return T(_mm512_cvtepi32_pd(chunk));

            } else if constexpr (std::is_same_v<T, VecI16<arch>>) {
                __m256i narrow = _mm512_cvtepi32_epi16(v.data);
                return T(_mm512_castsi256_si512(narrow));

            } else if constexpr (std::is_same_v<T, VecI8<arch>>) {
                __m128i narrow = _mm512_cvtepi32_epi8(v.data);
                return T(_mm512_castsi128_si512(narrow));

            } else {
                static_assert(std::is_same_v<T, void>, "Unsupported cast_trunc from VecI32<AVX512>");
            }
        }

        else if constexpr (std::is_same_v<F, VecI64<arch>>) {
            if constexpr (std::is_same_v<T, VecI32<arch>>) {
                __m256i narrow = _mm512_cvtepi64_epi32(v.data);
                return T(_mm512_castsi256_si512(narrow));

            } else if constexpr (std::is_same_v<T, VecI16<arch>>) {
                __m128i narrow = _mm512_cvtepi64_epi16(v.data);
                return T(_mm512_castsi128_si512(narrow));

            } else if constexpr (std::is_same_v<T, VecI8<arch>>) {
                __m128i narrow = _mm512_cvtepi64_epi8(v.data);
                return T(_mm512_castsi128_si512(narrow));

            } else {
                static_assert(std::is_same_v<T, void>, "Unsupported cast_trunc from VecI64<AVX512>");
            }
        }

        else if constexpr (std::is_same_v<F, VecF32<arch>>) {
            if constexpr (std::is_same_v<T, VecF64<arch>>) {
                __m256 chunk = (N == 0) ? _mm512_castps512_ps256(v.data) : _mm512_extractf32x8_ps(v.data, 1);
                return T(_mm512_cvtps_pd(chunk));

            } else if constexpr (std::is_same_v<T, VecI32<arch>>) {
                return T(_mm512_cvttps_epi32(v.data));

            } else if constexpr (std::is_same_v<T, VecI64<arch>>) {
                __m256 chunk = (N == 0) ? _mm512_castps512_ps256(v.data) : _mm512_extractf32x8_ps(v.data, 1);
                return T(_mm512_cvttps_epi64(chunk));

            } else if constexpr (std::is_same_v<T, VecI16<arch>>) {
                __m512i i32 = _mm512_cvttps_epi32(v.data);
                __m256i narrow = _mm512_cvtepi32_epi16(i32);
                return T(_mm512_castsi256_si512(narrow));

            } else if constexpr (std::is_same_v<T, VecI8<arch>>) {
                __m512i i32 = _mm512_cvttps_epi32(v.data);
                __m128i narrow = _mm512_cvtepi32_epi8(i32);
                return T(_mm512_castsi128_si512(narrow));

            } else {
                static_assert(std::is_same_v<T, void>, "Unsupported cast_trunc from VecF32<AVX512>");
            }
        }

        else if constexpr (std::is_same_v<F, VecF64<arch>>) {
            if constexpr (std::is_same_v<T, VecF32<arch>>) {
                __m256 narrow = _mm512_cvtpd_ps(v.data);
                return T(_mm512_castps256_ps512(narrow));

            } else if constexpr (std::is_same_v<T, VecI64<arch>>) {
                return T(_mm512_cvttpd_epi64(v.data));

            } else if constexpr (std::is_same_v<T, VecI32<arch>>) {
                __m256i narrow = _mm512_cvttpd_epi32(v.data);
                return T(_mm512_castsi256_si512(narrow));

            } else if constexpr (std::is_same_v<T, VecI16<arch>>) {
                __m256i i32 = _mm512_cvttpd_epi32(v.data);
                __m128i narrow = _mm256_cvtepi32_epi16(i32);
                return T(_mm512_castsi128_si512(narrow));

            } else if constexpr (std::is_same_v<T, VecI8<arch>>) {
                __m256i i32 = _mm512_cvttpd_epi32(v.data);
                __m128i narrow = _mm256_cvtepi32_epi8(i32);
                return T(_mm512_castsi128_si512(narrow));

            } else {
                static_assert(std::is_same_v<T, void>, "Unsupported cast_trunc from VecF64<AVX512>");
            }
        }
        else {
            static_assert(std::is_same_v<F, void>, "Unsupported source type for cast_trunc on AVX512");
        }
    }
#elif defined(__AVX2__)
    template<typename T, size_t N, typename F>
        requires IsVec<T> && IsVec<F>
    inline T cast_trunc(F v) {
        constexpr Arch arch = Arch::AVX2;

        if constexpr (std::is_same_v<F, VecI8<arch>>) {
            if constexpr (std::is_same_v<T, VecI16<arch>>) {
                __m128i chunk = _mm256_extracti128_si256(v.data, N);
                return T(_mm256_cvtepi8_epi16(chunk));

            } else if constexpr (std::is_same_v<T, VecI32<arch>>) {
                __m128i lane = _mm256_extracti128_si256(v.data, N / 2);
                __m128i chunk = (N % 2 == 0) ? lane : _mm_srli_si128(lane, 8);
                return T(_mm256_cvtepi8_epi32(chunk));

            } else if constexpr (std::is_same_v<T, VecI64<arch>>) {
                __m128i lane = _mm256_extracti128_si256(v.data, N / 4);
                __m128i chunk = _mm_srli_si128(lane, (N % 4) * 4);
                return T(_mm256_cvtepi8_epi64(chunk));

            } else if constexpr (std::is_same_v<T, VecF32<arch>>) {
                __m128i lane = _mm256_extracti128_si256(v.data, N / 2);
                __m128i chunk = (N % 2 == 0) ? lane : _mm_srli_si128(lane, 8);
                return T(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(chunk)));

            } else if constexpr (std::is_same_v<T, VecF64<arch>>) {
                __m128i lane = _mm256_extracti128_si256(v.data, N / 4);
                __m128i chunk = _mm_srli_si128(lane, (N % 4) * 4);
                __m256i i64 = _mm256_cvtepi8_epi64(chunk);
                return T(_mm256_cvtepi32_pd(_mm256_castsi256_si128(i64)));

            } else {
                static_assert(std::is_same_v<T, void>, "Unsupported cast_trunc from VecI8<AVX2>");
            }
        }

        else if constexpr (std::is_same_v<F, VecI16<arch>>) {
            if constexpr (std::is_same_v<T, VecI32<arch>>) {
                __m128i chunk = _mm256_extracti128_si256(v.data, N);
                return T(_mm256_cvtepi16_epi32(chunk));

            } else if constexpr (std::is_same_v<T, VecI64<arch>>) {
                __m128i lane = _mm256_extracti128_si256(v.data, N / 2);
                __m128i chunk = (N % 2 == 0) ? lane : _mm_srli_si128(lane, 8);
                return T(_mm256_cvtepi16_epi64(chunk));

            } else if constexpr (std::is_same_v<T, VecF32<arch>>) {
                __m128i chunk = _mm256_extracti128_si256(v.data, N);
                return T(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(chunk)));

            } else if constexpr (std::is_same_v<T, VecF64<arch>>) {
                __m128i lane = _mm256_extracti128_si256(v.data, N / 2);
                __m128i chunk = (N % 2 == 0) ? lane : _mm_srli_si128(lane, 8);
                __m256i i64 = _mm256_cvtepi16_epi64(chunk);
                return T(_mm256_cvtepi32_pd(_mm256_castsi256_si128(i64)));

            } else if constexpr (std::is_same_v<T, VecI8<arch>>) {
                __m256i shuf = _mm256_set_epi8(
                    -1, -1, -1, -1, -1, -1, -1, -1, 14, 12, 10, 8, 6, 4, 2, 0,
                    -1, -1, -1, -1, -1, -1, -1, -1, 14, 12, 10, 8, 6, 4, 2, 0
                );
                __m256i shuffled = _mm256_shuffle_epi8(v.data, shuf);
                return T(_mm256_permute4x64_epi64(shuffled, _MM_SHUFFLE(3, 3, 2, 0)));

            } else {
                static_assert(std::is_same_v<T, void>, "Unsupported cast_trunc from VecI16<AVX2>");
            }
        }

        else if constexpr (std::is_same_v<F, VecI32<arch>>) {
            if constexpr (std::is_same_v<T, VecI64<arch>>) {
                __m128i chunk = _mm256_extracti128_si256(v.data, N);
                return T(_mm256_cvtepi32_epi64(chunk));

            } else if constexpr (std::is_same_v<T, VecF32<arch>>) {
                return T(_mm256_cvtepi32_ps(v.data));

            } else if constexpr (std::is_same_v<T, VecF64<arch>>) {
                __m128i chunk = _mm256_extracti128_si256(v.data, N);
                return T(_mm256_cvtepi32_pd(chunk));

            } else if constexpr (std::is_same_v<T, VecI16<arch>>) {
                __m256i shuf = _mm256_set_epi8(
                    -1, -1, -1, -1, -1, -1, -1, -1, 13, 12, 9, 8, 5, 4, 1, 0,
                    -1, -1, -1, -1, -1, -1, -1, -1, 13, 12, 9, 8, 5, 4, 1, 0
                );
                __m256i shuffled = _mm256_shuffle_epi8(v.data, shuf);
                return T(_mm256_permute4x64_epi64(shuffled, _MM_SHUFFLE(3, 3, 2, 0)));

            } else if constexpr (std::is_same_v<T, VecI8<arch>>) {
                __m128i lo = _mm256_castsi256_si128(v.data);
                __m128i hi = _mm256_extracti128_si256(v.data, 1);
                __m128i shuf = _mm_set_epi8(
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0
                );
                lo = _mm_shuffle_epi8(lo, shuf);
                hi = _mm_shuffle_epi8(hi, shuf);
                return T(_mm256_castsi128_si256(_mm_unpacklo_epi32(lo, hi)));

            } else {
                static_assert(std::is_same_v<T, void>, "Unsupported cast_trunc from VecI32<AVX2>");
            }
        }

        else if constexpr (std::is_same_v<F, VecI64<arch>>) {
            if constexpr (std::is_same_v<T, VecI32<arch>>) {
                __m256i shuffled = _mm256_shuffle_epi32(v.data, _MM_SHUFFLE(2, 0, 2, 0));
                return T(_mm256_permute4x64_epi64(shuffled, _MM_SHUFFLE(3, 3, 2, 0)));

            } else if constexpr (std::is_same_v<T, VecI16<arch>>) {
                __m128i lo = _mm256_castsi256_si128(v.data);
                __m128i hi = _mm256_extracti128_si256(v.data, 1);
                __m128i shuf = _mm_set_epi8(
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9, 8, 1, 0
                );
                lo = _mm_shuffle_epi8(lo, shuf);
                hi = _mm_shuffle_epi8(hi, shuf);
                return T(_mm256_castsi128_si256(_mm_unpacklo_epi32(lo, hi)));

            } else if constexpr (std::is_same_v<T, VecI8<arch>>) {
                __m128i lo = _mm256_castsi256_si128(v.data);
                __m128i hi = _mm256_extracti128_si256(v.data, 1);
                __m128i shuf = _mm_set_epi8(
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 0
                );
                lo = _mm_shuffle_epi8(lo, shuf);
                hi = _mm_shuffle_epi8(hi, shuf);
                return T(_mm256_castsi128_si256(_mm_unpacklo_epi16(lo, hi)));

            } else {
                static_assert(std::is_same_v<T, void>, "Unsupported cast_trunc from VecI64<AVX2>");
            }
        }

        else if constexpr (std::is_same_v<F, VecF32<arch>>) {
            if constexpr (std::is_same_v<T, VecF64<arch>>) {
                __m128 chunk = (N == 0) ? _mm256_castps256_ps128(v.data) : _mm256_extractf128_ps(v.data, 1);
                return T(_mm256_cvtps_pd(chunk));

            } else if constexpr (std::is_same_v<T, VecI32<arch>>) {
                return T(_mm256_cvttps_epi32(v.data));

            } else if constexpr (std::is_same_v<T, VecI16<arch>>) {
                __m256i i32 = _mm256_cvttps_epi32(v.data);
                __m256i shuf = _mm256_set_epi8(
                    -1, -1, -1, -1, -1, -1, -1, -1, 13, 12, 9, 8, 5, 4, 1, 0,
                    -1, -1, -1, -1, -1, -1, -1, -1, 13, 12, 9, 8, 5, 4, 1, 0
                );
                __m256i shuffled = _mm256_shuffle_epi8(i32, shuf);
                return T(_mm256_permute4x64_epi64(shuffled, _MM_SHUFFLE(3, 3, 2, 0)));

            } else if constexpr (std::is_same_v<T, VecI8<arch>>) {
                __m256i i32 = _mm256_cvttps_epi32(v.data);
                __m128i lo = _mm256_castsi256_si128(i32);
                __m128i hi = _mm256_extracti128_si256(i32, 1);
                __m128i shuf = _mm_set_epi8(
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0
                );
                lo = _mm_shuffle_epi8(lo, shuf);
                hi = _mm_shuffle_epi8(hi, shuf);
                return T(_mm256_castsi128_si256(_mm_unpacklo_epi32(lo, hi)));

            } else {
                static_assert(std::is_same_v<T, void>, "Unsupported cast_trunc from VecF32<AVX2>");
            }
        }

        else if constexpr (std::is_same_v<F, VecF64<arch>>) {
            if constexpr (std::is_same_v<T, VecF32<arch>>) {
                __m128 narrow = _mm256_cvtpd_ps(v.data);
                return T(_mm256_castps128_ps256(narrow));

            } else if constexpr (std::is_same_v<T, VecI32<arch>>) {
                __m128i narrow = _mm256_cvttpd_epi32(v.data);
                return T(_mm256_castsi128_si256(narrow));

            } else if constexpr (std::is_same_v<T, VecI16<arch>>) {
                __m128i i32 = _mm256_cvttpd_epi32(v.data);
                __m128i shuf = _mm_set_epi8(
                    -1, -1, -1, -1, -1, -1, -1, -1, 13, 12, 9, 8, 5, 4, 1, 0
                );
                return T(_mm256_castsi128_si256(_mm_shuffle_epi8(i32, shuf)));

            } else if constexpr (std::is_same_v<T, VecI8<arch>>) {
                __m128i i32 = _mm256_cvttpd_epi32(v.data);
                __m128i shuf = _mm_set_epi8(
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0
                );
                return T(_mm256_castsi128_si256(_mm_shuffle_epi8(i32, shuf)));

            } else {
                static_assert(std::is_same_v<T, void>, "Unsupported cast_trunc from VecF64<AVX2>");
            }
        }
        else {
            static_assert(std::is_same_v<F, void>, "Unsupported source type for cast_trunc on AVX2");
        }
    }
#elif defined(__SSE4_1__)
    template<typename T, size_t N, typename F>
        requires IsVec<T> && IsVec<F>
    inline T cast_trunc(F v) {
        constexpr Arch arch = Arch::SSE4_1;

        if constexpr (std::is_same_v<F, VecI8<arch>>) {
            if constexpr (std::is_same_v<T, VecI16<arch>>) {
                __m128i chunk = (N == 0) ? v.data : _mm_srli_si128(v.data, 8);
                return T(_mm_cvtepi8_epi16(chunk));

            } else if constexpr (std::is_same_v<T, VecI32<arch>>) {
                __m128i chunk = _mm_srli_si128(v.data, N * 4);
                return T(_mm_cvtepi8_epi32(chunk));

            } else if constexpr (std::is_same_v<T, VecI64<arch>>) {
                __m128i chunk = _mm_srli_si128(v.data, N * 2);
                return T(_mm_cvtepi8_epi64(chunk));

            } else if constexpr (std::is_same_v<T, VecF32<arch>>) {
                __m128i chunk = _mm_srli_si128(v.data, N * 4);
                return T(_mm_cvtepi32_ps(_mm_cvtepi8_epi32(chunk)));

            } else if constexpr (std::is_same_v<T, VecF64<arch>>) {
                __m128i chunk = _mm_srli_si128(v.data, N * 2);
                return T(_mm_cvtepi32_pd(_mm_cvtepi8_epi32(chunk)));

            } else {
                static_assert(std::is_same_v<T, void>, "Unsupported cast_trunc from VecI8<SSE4_1>");
            }
        }

        else if constexpr (std::is_same_v<F, VecI16<arch>>) {
            if constexpr (std::is_same_v<T, VecI32<arch>>) {
                __m128i chunk = (N == 0) ? v.data : _mm_srli_si128(v.data, 8);
                return T(_mm_cvtepi16_epi32(chunk));

            } else if constexpr (std::is_same_v<T, VecI64<arch>>) {
                __m128i chunk = _mm_srli_si128(v.data, N * 4);
                return T(_mm_cvtepi16_epi64(chunk));

            } else if constexpr (std::is_same_v<T, VecF32<arch>>) {
                __m128i chunk = (N == 0) ? v.data : _mm_srli_si128(v.data, 8);
                return T(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(chunk)));

            } else if constexpr (std::is_same_v<T, VecF64<arch>>) {
                __m128i chunk = _mm_srli_si128(v.data, N * 4);
                return T(_mm_cvtepi32_pd(_mm_cvtepi16_epi32(chunk)));

            } else if constexpr (std::is_same_v<T, VecI8<arch>>) {
                __m128i shuf = _mm_set_epi8(
                    -1, -1, -1, -1, -1, -1, -1, -1, 14, 12, 10, 8, 6, 4, 2, 0
                );
                return T(_mm_shuffle_epi8(v.data, shuf));

            } else {
                static_assert(std::is_same_v<T, void>, "Unsupported cast_trunc from VecI16<SSE4_1>");
            }
        }

        else if constexpr (std::is_same_v<F, VecI32<arch>>) {
            if constexpr (std::is_same_v<T, VecI64<arch>>) {
                __m128i chunk = (N == 0) ? v.data : _mm_srli_si128(v.data, 8);
                return T(_mm_cvtepi32_epi64(chunk));

            } else if constexpr (std::is_same_v<T, VecF32<arch>>) {
                return T(_mm_cvtepi32_ps(v.data));

            } else if constexpr (std::is_same_v<T, VecF64<arch>>) {
                __m128i chunk = (N == 0) ? v.data : _mm_srli_si128(v.data, 8);
                return T(_mm_cvtepi32_pd(chunk));

            } else if constexpr (std::is_same_v<T, VecI16<arch>>) {
                __m128i shuf = _mm_set_epi8(
                    -1, -1, -1, -1, -1, -1, -1, -1, 13, 12, 9, 8, 5, 4, 1, 0
                );
                return T(_mm_shuffle_epi8(v.data, shuf));

            } else if constexpr (std::is_same_v<T, VecI8<arch>>) {
                __m128i shuf = _mm_set_epi8(
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0
                );
                return T(_mm_shuffle_epi8(v.data, shuf));

            } else {
                static_assert(std::is_same_v<T, void>, "Unsupported cast_trunc from VecI32<SSE4_1>");
            }
        }

        else if constexpr (std::is_same_v<F, VecI64<arch>>) {
            if constexpr (std::is_same_v<T, VecI32<arch>>) {
                __m128i shuffled = _mm_shuffle_epi32(v.data, _MM_SHUFFLE(3, 3, 2, 0));
                return T(shuffled);

            } else if constexpr (std::is_same_v<T, VecI16<arch>>) {
                __m128i shuf = _mm_set_epi8(
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9, 8, 1, 0
                );
                return T(_mm_shuffle_epi8(v.data, shuf));

            } else if constexpr (std::is_same_v<T, VecI8<arch>>) {
                __m128i shuf = _mm_set_epi8(
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 0
                );
                return T(_mm_shuffle_epi8(v.data, shuf));

            } else {
                static_assert(std::is_same_v<T, void>, "Unsupported cast_trunc from VecI64<SSE4_1>");
            }
        }

        else if constexpr (std::is_same_v<F, VecF32<arch>>) {
            if constexpr (std::is_same_v<T, VecF64<arch>>) {
                __m128 chunk = (N == 0) ? v.data : _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(v.data), 8));
                return T(_mm_cvtps_pd(chunk));

            } else if constexpr (std::is_same_v<T, VecI32<arch>>) {
                return T(_mm_cvttps_epi32(v.data));

            } else if constexpr (std::is_same_v<T, VecI16<arch>>) {
                __m128i i32 = _mm_cvttps_epi32(v.data);
                __m128i shuf = _mm_set_epi8(
                    -1, -1, -1, -1, -1, -1, -1, -1, 13, 12, 9, 8, 5, 4, 1, 0
                );
                return T(_mm_shuffle_epi8(i32, shuf));

            } else if constexpr (std::is_same_v<T, VecI8<arch>>) {
                __m128i i32 = _mm_cvttps_epi32(v.data);
                __m128i shuf = _mm_set_epi8(
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0
                );
                return T(_mm_shuffle_epi8(i32, shuf));

            } else {
                static_assert(std::is_same_v<T, void>, "Unsupported cast_trunc from VecF32<SSE4_1>");
            }
        }

        else if constexpr (std::is_same_v<F, VecF64<arch>>) {
            if constexpr (std::is_same_v<T, VecF32<arch>>) {
                return T(_mm_cvtpd_ps(v.data));

            } else if constexpr (std::is_same_v<T, VecI32<arch>>) {
                return T(_mm_cvttpd_epi32(v.data));

            } else if constexpr (std::is_same_v<T, VecI16<arch>>) {
                __m128i i32 = _mm_cvttpd_epi32(v.data);
                __m128i shuf = _mm_set_epi8(
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 4, 1, 0
                );
                return T(_mm_shuffle_epi8(i32, shuf));

            } else if constexpr (std::is_same_v<T, VecI8<arch>>) {
                __m128i i32 = _mm_cvttpd_epi32(v.data);
                __m128i shuf = _mm_set_epi8(
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 0
                );
                return T(_mm_shuffle_epi8(i32, shuf));

            } else {
                static_assert(std::is_same_v<T, void>, "Unsupported cast_trunc from VecF64<SSE4_1>");
            }
        }
        else {
            static_assert(std::is_same_v<F, void>, "Unsupported source type for cast_trunc on SSE4_1");
        }
    }
#else
    template<typename T, size_t N, typename F>
        requires IsVec<T> && IsVec<F>
    inline T cast_trunc(F v) {
        constexpr Arch arch = Arch::GENERIC;

        if constexpr (std::is_same_v<F, VecI8<arch>>) {
            if constexpr (std::is_same_v<T, VecI16<arch>>) {
                return T(static_cast<int16_t>(v.data));
            } else if constexpr (std::is_same_v<T, VecI32<arch>>) {
                return T(static_cast<int32_t>(v.data));
            } else if constexpr (std::is_same_v<T, VecI64<arch>>) {
                return T(static_cast<int64_t>(v.data));
            } else if constexpr (std::is_same_v<T, VecF32<arch>>) {
                return T(static_cast<float>(v.data));
            } else if constexpr (std::is_same_v<T, VecF64<arch>>) {
                return T(static_cast<double>(v.data));
            } else {
                static_assert(std::is_same_v<T, void>, "Unsupported cast_trunc from VecI8<GENERIC>");
            }
        }
        else if constexpr (std::is_same_v<F, VecI16<arch>>) {
            if constexpr (std::is_same_v<T, VecI32<arch>>) {
                return T(static_cast<int32_t>(v.data));
            } else if constexpr (std::is_same_v<T, VecI64<arch>>) {
                return T(static_cast<int64_t>(v.data));
            } else if constexpr (std::is_same_v<T, VecF32<arch>>) {
                return T(static_cast<float>(v.data));
            } else if constexpr (std::is_same_v<T, VecF64<arch>>) {
                return T(static_cast<double>(v.data));
            } else if constexpr (std::is_same_v<T, VecI8<arch>>) {
                return T(static_cast<int8_t>(v.data));
            } else {
                static_assert(std::is_same_v<T, void>, "Unsupported cast_trunc from VecI16<GENERIC>");
            }
        }
        else if constexpr (std::is_same_v<F, VecI32<arch>>) {
            if constexpr (std::is_same_v<T, VecI64<arch>>) {
                return T(static_cast<int64_t>(v.data));
            } else if constexpr (std::is_same_v<T, VecF32<arch>>) {
                return T(static_cast<float>(v.data));
            } else if constexpr (std::is_same_v<T, VecF64<arch>>) {
                return T(static_cast<double>(v.data));
            } else if constexpr (std::is_same_v<T, VecI16<arch>>) {
                return T(static_cast<int16_t>(v.data));
            } else if constexpr (std::is_same_v<T, VecI8<arch>>) {
                return T(static_cast<int8_t>(v.data));
            } else {
                static_assert(std::is_same_v<T, void>, "Unsupported cast_trunc from VecI32<GENERIC>");
            }
        }
        else if constexpr (std::is_same_v<F, VecI64<arch>>) {
            if constexpr (std::is_same_v<T, VecI32<arch>>) {
                return T(static_cast<int32_t>(v.data));
            } else if constexpr (std::is_same_v<T, VecI16<arch>>) {
                return T(static_cast<int16_t>(v.data));
            } else if constexpr (std::is_same_v<T, VecI8<arch>>) {
                return T(static_cast<int8_t>(v.data));
            } else {
                static_assert(std::is_same_v<T, void>, "Unsupported cast_trunc from VecI64<GENERIC>");
            }
        }

        else if constexpr (std::is_same_v<F, VecF32<arch>>) {
            if constexpr (std::is_same_v<T, VecF64<arch>>) {
                return T(static_cast<double>(v.data));
            } else if constexpr (std::is_same_v<T, VecI32<arch>>) {
                return T(static_cast<int32_t>(v.data));
            } else if constexpr (std::is_same_v<T, VecI64<arch>>) {
                return T(static_cast<int64_t>(v.data));
            } else if constexpr (std::is_same_v<T, VecI16<arch>>) {
                return T(static_cast<int16_t>(v.data));
            } else if constexpr (std::is_same_v<T, VecI8<arch>>) {
                return T(static_cast<int8_t>(v.data));
            } else {
                static_assert(std::is_same_v<T, void>, "Unsupported cast_trunc from VecF32<GENERIC>");
            }
        }
        else if constexpr (std::is_same_v<F, VecF64<arch>>) {
            if constexpr (std::is_same_v<T, VecF32<arch>>) {
                return T(static_cast<float>(v.data));
            } else if constexpr (std::is_same_v<T, VecI64<arch>>) {
                return T(static_cast<int64_t>(v.data));
            } else if constexpr (std::is_same_v<T, VecI32<arch>>) {
                return T(static_cast<int32_t>(v.data));
            } else if constexpr (std::is_same_v<T, VecI16<arch>>) {
                return T(static_cast<int16_t>(v.data));
            } else if constexpr (std::is_same_v<T, VecI8<arch>>) {
                return T(static_cast<int8_t>(v.data));
            } else {
                static_assert(std::is_same_v<T, void>, "Unsupported cast_trunc from VecF64<GENERIC>");
            }
        }
        else {
            static_assert(std::is_same_v<F, void>, "Unsupported source type for cast_trunc on GENERIC");
        }
    }

#endif

    template<typename T, size_t N = 0, typename F>
        requires ScalarCastable<T>
    inline vec<T> cast_trunc(F v) {
        return cast_trunc<vec<T>, N>(v);
    }
}

