#pragma once

#include <cstdint>

namespace agon::simd {
  enum class Arch : uint8_t {
    GENERIC,
    SSE4_1,
    AVX2,
    AVX512,
  };

  inline constexpr Arch detect_arch() {
#if defined(__AVX512F__)
    return Arch::AVX512;
#elif defined(__AVX2__)
    return Arch::AVX2;
#elif defined(__SSE4_1__)
    return Arch::SSE4_1;
#else
    return Arch::GENERIC;
#endif
  }

  constexpr Arch CURRENT_ARCH = detect_arch();
}