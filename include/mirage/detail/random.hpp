#pragma once

#include <Random123/philox.h>

#include <cstddef>
#include <cstdint>
#include <eve/module/core.hpp>
#include <eve/module/math.hpp>
#include <eve/wide.hpp>
#include <vector>

namespace mirage::detail {
namespace random {
template <typename T>
struct philox_for {};

template <>
struct philox_for<float> {
  using type = r123::Philox4x32;
};
template <>
struct philox_for<double> {
  using type = r123::Philox4x64;
};

template <typename T>
using philox_for_t = typename philox_for<T>::type;

template <typename WideF, typename WideU>
WideF u01_open(WideU x) {
  using F = eve::element_type_t<WideF>;
  using U = eve::element_type_t<WideU>;
  constexpr int bits = sizeof(U) * 8;
  constexpr F norm = (bits == 32) ? F(0x1.0p-32) : F(0x1.0p-64);

  WideF xf = eve::convert(x, eve::as<F>{});
  return eve::fma(xf, norm, F(0.5) * norm);
}

template <typename WideF, typename WideU>
kumi::tuple<WideF, WideF> boxmuller_wide(WideU i1, WideU i2) {
  using F = eve::element_type_t<WideF>;
  constexpr F two_pi = F(6.28318530717958647692528676655900577);

  WideF u1 = u01_open<WideF>(i1);
  WideF u2 = u01_open<WideF>(i2);

  WideF r = eve::sqrt(F(-2) * eve::log(u1));
  WideF theta = two_pi * u2;

  auto [s, c] = eve::sincos(theta);
  return {r * c, r * s};
}

template <typename Real>
class normal_stream {
  public:
  using RNG = philox_for_t<Real>;
  using word = RNG::ctr_type::value_type;
  using wide_u = eve::wide<word>;
  using wide_f = eve::wide<Real>;
  static constexpr std::size_t W = wide_f::size();

  normal_stream(std::uint64_t seed, std::uint64_t stream_id) : seed_(seed), buf1_(W), buf2_(W) {
    key_ = {{}};
    ctr_ = {{}};
    key_[0] = stream_id;
  }

  wide_f next() {
    if (have_spare_) {
      have_spare_ = false;
      return spare_;
    }
    refill();
    wide_u i1(buf1_.data());
    wide_u i2(buf2_.data());
    auto [z0, z1] = boxmuller_wide<wide_f>(i1, i2);
    spare_ = z1;
    have_spare_ = true;
    return z0;
  }

  private:
  void refill() {
    std::size_t filled = 0;
    while (filled < W) {
      ctr_[0] = counter_++;
      ctr_[1] = seed_;
      typename RNG::ctr_type r = rng_(ctr_, key_);
      for (std::size_t k = 0; k + 1 < r.size() && filled < W; k += 2) {
        buf1_[filled] = r[k];
        buf2_[filled] = r[k + 1];
        ++filled;
      }
    }
  }

  RNG rng_;
  typename RNG::key_type key_;
  typename RNG::ctr_type ctr_;
  std::uint64_t seed_;
  std::uint64_t counter_ = 0;

  std::vector<word> buf1_, buf2_;
  wide_f spare_{};
  bool have_spare_ = false;
};
}  // namespace random

template <typename Real>
std::vector<Real> random_gaussian(std::size_t n, std::uint64_t seed, std::uint64_t stream_id) {
  random::normal_stream<Real> ns(seed, stream_id);
  constexpr std::size_t W = random::normal_stream<Real>::W;

  const std::size_t rounded = ((n + W - 1) / W) * W;
  std::vector<Real> out(rounded);

  for (std::size_t pos = 0; pos < rounded; pos += W) eve::store(ns.next(), out.data() + pos);

  out.resize(n);
  return out;
}

template <typename Real>
std::vector<Real> random_uniform(
  std::size_t n_points, std::size_t D, std::uint64_t seed, std::uint64_t stream_id
) {
  if (D == 0) return {};

  using wide_f = typename random::normal_stream<Real>::wide_f;
  constexpr std::size_t W = random::normal_stream<Real>::W;

  random::normal_stream<Real> ns(seed, stream_id);

  const std::size_t rounded = ((n_points + W - 1) / W) * W;
  std::vector<Real> out(rounded * D);

  std::vector<wide_f> coord(D);

  std::size_t base = 0;
  for (std::size_t done = 0; done < rounded; done += W) {
    for (std::size_t d = 0; d < D; ++d) coord[d] = ns.next();

    wide_f sumsq = coord[0] * coord[0];
    for (std::size_t d = 1; d < D; ++d) sumsq = eve::fma(coord[d], coord[d], sumsq);

    wide_f inv_norm = eve::rsqrt(sumsq);

    for (std::size_t d = 0; d < D; ++d) coord[d] = coord[d] * inv_norm;

    for (std::size_t l = 0; l < W; ++l)
      for (std::size_t d = 0; d < D; ++d) out[base + (l * D) + d] = coord[d].get(l);

    base += W * D;
  }

  out.resize(n_points * D);
  return out;
}
}  // namespace mirage::detail
