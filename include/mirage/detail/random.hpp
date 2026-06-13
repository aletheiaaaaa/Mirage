#pragma once

#include <Random123/philox.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <eve/module/core.hpp>
#include <eve/module/math.hpp>
#include <eve/wide.hpp>
#include <numeric>
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
}  // namespace random

template <typename Real>
class Generator {
  public:
  using RNG = random::philox_for_t<Real>;
  using word = RNG::ctr_type::value_type;
  using four_u = eve::wide<word, eve::fixed<4>>;
  using four_f = eve::wide<Real, eve::fixed<4>>;
  static constexpr size_t W = four_f::size();

  Generator(uint64_t seed, uint64_t stream_id) : seed_(seed) {
    key_ = {{0, stream_id}};
    ctr_ = {{0, seed}};
  }

  kumi::tuple<four_f, four_f> generate() {
    auto reg = [&]() { return rng_({{counter_++, seed_}}, key_); };

    auto r1 = reg();
    auto r2 = reg();

    four_u i1(&r1.v[0]);
    four_u i2(&r2.v[0]);

    return random::boxmuller_wide<four_f>(i1, i2);
  }

  private:
  RNG rng_;
  typename RNG::key_type key_;
  typename RNG::ctr_type ctr_;
  uint64_t counter_ = 0;
  uint64_t seed_;
};

template <typename Real>
void normalize(std::vector<Real>& vec, Real eps) {
  Real norm = std::sqrt(std::inner_product(vec.begin(), vec.end(), vec.begin(), Real(0.0)));
  Real inv = 1 / (norm + Real(1e-8));

  for (auto& x : vec) x *= inv;
}
}  // namespace mirage::detail
