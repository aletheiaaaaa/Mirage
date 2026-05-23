#pragma once

#include "../../parameter.hpp"
#include "../estimator.hpp"

namespace mirage::estim {

template <typename DedupedPack>
  requires detail::NonConstPack<DedupedPack>
class Coordinate : public Estimator<DedupedPack> {
  public:
  explicit Coordinate(ParameterPack<DedupedPack> parameters, int num_evals)
    : Estimator<DedupedPack>(parameters, num_evals) {}

  void perturb() override {
    // TODO: actually do this
  }
};
}  // namespace mirage::estim
