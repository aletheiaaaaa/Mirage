#pragma once

#include "../parameter.hpp"

namespace mirage::estim {

template <typename TypeTuple>
struct EstimatorState {
  detail::ExtractedVector<TypeTuple> positives{};
  detail::ExtractedVector<TypeTuple> negatives{};

  uint32_t evals_left = 0;
};

template <typename DedupedPack>
  requires detail::NonConstPack<DedupedPack>
class Estimator {
  public:
  explicit Estimator(ParameterPack<DedupedPack> parameters) : parameters_(parameters) {
    // TODO: actually init the state lol
  }

  virtual bool needs_eval() { return state_.evals_left == 0; };

  virtual void perturb() = 0;
  virtual void observe() = 0;

  ~Estimator() = default;

  protected:
  ParameterPack<DedupedPack> parameters_;
  EstimatorState<DedupedPack> state_;
};
}  // namespace mirage::estim
