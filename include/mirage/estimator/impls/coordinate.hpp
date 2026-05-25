#pragma once

#include "../../parameter.hpp"
#include "../estimator.hpp"

namespace mirage::estim {

template <typename DedupedPack>
  requires detail::NonConstPack<DedupedPack>
class Coordinate : public Estimator<DedupedPack> {
  public:
  explicit Coordinate(ParameterPack<DedupedPack> parameters, int num_evals, float epsilon = 1e-3f)
    : Estimator<DedupedPack>(parameters, num_evals) {
    state_.norm = epsilon;
    this->init(state_);
    state_.needs_eval = true;
  }

  bool needs_eval() override { return state_.needs_eval; }

  void perturb() override { this->apply(current_, state_.is_pos ? state_.norm : -state_.norm); }

  void observe(float loss) override {
    this->apply(current_, state_.is_pos ? -state_.norm : state_.norm);

    if (state_.is_pos) {
      state_.pos_mean += (loss - state_.pos_mean) / ++state_.pairs_so_far;
      state_.is_pos = false;
    } else {
      state_.neg_mean += (loss - state_.neg_mean) / state_.pairs_so_far;
      state_.is_pos = true;

      state_.grad[current_] = (state_.pos_mean - state_.neg_mean) / (2.0f * state_.norm);

      if (state_.pairs_so_far == this->num_evals_) {
        ++current_;
        state_.pairs_so_far = 0;
        state_.pos_mean = 0.0f;
        state_.neg_mean = 0.0f;

        if (current_ == state_.total) state_.needs_eval = false;
      }
    }
  }

  void finalize() override {
    this->write(state_.grad);
    std::fill(state_.grad.begin(), state_.grad.end(), 0.0f);

    current_ = 0;
    state_.is_pos = true;
    state_.pairs_so_far = 0;
    state_.pos_mean = 0.0f;
    state_.neg_mean = 0.0f;
    state_.needs_eval = true;
  }

  private:
  EstimatorState<DedupedPack> state_;
  int current_ = 0;
};
}  // namespace mirage::estim
