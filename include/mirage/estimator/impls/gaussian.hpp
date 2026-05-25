#pragma once

#include "../../parameter.hpp"
#include "../estimator.hpp"

namespace mirage::estim {

template <typename DedupedPack>
  requires detail::NonConstPack<DedupedPack>
class Gaussian : public Estimator<DedupedPack> {
  public:
  explicit Gaussian(ParameterPack<DedupedPack> parameters, int num_evals = 8, float norm = 1e-3f)
    : Estimator<DedupedPack>(parameters, num_evals) {
    state_.norm = norm;
    this->init(state_);
    state_.needs_eval = true;
  }

  bool needs_eval() override { return state_.needs_eval; }

  void perturb() override {
    // TODO: compute
  }

  void observe(float loss) override {
    // TODO: restore

    if (state_.is_pos) {
      state_.pos_mean += (loss - state_.pos_mean) / ++state_.pairs_so_far;
      state_.is_pos = false;
    } else {
      state_.neg_mean += (loss - state_.neg_mean) / state_.pairs_so_far;
      state_.is_pos = true;

      // TODO: compute simultaneous grads

      if (state_.pairs_so_far == this->num_evals_) {
        state_.pairs_so_far = 0;
        state_.pos_mean = 0.0f;
        state_.neg_mean = 0.0f;

        state_.needs_eval = false;
      }
    }
  }

  void finalize() override {
    this->write(state_.grad);
    std::fill(state_.grad.begin(), state_.grad.end(), 0.0f);

    // TODO: reset perturb vect

    state_.is_pos = true;
    state_.pairs_so_far = 0;
    state_.pos_mean = 0.0f;
    state_.neg_mean = 0.0f;
    state_.needs_eval = true;
  }

  private:
  EstimatorState<DedupedPack> state_;
  detail::ExtractedVector<DedupedPack> perturbation_;
};
}  // namespace mirage::estim
