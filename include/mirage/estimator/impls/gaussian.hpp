#pragma once

#include <tuple>

#include "../../parameter.hpp"
#include "../estimator.hpp"

namespace mirage::estim {

template <typename DedupedPack>
  requires detail::NonConstPack<DedupedPack>
class Gaussian : public Estimator<DedupedPack> {
  public:
  explicit Gaussian(ParameterPack<DedupedPack> parameters, int num_evals = 8, double norm = 1e-3f)
    : Estimator<DedupedPack>(parameters, num_evals) {
    state_.norm = norm;
    state_.needs_eval = true;
  }

  void perturb() override {}

  void observe(double loss) override {
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
    state_.is_pos = true;
    state_.pairs_so_far = 0;
    state_.pos_mean = 0.0f;
    state_.neg_mean = 0.0f;
    state_.needs_eval = true;
  }

  private:
  EstimatorState<DedupedPack> state_;
  detail::ExtractedVector<DedupedPack> perturbation_;

  void apply(int seed, double delta, bool reset) {
    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            using T = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type::DataType;

            for (auto& param_ref : param_vec) {
              auto& param = param_ref.get();
              int n = param.numel();

              // TODO: write the loop
            }
          }(param_vecs),
          ...);
      },
      this->parameters_.data
    );
  }
};
}  // namespace mirage::estim
