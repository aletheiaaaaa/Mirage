#pragma once

#include "../../parameter.hpp"
#include "../estimator.hpp"

namespace mirage::estim {

template <typename DedupedPack>
  requires detail::NonConstPack<DedupedPack>
class Coordinate : public Estimator<DedupedPack> {
  public:
  explicit Coordinate(ParameterPack<DedupedPack> parameters, int num_evals = 1, double norm = 1e-3f)
    : Estimator<DedupedPack>(parameters, num_evals) {
    state_.norm = norm;
    state_.needs_eval = true;
  }

  void perturb() override { apply(current_, state_.is_pos ? state_.norm : -state_.norm); }

  void observe(float loss) override {
    apply(current_, state_.is_pos ? -state_.norm : state_.norm);

    if (state_.is_pos) {
      state_.pos_mean += (loss - state_.pos_mean) / ++state_.pairs_so_far;
      state_.is_pos = false;
    } else {
      state_.neg_mean += (loss - state_.neg_mean) / state_.pairs_so_far;
      state_.is_pos = true;

      write_grad(current_, (state_.pos_mean - state_.neg_mean) / (2.0f * state_.norm));

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

  void apply(int coord, double delta) {
    int remaining = coord;
    bool found = false;

    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            using T = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type::DataType;

            if (found) return;
            for (auto& param_ref : param_vec) {
              if (found) return;
              auto& param = param_ref.get();
              int n = param.numel();
              if (remaining < n) {
                param.data()[remaining] += T(delta);
                found = true;
                return;
              }
              remaining -= n;
            }
          }(param_vecs),
          ...);
      },
      this->parameters_.data
    );
  }

  void write_grad(int coord, double grad) {
    int remaining = coord;
    bool found = false;

    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            using T = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type::DataType;

            if (found) return;
            for (auto& param_ref : param_vec) {
              if (found) return;

              auto& param = param_ref.get();
              int n = param.numel();
              T smoothing = param.smoothing();

              if (remaining < n) {
                param.grad()[remaining] = smoothing * T(grad);
                found = true;
                return;
              }
              remaining -= n;
            }
          }(param_vecs),
          ...);
      },
      this->parameters_.data
    );
  }
};
}  // namespace mirage::estim
