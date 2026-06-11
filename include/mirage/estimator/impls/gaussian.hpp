#pragma once

#include <tuple>

#include "../../detail/thread.hpp"
#include "../../parameter.hpp"
#include "../estimator.hpp"

namespace mirage::estim {

template <typename DedupedPack>
  requires detail::NonConstPack<DedupedPack>
class Gaussian : public Estimator<DedupedPack> {
  public:
  explicit Gaussian(
    ParameterPack<DedupedPack> parameters, int num_evals = 8, double norm = 1e-3f, int num_proc = 4
  )
    : Estimator<DedupedPack>(parameters, num_evals), num_proc_(num_proc), pool_(num_proc) {
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
  detail::ThreadPool pool_;
  detail::StateTuple<DedupedPack> generators_;
  int num_proc_;

  void apply(uint64_t seed, double delta, bool reset) {
    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            using T = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type::DataType;

            for (auto& param_ref : param_vec) {
              auto& param = param_ref.get();
              auto& grad_full = param.grad();
              int n = param.numel();

              constexpr int vec_size = eve::wide<T>::size();
              constexpr int unroll_factor = detail::UNROLL_FACTOR / 2;

              int chunk_size = (param.numel() + num_proc_ - 1) / num_proc_;

              pool_.run(
                [&](int i) {
                  int start = i * chunk_size;
                  int end = std::min(start + chunk_size, param.numel());

                  for (int j = start; j < end; j += vec_size * unroll_factor) {
                    detail::unroll<unroll_factor>([&]<int index>() {
                      constexpr int offset = 2 * index * vec_size;

                      eve::wide<T> grad1(&grad_full[j + offset]);
                      eve::wide<T> grad2(&grad_full[j + offset + vec_size]);

                      auto [rand1, rand2] = detail::random_gaussian<T>(seed, i);
                    });
                  }
                },
                num_proc_
              );
            }
          }(param_vecs),
          ...);
      },
      this->parameters_.data
    );
  }
};
}  // namespace mirage::estim
