#pragma once

#include "../../parameter.hpp"
#include "../estimator.hpp"

namespace mirage::estim {

template <typename DedupedPack>
  requires detail::NonConstPack<DedupedPack>
class Gaussian : public Estimator<DedupedPack> {
  public:
  using Estimator<DedupedPack>::Estimator;

  void perturb() override {
    this->apply(
      this->current_,
      this->state_.is_pos ? this->state_.norm : -this->state_.norm
    );
  }

  void observe(float loss) override {
    this->apply(
      this->current_,
      this->state_.is_pos ? -this->state_.norm : this->state_.norm
    );

    if (this->state_.is_pos) {
      this->state_.pos_mean += (loss - this->state_.pos_mean) / ++this->state_.pairs_so_far;
      this->state_.is_pos = false;
    } else {
      this->state_.neg_mean += (loss - this->state_.neg_mean) / this->state_.pairs_so_far;
      this->state_.is_pos = true;

      this->state_.grad[this->current_] =
        (this->state_.pos_mean - this->state_.neg_mean) / (2.0f * this->state_.norm);

      if (this->state_.pairs_so_far == this->num_evals_) {
        ++this->current_;
        this->state_.pairs_so_far = 0;
        this->state_.pos_mean = 0.0f;
        this->state_.neg_mean = 0.0f;

        if (this->current_ == this->state_.total)
          this->state_.needs_eval = false;
      }
    }
  }
};
}  // namespace mirage::estim
