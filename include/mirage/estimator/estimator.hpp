#pragma once

#include "../parameter.hpp"

namespace mirage::estim {

template <typename TypeTuple>
struct EstimatorState {
  bool needs_eval = false;
  int total = 0;
  float norm = 0.0f;
  float pos_mean = 0.0f;
  float neg_mean = 0.0f;

  std::vector<float> grad;
  bool is_pos = true;
  int pairs_so_far = 0;
};

template <typename DedupedPack>
  requires detail::NonConstPack<DedupedPack>
class Estimator {
  public:
  explicit Estimator(ParameterPack<DedupedPack> parameters, int num_evals)
    : parameters_(parameters), num_evals_(num_evals) {}

  bool needs_eval() { return state_.needs_eval; }
  virtual void perturb() = 0;
  virtual void observe(float loss) = 0;

  void finalize() {
    int i = 0;
    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            for (auto& param_ref : param_vec) {
              auto& param = param_ref.get();
              using T = typename std::remove_cvref_t<decltype(param)>::DataType;
              for (int j = 0; j < param.numel(); ++j, ++i)
                param.grad()[j] = static_cast<T>(state_.grad[i]) * param.attenuation();
            }
          }(param_vecs),
          ...);
      },
      parameters_.data
    );

    std::fill(state_.grad.begin(), state_.grad.end(), 0.0f);
    current_ = 0;
    state_.is_pos = true;
    state_.pairs_so_far = 0;
    state_.pos_mean = 0.0f;
    state_.neg_mean = 0.0f;
    state_.needs_eval = true;
  }

  ~Estimator() = default;

  protected:
  ParameterPack<DedupedPack> parameters_;
  int num_evals_;
  EstimatorState<DedupedPack> state_;
  int current_ = 0;

  void init(EstimatorState<DedupedPack>& s) {
    int64_t bytes = 0;
    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            using ParamType =
              typename std::remove_cvref_t<decltype(param_vec)>::value_type::type::DataType;

            for (auto& param_ref : param_vec) {
              auto& param = param_ref.get();
              bytes += sizeof(ParamType) * 2 * param.numel();
              s.total += param.numel();
            }
          }(param_vecs),
          ...);
      },
      parameters_.data
    );

    s.grad.resize(s.total, 0.0f);

    bytes /= (1024 * 1024);
    const char* max = std::getenv("MIRAGE_ESTIMATOR_MEMMAX");
    int max_bytes = (max == nullptr) ? 256 : std::stoi(max);
    if (bytes > max_bytes)
      throw std::runtime_error(
        "Out of memory, tried to allocate " + std::to_string(bytes) + " MiB out of a max of " +
        std::to_string(max_bytes) +
        " MiB. You can increase this using the \"MIRAGE_ESTIMATOR_MEMMAX\" environment "
        "variable."
      );

    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            using ParamType = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type;
            auto& pos = std::get<detail::ExtractType_t<ParamType>>(s.positives);
            auto& neg = std::get<detail::ExtractType_t<ParamType>>(s.negatives);

            for (auto& param_ref : param_vec) {
              auto& param = param_ref.get();
              using T = typename ParamType::DataType;
              pos.insert(pos.end(), param.numel(), T(0));
              neg.insert(neg.end(), param.numel(), T(0));
            }
          }(param_vecs),
          ...);
      },
      parameters_.data
    );
  }

  // TODO: support doubles;
  void apply(int coord, float delta) {
    int remaining = coord;
    bool found = false;

    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            if (found) return;
            for (auto& param_ref : param_vec) {
              if (found) return;
              auto& param = param_ref.get();
              int n = param.numel();
              if (remaining < n) {
                param.data()[remaining] += delta;
                found = true;
                return;
              }
              remaining -= n;
            }
          }(param_vecs),
          ...);
      },
      parameters_.data
    );
  }

  void apply(const std::vector<float>& deltas) {
    int i = 0;
    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            for (auto& param_ref : param_vec) {
              auto& param = param_ref.get();
              for (int j = 0; j < param.numel(); ++j, ++i) param.data()[j] += deltas[i];
            }
          }(param_vecs),
          ...);
      },
      parameters_.data
    );
  }
};
}  // namespace mirage::estim
