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
  explicit Estimator(ParameterPack<DedupedPack> parameters, int num_evals)
    : parameters_(parameters), num_evals_(num_evals) {
    int64_t bytes = 0;
    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            using ParamType =
              typename std::remove_cvref_t<decltype(param_vec)>::value_type::type::DataType;

            std::ranges::for_each(param_vec, [&](auto& param_ref) {
              auto param = param_ref.get();
              bytes += sizeof(ParamType) * 2 * param.numel();
            });
          }(param_vecs),
          ...);
      },
      parameters
    );

    bytes /= (1024 * 1024);
    const char* max = std::getenv("MIRAGE_ESTIMIZER_MEMMAX");
    int max_bytes = (max == nullptr) ? 256 : std::stoi(max);
    if (bytes > max_bytes)
      throw std::runtime_error(
        "Out of memory, tried to allocate " + std::to_string(bytes) + " MiB out of a max of " +
        std::to_string(max_bytes) +
        " MiB. You can increase this using the \"MIRAGE_ESTIMIZER_MEMMAX\" environment "
        "variable."
      );

    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            using ParamType = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type;
            auto& pos = std::get<detail::ExtractType_t<ParamType>>(state_.positives);
            auto& neg = std::get<detail::ExtractType_t<ParamType>>(state_.negatives);

            for (auto& param_ref : param_vec) {
              auto& param = param_ref.get();
              using T = typename ParamType::DataType;
              pos.insert(pos.end(), param.numel(), T(0));
              neg.insert(neg.end(), param.numel(), T(0));
            }
          }(param_vecs),
          ...);
      },
      this->parameters_.data
    );
  }

  virtual bool needs_eval() { return state_.evals_left == 0; };

  virtual void perturb() = 0;
  virtual void observe() = 0;

  ~Estimator() = default;

  protected:
  ParameterPack<DedupedPack> parameters_;
  EstimatorState<DedupedPack> state_;
  int num_evals_;
};
}  // namespace mirage::estim
