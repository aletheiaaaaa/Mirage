#pragma once

#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "../../detail/thread.hpp"
#include "../../detail/utils.hpp"
#include "../optimizer.hpp"

namespace mirage::optim {
struct SarahOptions {
  float lr = 0.01f;
  float lambda = 0.0f;
  int recompute_every = 64;

  bool maximize = false;

  int num_proc = 1;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(lr, lambda, recompute_every, maximize);
  }
};

template <typename TypeTuple>
struct SarahState : public OptimizerState {
  detail::ExtractedVector<TypeTuple> prev_grad{};
  detail::ExtractedVector<TypeTuple> prev_update{};
};

template <typename DedupedPack>
  requires detail::NonConstPack<DedupedPack>
class Sarah : public Optimizer<DedupedPack> {
  public:
  explicit Sarah(ParameterPack<DedupedPack> parameters, SarahOptions options = {})
    : Optimizer<DedupedPack>(parameters), options_(options), pool_(options.num_proc) {
    if ((options_.recompute_every != -1) && options_.recompute_every == 0)
      throw std::invalid_argument(
        "Recompute every must be greater than 0 when recompute is enabled"
      );

    detail::test_oom(this->parameters_.data, [&](auto& param) { return 2 * param.numel(); });

    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            using ParamType = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type;
            auto& prev_grad = std::get<detail::ExtractType_t<ParamType>>(state_.prev_grad);
            auto& prev_update = std::get<detail::ExtractType_t<ParamType>>(state_.prev_update);
            for (auto& param_ref : param_vec) {
              auto& param = param_ref.get();
              using T = typename ParamType::DataType;
              prev_grad.insert(prev_grad.end(), param.numel(), T(0));
              prev_update.insert(prev_update.end(), param.numel(), T(0));
            }
          }(param_vecs),
          ...);
      },
      this->parameters_.data
    );
  }

  bool recompute() const override {
    return (options_.recompute_every != -1) && state_.step % options_.recompute_every == 0;
  }

  void step() override {
    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            using ParamType = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type;
            auto& prev_grad_full = std::get<detail::ExtractType_t<ParamType>>(state_.prev_grad);
            auto& prev_update_full = std::get<detail::ExtractType_t<ParamType>>(state_.prev_update);

            int state_offset = 0;
            for (auto param_ref : param_vec) {
              auto& param = param_ref.get();
              using T = typename ParamType::DataType;

              auto& grad_full = param.grad();
              auto& data_full = param.data();

              constexpr int vec_size = eve::wide<T>::size();
              constexpr int unroll_factor = detail::UNROLL_FACTOR;

              int chunk_size = (param.numel() + options_.num_proc - 1) / options_.num_proc;

              pool_.run(
                [&](int i) {
                  int start = i * chunk_size;
                  int end = std::min(start + chunk_size, param.numel());

                  int j = start;
                  for (; j + vec_size * unroll_factor <= end; j += vec_size * unroll_factor) {
                    detail::unroll<unroll_factor>([&]<int index>() {
                      constexpr int offset = index * vec_size;

                      eve::wide<T> grad(&grad_full[j + offset]);
                      eve::wide<T> data(&data_full[j + offset]);
                      if (!options_.maximize) grad = -grad;

                      auto update = [&]() {
                        if (
                          (options_.recompute_every != -1) &&
                          state_.step % options_.recompute_every == 0
                        )
                          return grad;

                        eve::wide<T> prev_grad(&prev_grad_full[state_offset + j + offset]);
                        eve::wide<T> prev_update(&prev_update_full[state_offset + j + offset]);

                        grad = eve::add(eve::sub(grad, prev_grad), prev_update);
                        if (options_.lambda)
                          grad = eve::fnma(eve::wide<T>(options_.lambda), data, grad);

                        return grad;
                      }();

                      data = eve::fma(eve::wide<T>(options_.lr), update, data);

                      eve::store(data, &data_full[j + offset]);
                      eve::store(grad, &prev_grad_full[state_offset + j + offset]);
                    });
                  }

                  for (; j < end; ++j) {
                    T grad = options_.maximize ? grad_full[j] : -grad_full[j];
                    T update = ((options_.recompute_every != -1) &&
                                state_.step % options_.recompute_every == 0)
                                 ? grad
                                 : grad - prev_grad_full[state_offset + j] +
                                     prev_update_full[state_offset + j];

                    if (options_.lambda) update = update - options_.lambda * data_full[j];

                    data_full[j] += options_.lr * update;
                    prev_grad_full[state_offset + j] = grad;
                    prev_update_full[state_offset + j] = update;
                  }
                },
                options_.num_proc
              );

              state_offset += param.numel();
            }
          }(param_vecs),
          ...);
      },
      this->parameters_.data
    );
    state_.step++;
  }

  void load_from_bin(const std::string& path_str) override {
    std::filesystem::path path(path_str);
    path.replace_extension(".bin");

    if (!std::filesystem::exists(path)) throw std::runtime_error("File not found: " + path_str);

    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open file: " + path_str);

    cereal::BinaryInputArchive ar(in);
    std::string name;
    ar(name);
    if (name != optimizer_type()) this->handle_type_error(name);

    ar(options_, state_.step, state_.prev_grad, state_.prev_update);

    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            for (auto& param_ref : param_vec) {
              ar(param_ref.get().data());
              ar(param_ref.get().grad());
            }
          }(param_vecs),
          ...);
      },
      this->parameters_.data
    );
  }

  void save_to_bin(const std::string& path_str) const override {
    std::filesystem::path path(path_str);
    path.replace_extension(".bin");

    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open file: " + path_str);

    cereal::BinaryOutputArchive ar(out);
    std::string name(optimizer_type());
    ar(name, options_, state_.step, state_.prev_grad, state_.prev_update);

    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            for (auto& param_ref : param_vec) {
              ar(param_ref.get().data());
              ar(param_ref.get().grad());
            }
          }(param_vecs),
          ...);
      },
      this->parameters_.data
    );
  }

  private:
  SarahOptions options_;
  SarahState<DedupedPack> state_;
  detail::ThreadPool pool_;

  std::string optimizer_type() const override {
    return "Sarah<" + detail::type_names(this->parameters_.data) + ">";
  }
};
}  // namespace mirage::optim
