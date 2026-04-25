#pragma once

#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "../../detail/utils.hpp"
#include "../optimizer.hpp"

namespace mirage::optim {
struct SGDOptions {
  float lr = 0.01f;
  float momentum = 0.0f;
  float lambda = 0.0f;

  bool nesterov = false;
  bool maximize = false;

  int num_proc = 1;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(lr, momentum, lambda, nesterov, maximize);
  }
};

template <typename TypeTuple>
struct SGDState : public OptimizerState {
  detail::ExtractedVector<TypeTuple> momentum{};
};

template <typename DedupedPack>
  requires detail::NonConstPack<DedupedPack>
class SGD : public Optimizer<DedupedPack> {
  public:
  explicit SGD(ParameterPack<DedupedPack> parameters, SGDOptions options = {})
    : Optimizer<DedupedPack>(parameters), options_(options) {
    detail::test_oom(this->parameters_.data, [&](auto& param) { return param.numel(); });

    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            using ParamType = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type;
            auto& mom = std::get<detail::ExtractType_t<ParamType>>(state_.momentum);
            for (auto& param_ref : param_vec) {
              auto& param = param_ref.get();
              using T = typename ParamType::DataType;
              mom.insert(mom.end(), param.numel(), T(0));
            }
          }(param_vecs),
          ...);
      },
      this->parameters_.data
    );
  }

  void step() override {
    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            using ParamType = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type;
            auto& mom_full = std::get<detail::ExtractType_t<ParamType>>(state_.momentum);

            int state_offset = 0;
            for (auto param_ref : param_vec) {
              auto& param = param_ref.get();
              using T = typename ParamType::DataType;

              auto& grad_full = param.grad();
              auto& data_full = param.data();

              constexpr int vec_size = eve::wide<T>::size();
              constexpr int unroll_factor = detail::UNROLL_FACTOR;

              int chunk_size = (param.numel() + options_.num_proc - 1) / options_.num_proc;

              detail::parallel(
                [&](int i) {
                  int start = i * chunk_size;
                  int end = std::min(start + chunk_size, param.numel());

                  int j = start;
                  for (; j + vec_size * unroll_factor <= end; j += vec_size * unroll_factor) {
                    detail::unroll<unroll_factor>([&]<int index>() {
                      constexpr int offset = index * vec_size;

                      eve::wide<T> grad(&grad_full[j + offset]);
                      eve::wide<T> mom(&mom_full[state_offset + j + offset]);
                      eve::wide<T> data(&data_full[j + offset]);

                      if (!options_.maximize) grad = -grad;

                      auto update = [&]() {
                        if (options_.momentum) {
                          mom = eve::fma(eve::wide<T>(options_.momentum), mom, grad);
                          eve::store(mom, &mom_full[state_offset + j + offset]);

                          grad = (options_.nesterov)
                                   ? eve::fma(eve::wide<T>(options_.momentum), mom, grad)
                                   : mom;
                        }
                        if (options_.lambda)
                          grad = eve::fma(eve::wide<T>(options_.lambda), data, grad);

                        return grad;
                      }();

                      data = eve::fma(eve::wide<T>(options_.lr), update, data);
                      eve::store(data, &data_full[j + offset]);
                    });
                  }

                  for (; j < end; ++j) {
                    T grad = options_.maximize ? grad_full[j] : -grad_full[j];
                    T mom = options_.momentum * mom_full[state_offset + j] + grad;
                    mom_full[state_offset + j] = mom;

                    T update = options_.nesterov ? (options_.momentum * mom + grad) : mom;
                    if (options_.lambda) update = update - options_.lambda * data_full[j];

                    data_full[j] += options_.lr * update;
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

    ar(options_, state_.step, state_.momentum);

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

    ar(name, options_, state_.step, state_.momentum);

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
  SGDOptions options_;
  SGDState<DedupedPack> state_;

  std::string optimizer_type() const override {
    return "SGD<" + detail::type_names(this->parameters_.data) + ">";
  }
};
}  // namespace mirage::optim
