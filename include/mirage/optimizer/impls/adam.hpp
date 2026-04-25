#pragma once

#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "../../detail/utils.hpp"
#include "../optimizer.hpp"

namespace mirage::optim {
struct AdamOptions {
  float lr = 1e-4f;
  float beta1 = 0.9f;
  float beta2 = 0.4f;
  float epsilon = 1e-8f;
  float lambda = 0.0f;

  bool maximize = false;
  bool use_adazo = false;

  int num_proc = 1;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(lr, beta1, beta2, epsilon, lambda, maximize, use_adazo);
  }
};

template <typename TypeTuple>
struct AdamState : public OptimizerState {
  detail::ExtractedVector<TypeTuple> momentum{};
  detail::ExtractedVector<TypeTuple> velocity{};
};

template <typename DedupedPack>
  requires detail::NonConstPack<DedupedPack>
class Adam : public Optimizer<DedupedPack> {
  public:
  explicit Adam(ParameterPack<DedupedPack> parameters, AdamOptions options = {})
    : Optimizer<DedupedPack>(parameters), options_(options) {
    detail::test_oom(this->parameters_.data, [&](auto& param) { return 2 * param.numel(); });

    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            using ParamType = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type;
            auto& mom = std::get<detail::ExtractType_t<ParamType>>(state_.momentum);
            auto& vel = std::get<detail::ExtractType_t<ParamType>>(state_.velocity);
            for (auto& param_ref : param_vec) {
              auto& param = param_ref.get();
              using T = typename ParamType::DataType;
              mom.insert(mom.end(), param.numel(), T(0));
              vel.insert(vel.end(), param.numel(), T(0));
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
            auto& vel_full = std::get<detail::ExtractType_t<ParamType>>(state_.velocity);

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
                      eve::wide<T> vel(&vel_full[state_offset + j + offset]);

                      if (!options_.maximize) grad = -grad;

                      eve::wide<T> beta1(options_.beta1);
                      mom = eve::fma(beta1, mom, grad);
                      mom = eve::fnma(beta1, grad, mom);

                      eve::wide<T> beta2(options_.beta2);
                      auto grad_squared =
                        (options_.use_adazo) ? eve::mul(mom, mom) : eve::mul(grad, grad);
                      vel = eve::fma(beta2, vel, grad_squared);
                      vel = eve::fnma(beta2, grad_squared, vel);

                      eve::store(mom, &mom_full[state_offset + j + offset]);
                      eve::store(vel, &vel_full[state_offset + j + offset]);

                      auto update =
                        eve::div(mom, eve::add(eve::sqrt(vel), eve::wide<T>(options_.epsilon)));
                      eve::wide<T> data(&data_full[j + offset]);

                      if (options_.lambda)
                        update = eve::fnma(eve::wide<T>(options_.lambda), data, update);
                      data = eve::fma(eve::wide<T>(options_.lr), update, data);
                      eve::store(data, &data_full[j + offset]);
                    });
                  }

                  for (; j < end; ++j) {
                    T grad = options_.maximize ? grad_full[j] : -grad_full[j];

                    T mom =
                      options_.beta1 * mom_full[state_offset + j] + (1 - options_.beta1) * grad;
                    T vel = options_.beta2 * vel_full[state_offset + j] +
                            (1 - options_.beta2) * grad * grad;

                    mom_full[state_offset + j] = mom;
                    vel_full[state_offset + j] = vel;

                    T update = mom / (std::sqrt(vel) + options_.epsilon);
                    if (options_.lambda) update = -options_.lambda * data_full[j] + update;

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

    ar(options_, state_.step, state_.momentum, state_.velocity);

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

    ar(name, options_, state_.step, state_.momentum, state_.velocity);
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
  AdamOptions options_;
  AdamState<DedupedPack> state_;

  std::string optimizer_type() const override {
    return "Adam<" + detail::type_names(this->parameters_.data) + ">";
  }
};
}  // namespace mirage::optim
