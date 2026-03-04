#pragma once

#include "../optimizer.h"
#include "../../detail/simd/ops.h"
#include "../../detail/simd/utils.h"

#include <cstring>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <stdexcept>

namespace agon::optim {
  struct LionParams {
    float lr = 1e-5f;
    float beta1 = 0.9f;
    float beta2 = 0.9f;
    float epsilon = 1e-8;
    float lambda = 0.0f;

    bool maximize = false;
  };

  template<typename DedupedTuple>
  struct LionState : OptimizerState {
    ExtractedVector<DedupedTuple> momentum{};
  };

  template<typename... Ts>
  class Lion : public Optimizer<Ts...> {
    public:
      explicit Lion(ParameterPack<Ts...> parameters, LionParams options = {})
        : Optimizer<Ts...>(parameters), options_(options) {
          std::apply([&](auto&... param_vecs) {
            ([&](auto& param_vec) {
              using ParamType = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type;
              auto& mom = std::get<ExtractType_t<ParamType>>(this->state_.momentum);
              for (auto& param_ref : param_vec) {
                auto& param = param_ref.get();
                using T = typename ParamType::DataType;
                mom.insert(mom.end(), param.numel(), T(0));
              }
            }(param_vecs), ...);
          }, this->parameters_.data);
        }

      void step() override {
        std::apply([&](auto&... param_vecs) {
          ([&](auto& param_vec) {
            using ParamType = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type;
            auto& mom_full = std::get<ExtractType_t<ParamType>>(state_.momentum);

            size_t state_offset = 0;
            for (auto param_ref : param_vec) {
              auto& param = param_ref.get();
              using T = typename ParamType::DataType;

              auto& grad_full = param.grad();
              auto& data_full = param.data();

              constexpr size_t vec_size = simd::vec<T>::size;
              constexpr size_t unroll_factor = simd::UNROLL_FACTOR;

              size_t i = 0;
              for (; i + vec_size * unroll_factor <= grad_full.size(); i += vec_size * unroll_factor) {
                simd::unroll<unroll_factor>([&]<size_t index>(){
                  constexpr size_t offset = index * vec_size;

                  auto grad = simd::load<T>(&grad_full[i + offset]);
                  auto mom = simd::load<T>(&mom_full[state_offset + i + offset]);

                  if (options_.maximize) grad = simd::neg(grad);

                  auto beta1 = simd::set1<T>(options_.beta1);
                  auto update = simd::fmadd(beta1, mom, grad);
                  update = simd::fnmadd(beta1, grad, mom);

                  auto data = simd::load<T>(&data_full[i + offset]);

                  if (options_.lambda) update = simd::fnmadd(simd::set1<T>(options_.lambda), data, update);
                  data = simd::fmadd(simd::set1<T>(options_.lr), simd::sign(update), data);
                  simd::store(&data_full[i + offset], data);

                  auto beta2 = simd::set1<T>(options_.beta2);
                  mom = simd::fmadd(beta2, mom, grad);
                  mom = simd::fnmadd(beta2, grad, mom);
                  simd::store(&mom_full[state_offset + i + offset], mom);
                });
              }

              for (; i < grad_full.size(); ++i) {
                T grad = options_.maximize ? -grad_full[i] : grad_full[i];
                T mom = options_.beta1 * mom_full[state_offset + i] + (1 - options_.beta1) * grad;

                T update = std::copysign(options_.lr, mom);
                if (options_.lambda) update = -options_.lambda * data_full[i] + update;

                data_full[i] += update;
                mom_full[state_offset + i] = options_.beta2 * mom_full[state_offset + i] + (1 - options_.beta2) * grad;
              }

              state_offset += param.numel();
            }
          }(param_vecs), ...);
        }, this->parameters_.data);
        this->state_.step++;
      }

      void load_from_bin(const std::string& path_str) override {
        std::filesystem::path path(path_str + ".bin");
        if (!std::filesystem::exists(path)) throw std::runtime_error("File not found: " + path_str);

        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("Failed to open file: " + path_str);

        std::string name;
        std::getline(in, name, '\0');
        if (name != optimizer_name()) throw std::runtime_error("Optimizer type mismatch: expected " + std::string(optimizer_name()));

        in.read(reinterpret_cast<char*>(&options_), sizeof(options_));
        in.read(reinterpret_cast<char*>(&state_.step), sizeof(state_.step));

        std::apply([&](auto&... param_vecs) {
          ([&](auto& param_vec) {
            using ParamType = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type;
            auto& mom = std::get<ExtractType_t<ParamType>>(state_.momentum);

            size_t state_offset = 0;
            for (auto& param_ref : param_vec) {
              auto& param = param_ref.get();
              using T = typename ParamType::DataType;
              in.read(reinterpret_cast<char*>(param.data().data()), param.numel() * sizeof(T));
              in.read(reinterpret_cast<char*>(mom.data() + state_offset), param.numel() * sizeof(T));
              state_offset += param.numel();
            }
          }(param_vecs), ...);
        }, this->parameters_.data);
      }

      void save_to_bin(const std::string& path_str) const override {
        std::filesystem::path path(path_str + ".bin");
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("Failed to open file: " + path_str);

        out.write(optimizer_name(), std::strlen(optimizer_name()) + 1);
        out.write(reinterpret_cast<const char*>(&options_), sizeof(options_));
        out.write(reinterpret_cast<const char*>(&state_.step), sizeof(state_.step));

        std::apply([&](auto&... param_vecs) {
          ([&](auto& param_vec) {
            using ParamType = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type;
            auto& mom = std::get<ExtractType_t<ParamType>>(state_.momentum);

            size_t state_offset = 0;
            for (auto& param_ref : param_vec) {
              auto& param = param_ref.get();
              using T = typename ParamType::DataType;
              out.write(reinterpret_cast<const char*>(param.data().data()), param.numel() * sizeof(T));
              out.write(reinterpret_cast<const char*>(mom.data() + state_offset), param.numel() * sizeof(T));
              state_offset += param.numel();
            }
          }(param_vecs), ...);
        }, this->parameters_.data);
      }

    private:
      LionParams options_;
      LionState<Ts...> state_;

      constexpr const char* optimizer_name() const { return "lion\0"; }
  };
}
