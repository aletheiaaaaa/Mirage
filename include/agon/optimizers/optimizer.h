#pragma once

#include "../core/parameter.h"

#include <algorithm>
#include <variant>
#include <vector>
#include <cstdint>

namespace agon::optim {
    struct OptimizerState {
        int64_t step = 0;
    };

    template<typename... Ts>
    class Optimizer {
        public:
            explicit Optimizer(ParameterPack<Ts...> parameters) : parameters_(parameters) {}

            void zero_grad();

            template<typename T>
                requires (std::same_as<T, Parameter<typename T::DataType>>)
            void add_parameter(T& param) {
                parameters_.template add_parameter(param);
            }

            virtual void step() = 0;

            virtual void load_from_bin(const std::string& path) = 0;
            virtual void save_to_bin(const std::string& path) const = 0;

            ~Optimizer() = default;
        protected:
            OptimizerState state_;
            ParameterPack<Ts...> parameters_;
    };

    template<typename... Ts>
    void Optimizer<Ts...>::zero_grad() {
        std::apply([](auto&... param_vecs) {
            (std::ranges::for_each(param_vecs.begin(), param_vecs.end(), [](auto& param) {
                param.zero_grad();
            }), ...);
        }, parameters_.data);
    }
}
