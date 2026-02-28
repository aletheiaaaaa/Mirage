#pragma once

#include "../optimizer.h"

#include <algorithm>

namespace agon::optim {
    struct LionParams {
        float lr = 1e-5f;
        float beta1 = 0.9f;
        float beta2 = 0.4f;
        float epsilon = 1e-8;
        float lambda = 0.0f;

        bool maximize = false;
    };

    template<typename DedupedTuple>
    struct LionState : OptimizerState {
        dedup::TransformTuple_t<std::vector, dedup::TransformTuple_t<ExtractType_t, DedupedTuple>> momenta{};
    };

    template<typename... Ts>
    class Lion : public Optimizer<Ts...> {
        public:
            explicit Lion(ParameterPack<Ts...> parameters, LionParams options = {})
                : Optimizer<Ts...>(parameters), options_(options) {
                    std::apply([&](auto&... param_vecs) {
                        (std::ranges::for_each(param_vecs, [&](auto& param_ref) {
                            auto& param = param_ref.get();
                            using T = typename std::unwrap_ref_decay_t<decltype(param)>::DataType;
                            auto& mom = std::get<std::vector<T>>(this->state_.momenta);
                            mom.insert(mom.end(), param.size(), T(0));
                        }), ...);
                    }, this->parameters_.data);
                }

            void step() override;

            void load_from_bin(const std::string& path_str) override;
            void save_to_bin(const std::string& path_str) const override;

        private:
            LionParams options_;
            LionState<Ts...> state_;
    };
}