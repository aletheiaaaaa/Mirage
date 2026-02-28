#pragma once

#include "../optimizer.h"

#include <algorithm>

namespace agon::optim {
    struct SarahParams {
        float lr = 0.01f;

        bool recompute = false;
        int recompute_interval = 0;

        bool maximize = false;
    };

    template<typename DedupedTuple>
    struct SarahState : public OptimizerState {
        dedup::TransformTuple_t<std::vector, dedup::TransformTuple_t<ExtractType_t, DedupedTuple>> prev_grad{};
        dedup::TransformTuple_t<std::vector, dedup::TransformTuple_t<ExtractType_t, DedupedTuple>> prev_update{};
    };

    template<typename... Ts>
    class Sarah : public Optimizer<Ts...> {
        public:
            explicit Sarah(ParameterPack<Ts...> parameters, SarahParams options = {})
                : Optimizer<Ts...>(parameters), options_(options) {
                    std::apply([&](auto&... param_vecs) {
                        (std::ranges::for_each(param_vecs, [&](auto& param_ref) {
                            auto& param = param_ref.get();
                            using T = typename std::unwrap_ref_decay_t<decltype(param)>::DataType;
                            auto& prev_grad = std::get<std::vector<T>>(this->state_.prev_grad);
                            prev_grad.insert(prev_grad.end(), param.size(), T(0));
                        }), ...);
                    }, this->parameters_.data);
                }

            void step() override;

            void load_from_bin(const std::string& path_str) override;
            void save_to_bin(const std::string& path_str) const override;

        private:
            SarahParams options_;
            SarahState<Ts...> state_;
    };
}