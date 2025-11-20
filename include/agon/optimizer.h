#pragma once

#include "parameter.h"

#include <vector>
#include <cstdint>

namespace agon::optim {
    struct OptimizerState {
        int64_t step = 0;
    };

    class Optimizer {
        public:
            template<typename... Params>
            explicit Optimizer(Params&... params);
            explicit Optimizer(std::initializer_list<ParameterView*> params);

            void zero_grad();
            virtual void step() = 0;

            virtual void load_from_bin(const std::string& path) = 0;
            virtual void save_to_bin(const std::string& path) const = 0;

            ~Optimizer() = default;
        protected:
            OptimizerState state;
            std::vector<ParameterView*> parameters;
    };
}
