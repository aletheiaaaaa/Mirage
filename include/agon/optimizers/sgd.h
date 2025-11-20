#pragma once

#include "../optimizer.h"  

namespace agon::optim {
    struct SGDParams {
        float lr = 0.01f;
        float momentum = 0.0f;

        bool nesterov = false;
        bool maximize = false;
    };

    struct SGDState : public OptimizerState {
        std::variant<std::vector<float>, std::vector<double>> momenta;
    };

    template<class ...Params>
    class SGD : public Optimizer {
        public:
            explicit SGD(
                Params&... params, 
                float learning_rate = 0.01f, 
                float momentum = 0.0f, 
                bool nesterov = false, 
                bool maximize = false
            );

            void step() override;

            void load_from_bin(const std::string& path);
            void save_to_bin(const std::string& path) const;

        private:
            SGDParams options;
            SGDState state;
    };
}