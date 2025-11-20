#include "../include/agon/optimizer.h"

namespace agon::optim {
    template<typename... Params>
    Optimizer::Optimizer(Params&... params) : parameters{ &params... } {}

    Optimizer::Optimizer(std::initializer_list<ParameterView*> params) : parameters(params) {}

    void Optimizer::zero_grad() {
        for (auto& param : parameters) {
            param->zero_grad();
        }
    }
}