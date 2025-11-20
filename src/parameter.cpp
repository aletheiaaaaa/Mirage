#include "../include/agon/parameter.h"

namespace agon {
    template<typename T, typename G>
    Parameter<T, G>::Parameter(size_t size) : vals(size), grads(size, T(0)) {};

    template<typename T, typename G>
    template<size_t N>
    Parameter<T, G>::Parameter(const std::array<T, N>& data) : vals(data), grads(data.size(), T(0)) {};

    template<typename T, typename G>
    Parameter<T, G>::Parameter(const std::vector<T>& data) : vals(data), grads(data.size(), T(0)) {};

    template<typename T, typename G>
    std::vector<G>& Parameter<T, G>::grad() {
        return grads;
    }

    template<typename T, typename G>
    const std::vector<G>& Parameter<T, G>::grad() const {
        return grads;
    }

    template<typename T, typename G>
    std::vector<T>& Parameter<T, G>::data() {
        return vals;
    }

    template<typename T, typename G>
    const std::vector<T>& Parameter<T, G>::data() const {
        return vals;
    }

    template<typename T, typename G>
    size_t Parameter<T, G>::size() const {
        return vals.size();
    }

    template<typename T, typename G>
    void Parameter<T, G>::zero_grad() {
        std::fill(grads.begin(), grads.end(), T(0));
    }

    template<typename T, typename G>
    void Parameter<T, G>::accumulate(const std::vector<G>& new_grad) {
        for (size_t i = 0; i < grads.size(); i++) {
            grads[i] += new_grad[i];
        }
    }

    template<typename T, typename G>
    void Parameter<T, G>::update(const std::vector<T>& new_val) {
        vals = new_val;
    }
}