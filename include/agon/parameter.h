#pragma once

#include <vector>
#include <array>
#include <cstdint>
#include <unordered_map>
#include <string>
#include <variant>
#include <functional>
#include <memory>

namespace agon {
    class ParameterView {
        public:
            virtual ~ParameterView() = default;

            virtual size_t size() const = 0;
            virtual void zero_grad() = 0;
    };

    template<typename T, typename G = float>
    class Parameter : public ParameterView {
        public:
            Parameter(size_t size);
            Parameter(const std::vector<T>& data);
            template<size_t N>
            Parameter(const std::array<T, N>& data);

            std::vector<G>& grad();
            const std::vector<G>& grad() const;

            std::vector<T>& data();
            const std::vector<T>& data() const;

            size_t size() const override;

            void zero_grad() override;

            void accumulate(const std::vector<G>& new_grad);
            void update(const std::vector<T>& new_val);
        private:
            std::vector<T> vals;
            std::vector<G> grads;
    };
}