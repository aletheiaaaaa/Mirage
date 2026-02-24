#pragma once

#include <vector>
#include <array>
#include <cstdint>
#include <cstddef>
#include <string>
#include <typeinfo>
#include <span>

#include "detail/dedup.h"

namespace agon {
    template<typename T>
    class Parameter {
        public:
            using DataType = T;

            Parameter(size_t size);
            Parameter(const std::span<T>& data);

            std::vector<T>& grad();
            const std::vector<T>& grad() const;

            std::vector<T>& data();
            const std::vector<T>& data() const;

            size_t size() const;

            void zero_grad();

            void accumulate(const std::vector<T>& new_grad);
            void update(const std::vector<T>& new_val);

        private:
            std::vector<T> data_;
            std::vector<T> grad_;
    };

    template<typename Q, typename T>
    class Quantized : public Parameter<T> {
        public:
            using QuantizedType = Q;

            Quantized(size_t size);
            Quantized(const std::span<Q>& data, float scale = 1.0f, float zero_point = 0.0f);

            std::vector<Q> quantized() const;
            std::vector<T> fake_quantized() const;

            float scale() const;
            float zero_point() const;

        private:
            float scale_ = 1.0f;
            float zero_point_ = 0.0f;
    };

    template<typename DedupedTuple>
    struct ParameterPack {
        dedup::TransformTuple_t<std::vector, DedupedTuple> data{};

        template<typename... Ts>
            requires (std::same_as<Ts, Parameter<typename Ts::DataType>> && ...)
        ParameterPack(Ts&... params) {
            (std::get<std::vector<Ts>>(data).push_back(std::forward<Ts>(params)), ...);
        }

        template<typename T>
            requires (std::same_as<T, Parameter<typename T::DataType>>)
        void add_parameter(T& param) {
            std::get<std::vector<T>>(data).push_back(std::forward<T>(param));
        }
    };
}