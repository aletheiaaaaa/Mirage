#include "agon/optimizers/impls/sarah.h"

#include "agon/detail/simd/ops.h"
#include "agon/detail/simd/utils.h"

#include <fstream>
#include <filesystem>

namespace agon::optim {
    template<typename... Ts>
    void Sarah<Ts...>::step() {
        std::apply([&](auto&... param_vecs) {
            (std::ranges::for_each(param_vecs, [&](auto& param_ref) {
                auto& param = param_ref.get();
                using T = typename std::unwrap_ref_decay_t<decltype(param)>::DataType;

                auto& grad_full = param.grad();
                auto& data_full = param.data();
                auto& prev_grad_full = std::get<std::vector<T>>(state_.prev_grad);
                auto& prev_update_full = std::get<std::vector<T>>(state_.prev_update);

                constexpr size_t vec_size = simd::vec<T>::size;
                constexpr size_t unroll_factor = simd::UNROLL_FACTOR;

                size_t i = 0;
                for (; i + vec_size * unroll_factor <= grad_full.size(); i += vec_size * unroll_factor) {
                    simd::unroll<unroll_factor>([&]<size_t index>() {
                        constexpr size_t offset = index * vec_size;

                        auto grad = simd::load<T>(&grad_full[i + offset]);
                        if (options_.maximize) grad = simd::neg(grad);

                        auto update = [&](){
                            if (options_.recompute && state_.step % options_.recompute_interval == 0) return grad;

                            auto prev_grad = simd::load<T>(&prev_grad_full[i + offset]);
                            auto prev_update = simd::load<T>(&prev_update_full[i + offset]);

                            return simd::add(simd::sub(grad, prev_grad), prev_update);
                        }();

                        auto data = simd::load<T>(&data_full[i + offset]);
                        data = simd::fmadd(simd::set1<T>(options_.lr), update, data);

                        simd::store(&data_full[i + offset], data);
                        simd::store(&prev_grad_full[i + offset], grad);
                    });
                }

                for (; i < grad_full.size(); ++i) {
                    T grad = options_.maximize ? -grad_full[i] : grad_full[i];
                    T update = (options_.recompute && state_.step % options_.recompute_interval == 0) 
                        ? grad : grad - prev_grad_full[i] + prev_update_full[i];

                    data_full[i] += options_.lr * update;
                    prev_grad_full[i] = grad;
                }
            }), ...);
        }, this->parameters_.data);

        state_.step++;
    }

    template<typename... Ts>
    void Sarah<Ts...>::load_from_bin(const std::string& path_str) {
        std::filesystem::path path(path_str);
        if (!std::filesystem::exists(path)) throw std::runtime_error("File not found: " + path_str);

        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("Failed to open file: " + path_str);

        in.read(reinterpret_cast<char*>(&options_), sizeof(options_));
        in.read(reinterpret_cast<char*>(&state_.step), sizeof(state_.step));

        std::apply([&](auto&... param_vecs) {
            (std::ranges::for_each(param_vecs, [&](auto& param_ref) {
                auto& param = param_ref.get();
                using T = typename std::unwrap_ref_decay_t<decltype(param)>::DataType;
                auto& prev_grad = std::get<std::vector<T>>(state_.prev_grad);
                auto& prev_update = std::get<std::vector<T>>(state_.prev_update);

                in.read(reinterpret_cast<char*>(&param.data()), param.size() * sizeof(T));
                in.read(reinterpret_cast<char*>(prev_grad.data()), prev_grad.size() * sizeof(T));
                in.read(reinterpret_cast<char*>(prev_update.data()), prev_update.size() * sizeof(T));
            }), ...);
        }, this->parameters_.data);
    }

    template<typename... Ts>
    void Sarah<Ts...>::save_to_bin(const std::string& path_str) const {
        std::filesystem::path path(path_str);
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("Failed to open file: " + path_str);

        out.write(reinterpret_cast<const char*>(&options_), sizeof(options_));
        out.write(reinterpret_cast<const char*>(&state_.step), sizeof(state_.step));

        std::apply([&](auto&... param_vecs) {
            (std::ranges::for_each(param_vecs, [&](auto& param_ref) {
                auto& param = param_ref.get();
                using T = typename std::unwrap_ref_decay_t<decltype(param)>::DataType;
                auto& prev_grad = std::get<std::vector<T>>(state_.prev_grad);
                auto& prev_update = std::get<std::vector<T>>(state_.prev_update);

                out.write(reinterpret_cast<const char*>(&param.data()), param.size() * sizeof(T));
                out.write(reinterpret_cast<const char*>(prev_grad.data()), prev_grad.size() * sizeof(T));
                out.write(reinterpret_cast<const char*>(prev_update.data()), prev_update.size() * sizeof(T));
            }), ...);
        }, this->parameters_.data);
    }

    template class Sarah<std::tuple<Parameter<float>>>;
    template class Sarah<std::tuple<Parameter<double>>>;
    template class Sarah<std::tuple<Parameter<float>, Parameter<double>>>;
}