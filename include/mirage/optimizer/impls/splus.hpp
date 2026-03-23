#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../../detail/matrix.hpp"
#include "../../detail/utils.hpp"
#include "../optimizer.hpp"

namespace mirage::optim {
struct SPlusOptions {
  float lr = 0.5f;
  float beta1 = 0.9f;
  float beta2 = 0.4f;
  float beta3 = 0.999f;
  float lambda = 0.0f;
  int decompose_every = 64;

  int num_proc = 1;

  bool maximize = false;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(lr, beta1, beta2, beta3, lambda, decompose_every, maximize);
  }
};

template <typename DedupedTuple>
struct SPlusState : public OptimizerState {
  detail::ExtractedVector<DedupedTuple> momentum{};
  detail::ExtractedVector<DedupedTuple> left_velocity{};
  detail::ExtractedVector<DedupedTuple> right_velocity{};
  detail::ExtractedVector<DedupedTuple> left_eigenvectors{};
  detail::ExtractedVector<DedupedTuple> right_eigenvectors{};
  detail::ExtractedVector<DedupedTuple> param_ema{};
};

template <typename DedupedPack>
  requires detail::NonConstPack<DedupedPack>
class SPlus : public Optimizer<DedupedPack> {
  public:
  explicit SPlus(
    ParameterPack<DedupedPack> parameters, SPlusOptions options = {}
  )
      : Optimizer<DedupedPack>(parameters), options_(options) {
    assert(
      std::apply(
        [](auto&... param_vecs) {
          return (
            std::all_of(
              param_vecs.begin(),
              param_vecs.end(),
              [](auto& p) { return p.get().rank() >= 2; }
            ) &&
            ...
          );
        },
        this->parameters_.data
      ) &&
      "All parameters must have at least 2 dimensions"
    );

    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            using ParamType = typename std::remove_cvref_t<
              decltype(param_vec)>::value_type::type;
            auto& mom =
              std::get<detail::ExtractType_t<ParamType>>(this->state_.momentum);
            auto& lvel = std::get<detail::ExtractType_t<ParamType>>(
              this->state_.left_velocity
            );
            auto& rvel = std::get<detail::ExtractType_t<ParamType>>(
              this->state_.right_velocity
            );
            auto& leig = std::get<detail::ExtractType_t<ParamType>>(
              this->state_.left_eigenvectors
            );
            auto& reig = std::get<detail::ExtractType_t<ParamType>>(
              this->state_.right_eigenvectors
            );
            auto& ema = std::get<detail::ExtractType_t<ParamType>>(
              this->state_.param_ema
            );
            for (auto& param_ref : param_vec) {
              auto& param = param_ref.get();
              size_t l_numel = param.size(0) * param.size(0);
              size_t r_numel = param.strides(0) * param.strides(0);

              using T = typename ParamType::DataType;
              mom.insert(mom.end(), param.numel(), T(0));
              lvel.insert(lvel.end(), l_numel, T(0));
              rvel.insert(rvel.end(), r_numel, T(0));
              leig.insert(leig.end(), l_numel, T(0));
              reig.insert(reig.end(), r_numel, T(0));
              ema.insert(ema.end(), param.numel(), T(0));
            }
          }(param_vecs),
          ...);
      },
      this->parameters_.data
    );
  }

  void step() override {
    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            using ParamType = typename std::remove_cvref_t<
              decltype(param_vec)>::value_type::type;
            auto& mom_full =
              std::get<detail::ExtractType_t<ParamType>>(state_.momentum);
            auto& lvel_full =
              std::get<detail::ExtractType_t<ParamType>>(state_.left_velocity);
            auto& rvel_full =
              std::get<detail::ExtractType_t<ParamType>>(state_.right_velocity);
            auto& leig_full = std::get<detail::ExtractType_t<ParamType>>(
              state_.left_eigenvectors
            );
            auto& reig_full = std::get<detail::ExtractType_t<ParamType>>(
              state_.right_eigenvectors
            );
            auto& ema_full =
              std::get<detail::ExtractType_t<ParamType>>(state_.param_ema);

            size_t state_offset = 0;
            for (auto param_ref : param_vec) {
              auto& param_og = param_ref.get();
              using T = typename ParamType::DataType;
              size_t width = param_og.size(0);
              size_t height = param_og.strides(0);

              auto& param_tp = param_og.copy();
              param_tp.view(width, height);
              param_tp.transpose(0, 1);
              param_tp.view(param_og.size());

              size_t numel_off = param_og.numel() - 1;

              auto& mom_slice =
                std::span(mom_full).subspan(state_offset, numel_off);
              auto& rvel_slice =
                std::span(rvel_full).subspan(state_offset, numel_off);
              auto& lvel_slice =
                std::span(lvel_full).subspan(state_offset, numel_off);
              auto& reig_slice =
                std::span(reig_full).subspan(state_offset, numel_off);
              auto& leig_slice =
                std::span(leig_full).subspan(state_offset, numel_off);
              auto& ema_slice =
                std::span(ema_full).subspan(state_offset, numel_off);

              auto& og_grad_full = param_og.grad();
              auto& tp_grad_full = param_tp.grad();
              auto& data_full = param_og.data();

              constexpr size_t vec_size = eve::wide<T>::size();
              constexpr size_t unroll_factor = detail::UNROLL_FACTOR;

              auto compute_chunks = [&](
                                      size_t chunk_width, size_t chunk_height
                                    ) {
                if (options_.num_proc % 2) {
                  return std::make_pair(
                    (chunk_width + options_.num_proc - 1) / options_.num_proc,
                    chunk_height
                  );
                }
                return std::make_pair(
                  (2 * chunk_width + options_.num_proc - 1) / options_.num_proc,
                  (chunk_height + 1) / 2
                );
              };

              const auto offsets =
                [&](size_t id, size_t chunk_width, size_t chunk_height) {
                  if (options_.num_proc % 2) {
                    return std::make_pair(id * chunk_width, 0);
                  }
                  return std::make_pair(
                    (id / 2) * chunk_width, (id % 2) * chunk_height
                  );
                };

              const auto [wh_width, wh_height] = compute_chunks(width, height);
              const auto [ww_width, ww_height] = compute_chunks(width, width);
              const auto [hh_width, hh_height] = compute_chunks(height, height);

              detail::parallel<options_.num_proc>([&]<size_t index>() {
                const auto [wh_x_off, wh_y_off] =
                  offsets(index, wh_width, wh_height);
                const auto [ww_x_off, ww_y_off] =
                  offsets(index, ww_width, ww_height);
                const auto [hh_x_off, hh_y_off] =
                  offsets(index, hh_width, hh_height);

                detail::symmetrized_ema_tile(
                  og_grad_full,
                  tp_grad_full,
                  lvel_slice,
                  width,
                  height,
                  std::min(ww_width, width - ww_x_off),
                  std::min(ww_height, height - ww_y_off),
                  ww_x_off,
                  ww_y_off,
                  options_.beta2
                );

                detail::symmetrized_ema_tile(
                  tp_grad_full,
                  og_grad_full,
                  rvel_slice,
                  height,
                  width,
                  std::min(hh_width, width - hh_x_off),
                  std::min(hh_height, height - hh_y_off),
                  hh_x_off,
                  hh_y_off,
                  options_.beta2
                );

                detail::ema_tile(
                  og_grad_full,
                  mom_slice,
                  width,
                  height,
                  std::min(wh_width, width - wh_x_off),
                  std::min(wh_height, height - wh_y_off),
                  wh_x_off,
                  wh_y_off,
                  options_.beta1
                );
              });

              using Matrix = Eigen::
                Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

              Eigen::Map<Matrix> lvel_format(lvel_full, width, width);
              Eigen::Map<Matrix> rvel_format(rvel_full, height, height);

              Eigen::EigenSolver<Matrix> lvel_solver(lvel_format);
              Eigen::EigenSolver<Matrix> rvel_solver(rvel_format);

              auto leig_new = lvel_solver.eigenvectors().real();
              auto reig_new = rvel_solver.eigenvectors().real();

              Eigen::Map<Matrix>(leig_slice, width, width) = leig_new;
              Eigen::Map<Matrix>(reig_slice, height, height) = reig_new;

              std::vector<T> temp;
              detail::parallel<options_.num_proc>([&]<size_t index>() {
                const auto [x_off, y_off] = offsets(index, wh_width, wh_height);

                detail::triple_matmul_sign(
                  detail::transpose(
                    std::vector(leig_slice.begin(), leig_slice.end())
                  ),
                  mom_slice,
                  std::vector(reig_slice.begin(), reig_slice.end()),
                  temp,
                  width,
                  width,
                  height,
                  height,
                  wh_width,
                  wh_height,
                  x_off,
                  y_off
                );
              });

              std::vector<T> update;
              detail::parallel<options_.num_proc>([&]<size_t index>() {
                const auto [x_off, y_off] = offsets(index, wh_width, wh_height);

                detail::triple_matmul_tile(
                  std::vector(leig_slice.begin(), leig_slice.end()),
                  mom_slice,
                  detail::transpose(
                    std::vector(reig_slice.begin(), reig_slice.end())
                  ),
                  update,
                  width,
                  width,
                  height,
                  height,
                  wh_width,
                  wh_height,
                  x_off,
                  y_off
                );

                detail::fma_tile(
                  update,
                  ema_slice,
                  width,
                  height,
                  wh_width,
                  wh_height,
                  x_off,
                  y_off,
                  options_.lr
                );

                detail::ema_tile(
                  ema_slice,
                  data_full,
                  width,
                  height,
                  wh_width,
                  wh_height,
                  x_off,
                  y_off,
                  options_.beta3
                );
              });

              state_offset += param_og.numel();
            }
          }(param_vecs),
          ...);
      },
      this->parameters_.data
    );
    this->state_.step++;
  }

  void load_from_bin(const std::string& path_str) override {
    std::filesystem::path path(path_str);
    path.replace_extension(".bin");

    if (!std::filesystem::exists(path))
      throw std::runtime_error("File not found: " + path_str);

    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open file: " + path_str);

    cereal::BinaryInputArchive ar(in);
    std::string name;
    ar(name);
    if (name != optimizer_type()) this->handle_type_error(name);

    ar(
      options_,
      state_.step,
      state_.momentum,
      state_.left_velocity,
      state_.right_velocity,
      state_.param_ema
    );

    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            for (auto& param_ref : param_vec) {
              ar(param_ref.get().data());
            }
          }(param_vecs),
          ...);
      },
      this->parameters_.data
    );
  }

  void save_to_bin(const std::string& path_str) const override {
    std::filesystem::path path(path_str);
    path.replace_extension(".bin");

    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open file: " + path_str);

    cereal::BinaryOutputArchive ar(out);
    std::string name(optimizer_type());

    ar(
      name,
      options_,
      state_.step,
      state_.momentum,
      state_.left_velocity,
      state_.right_velocity,
      state_.param_ema
    );

    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            for (auto& param_ref : param_vec) {
              ar(param_ref.get().data());
            }
          }(param_vecs),
          ...);
      },
      this->parameters_.data
    );
  }

  std::string optimizer_type() const override {
    std::string type = "SPlus<";
    bool first = true;

    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            using ParamType = typename std::remove_cvref_t<
              decltype(param_vec)>::value_type::type;

            if (!first) type += ", ";
            first = false;
            type += detail::PrintType<ParamType>::name() + "[";

            bool pfirst = true;
            for (auto& param_ref : param_vec) {
              if (!pfirst) type += ",";
              pfirst = false;

              auto& shape = param_ref.get().size();
              for (size_t i = 0; i < shape.size(); ++i) {
                if (i > 0) type += "x";
                type += std::to_string(shape[i]);
              }
            }

            type += "]";
          }(param_vecs),
          ...);
      },
      this->parameters_.data
    );

    type += ">";
    return type;
  }

  private:
  SPlusOptions options_;
  SPlusState<DedupedPack> state_;
};
}  // namespace mirage::optim