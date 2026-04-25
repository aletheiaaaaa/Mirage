#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "../../detail/matrix.hpp"
#include "../../detail/utils.hpp"
#include "../optimizer.hpp"

namespace mirage::optim {
struct MuonOptions {
  float lr = 0.01f;
  float momentum = 0.9f;
  float epsilon = 1e-7;
  int newton_schulz_iters = 5;
  float lambda = 0.0f;

  int num_proc = 1;

  bool maximize = false;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(lr, momentum, epsilon, newton_schulz_iters, lambda, maximize);
  }
};

template <typename TypeTuple>
struct MuonState : public OptimizerState {
  detail::ExtractedVector<TypeTuple> momentum{};
};

template <typename DedupedPack>
class Muon : public Optimizer<DedupedPack> {
  public:
  explicit Muon(ParameterPack<DedupedPack> parameters, MuonOptions options = {})
    : Optimizer<DedupedPack>(parameters), options_(options) {
    detail::test_multidim(this->parameters_.data);
    detail::test_oom(this->parameters_.data, [&](auto& param) { return param.numel(); });

    std::apply(
      [&](auto&... param_vecs) {
        (
          [&](auto& param_vec) {
            using ParamType = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type;
            auto& mom = std::get<detail::ExtractType_t<ParamType>>(state_.momentum);
            for (auto& param_ref : param_vec) {
              auto& param = param_ref.get();
              using T = typename ParamType::DataType;
              mom.insert(mom.end(), param.numel(), T(0));
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
            using ParamType = typename std::remove_cvref_t<decltype(param_vec)>::value_type::type;
            auto& mom_full = std::get<detail::ExtractType_t<ParamType>>(state_.momentum);

            int state_offset = 0;
            for (auto& param_ref : param_vec) {
              auto& param = param_ref.get();
              using T = typename ParamType::DataType;
              int width = param.size(0);
              int height = param.strides(0);

              auto og_mom_slice = std::span(mom_full).subspan(state_offset, param.numel());

              auto& grad_full = param.grad();
              auto& data_full = param.data();

              constexpr int vec_size = eve::wide<T>::size();

              const auto chunks = [&](int chunk_width, int chunk_height) {
                if (options_.num_proc % 2) {
                  return std::make_pair(
                    (chunk_width + options_.num_proc - 1) / options_.num_proc, chunk_height
                  );
                }
                return std::make_pair(
                  (2 * chunk_width + options_.num_proc - 1) / options_.num_proc,
                  (chunk_height + 1) / 2
                );
              };

              const auto offsets = [&](int id, int chunk_width, int chunk_height) {
                if (options_.num_proc % 2) {
                  return std::make_pair(id * chunk_width, 0);
                }
                return std::make_pair((id / 2) * chunk_width, (id % 2) * chunk_height);
              };

              int smaller = std::min(width, height);
              int larger = std::max(width, height);

              const auto [sq_width, sq_height] = chunks(smaller, smaller);
              const auto [rc_width, rc_height] = chunks(smaller, larger);
              const auto [og_width, og_height] = chunks(width, height);

              detail::parallel(
                [&](int i) {
                  const auto [x_off, y_off] = offsets(i, og_width, og_height);
                  detail::fma_tile(
                    std::span<const T>(grad_full),
                    std::span<T>(og_mom_slice),
                    width,
                    height,
                    std::min(og_width, width - x_off),
                    std::min(og_height, height - y_off),
                    x_off,
                    y_off,
                    options_.momentum
                  );
                },
                options_.num_proc
              );

              // TODO: add frobenius normalization;

              auto tp_mom_slice =
                detail::transpose(std::span<const T>(og_mom_slice), height, width);

              std::vector<T> og_mom, tp_mom;

              if (width > height) {
                og_mom = std::move(tp_mom_slice);
                tp_mom = std::vector<T>(og_mom_slice.begin(), og_mom_slice.end());
              } else {
                og_mom = std::vector<T>(og_mom_slice.begin(), og_mom_slice.end());
                tp_mom = std::move(tp_mom_slice);
              }

              for (int iter = 0; iter < options_.newton_schulz_iters; ++iter) {
                std::vector<T> step1(smaller * smaller, T(0));
                detail::parallel(
                  [&](int i) {
                    const auto [sq_x_off, sq_y_off] = offsets(i, sq_width, sq_height);

                    detail::symmetrized_tile(
                      std::span<const T>(og_mom),
                      std::span<const T>(tp_mom),
                      std::span<T>(step1),
                      smaller,
                      larger,
                      std::min(sq_width, smaller - sq_x_off),
                      std::min(sq_height, smaller - sq_y_off),
                      sq_x_off,
                      sq_y_off
                    );
                  },
                  options_.num_proc
                );

                std::vector<T> step2(smaller * smaller, T(0));
                detail::parallel(
                  [&](int i) {
                    const auto [sq_x_off, sq_y_off] = offsets(i, sq_width, sq_height);

                    detail::quadratic_tile(
                      std::span<const T>(step1),
                      std::span<T>(step2),
                      -4.7750f,
                      2.0315f,
                      smaller,
                      std::min(sq_width, smaller - sq_x_off),
                      std::min(sq_height, smaller - sq_y_off),
                      sq_x_off,
                      sq_y_off
                    );
                  },
                  options_.num_proc
                );

                std::vector<T> step3(smaller * larger, T(0));
                detail::parallel(
                  [&](int i) {
                    const auto [rc_x_off, rc_y_off] = offsets(i, rc_width, rc_height);

                    detail::ns_final_tile(
                      std::span<const T>(step2),
                      std::span<const T>(og_mom),
                      std::span<T>(step3),
                      3.4445f,
                      smaller,
                      larger,
                      std::min(rc_width, smaller - rc_x_off),
                      std::min(rc_height, larger - rc_y_off),
                      rc_x_off,
                      rc_y_off
                    );
                  },
                  options_.num_proc
                );

                og_mom = step3;
                tp_mom = detail::transpose(std::span<const T>(step3), larger, smaller);
              }

              auto span = std::span<const T>(og_mom);
              if (width > height) {
                auto transposed = detail::transpose(span, smaller, larger);
                std::copy(transposed.begin(), transposed.end(), og_mom_slice.begin());
              } else {
                std::copy(span.begin(), span.end(), og_mom_slice.begin());
              }

              auto& update = (width > height) ? og_mom : tp_mom;

              detail::parallel(
                [&](int i) {
                  const auto [x_off, y_off] = offsets(i, og_width, og_height);
                  if (options_.lambda) {
                    detail::fma_tile(
                      std::span<const T>(data_full),
                      std::span(update),
                      width,
                      height,
                      og_width,
                      og_height,
                      x_off,
                      y_off,
                      -options_.lambda
                    );
                  }

                  detail::fma_tile(
                    std::span<const T>(update),
                    std::span(data_full),
                    width,
                    height,
                    og_width,
                    og_height,
                    x_off,
                    y_off,
                    options_.lr
                  );
                },
                options_.num_proc
              );

              state_offset += param.numel();
            }
          }(param_vecs),
          ...);
      },
      this->parameters_.data
    );
    state_.step++;
  }

  void load_from_bin(const std::string& path_str) override {
    std::filesystem::path path(path_str);
    path.replace_extension(".bin");

    if (!std::filesystem::exists(path)) throw std::runtime_error("File not found: " + path_str);

    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open file: " + path_str);

    cereal::BinaryInputArchive ar(in);
    std::string name;
    ar(name);
    if (name != optimizer_type()) this->handle_type_error(name);

    ar(options_, state_.step, state_.momentum);

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

    ar(name, options_, state_.step, state_.momentum);

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

  private:
  MuonOptions options_;
  MuonState<DedupedPack> state_;

  std::string optimizer_type() const override {
    return "Muon<" + detail::type_names(this->parameters_.data) + ">";
  }
};
}  // namespace mirage::optim
