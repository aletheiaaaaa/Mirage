#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <filesystem>
#include <fstream>
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
              auto& param_og = param_ref.get();
              using T = typename ParamType::DataType;
              int width = param_og.size(0);
              int height = param_og.strides(0);

              auto param_tp = param_og.copy();
              param_tp.view(std::array{width, height});
              param_tp.transpose(0, 1);
              param_tp.view(
                std::rotate(
                  param_og.size().begin(), param_og.size().begin() + 1, param_og.size().end()
                )
              );

              auto& grad_full = param_og.grad();
              auto& data_full = param_og.data();

              constexpr int vec_size = eve::wide<T>::size();

              int chunk_size = (param_og.numel() + options_.num_proc - 1) / options_.num_proc;

              detail::parallel(
                [&](int i) {
                  // TODO: actually do this;
                },
                options_.num_proc
              );
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

  std::string optimizer_type() {
    return "Muon<" + detail::type_names(this->parameters_.data) + ">";
  }
};
}  // namespace mirage::optim
