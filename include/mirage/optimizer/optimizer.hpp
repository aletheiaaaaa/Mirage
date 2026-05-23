#pragma once

#include <cstdint>
#include <sstream>

#include "../detail/thread.hpp"
#include "../parameter.hpp"

namespace mirage {
namespace detail {
void test_multidim(auto& parameters) {
  bool is_multidim = std::apply(
    [](auto&... param_vecs) {
      return (
        std::all_of(
          param_vecs.begin(), param_vecs.end(), [](auto& p) { return p.get().rank() >= 2; }
        ) &&
        ...
      );
    },
    parameters
  );

  if (!is_multidim)
    throw std::runtime_error("All parameters must have at least 2 dimensions for this optimizer");
}

template <typename F>
void test_oom(auto& parameters, F&& func) {
  int64_t bytes = 0;
  std::apply(
    [&](auto&... param_vecs) {
      (
        [&](auto& param_vec) {
          using ParamType =
            typename std::remove_cvref_t<decltype(param_vec)>::value_type::type::DataType;

          std::ranges::for_each(param_vec, [&](auto& param_ref) {
            auto param = param_ref.get();
            bytes += sizeof(ParamType) * func(param);
          });
        }(param_vecs),
        ...);
    },
    parameters
  );

  bytes /= (1024 * 1024);
  const char* max = std::getenv("MIRAGE_OPTIMIZER_MEMMAX");
  int max_bytes = (max == nullptr) ? 512 : std::stoi(max);
  if (bytes > max_bytes)
    throw std::runtime_error(
      "Out of memory, tried to allocate " + std::to_string(bytes) + " MiB out of a max of " +
      std::to_string(max_bytes) +
      " MiB. You can increase this using the \"MIRAGE_OPTIMIZER_MEMMAX\" environment "
      "variable."
    );
}
}  // namespace detail

namespace optim {
struct OptimizerState {
  int64_t step = 0;
};

template <typename DedupedPack>
  requires detail::NonConstPack<DedupedPack>
class Optimizer {
  public:
  explicit Optimizer(ParameterPack<DedupedPack> parameters, int num_proc = 1)
    : parameters_(parameters), pool_(num_proc) {}

  void zero_grad() {
    std::apply(
      [](auto&... param_vecs) {
        (std::ranges::for_each(
           param_vecs,
           [](auto& param_ref) {
             auto& param = param_ref.get();
             param.zero_grad();
           }
         ),
         ...);
      },
      parameters_.data
    );
  }

  virtual void step() = 0;

  virtual void load_from_bin(const std::string& path) = 0;
  virtual void save_to_bin(const std::string& path) const = 0;

  ~Optimizer() = default;

  protected:
  ParameterPack<DedupedPack> parameters_;
  detail::ThreadPool pool_;

  virtual std::string optimizer_type() const = 0;

  void handle_type_error(const std::string& recieved) const {
    const std::string expected = optimizer_type();
    std::ostringstream err;

    auto get_name = [](const std::string& type) {
      auto delimit = type.find('<');
      return type.substr(0, delimit);
    };

    auto get_body = [](const std::string& type) {
      auto open = type.find('<');
      return type.substr(open + 1, type.length() - open - 2);
    };

    auto get_types = [](const std::string& body) {
      std::vector<std::string> types;
      int depth = 0, start = 0;

      for (int i = 0; i < body.size(); ++i) {
        if (body[i] == '<')
          ++depth;
        else if (body[i] == '>')
          --depth;
        else if (body[i] == ',' && depth == 0) {
          auto token = body.substr(start, i - start);
          if (!token.empty() && token.front() == ' ') token.erase(0, 1);
          types.push_back(token);
          start = i + 1;
        }
      }

      auto token = body.substr(start);
      if (!token.empty() && token.front() == ' ') token.erase(0, 1);
      if (!token.empty()) types.push_back(token);

      return types;
    };

    auto get_type_name = [](const std::string& group) {
      auto bracket = group.find('[');
      return bracket != std::string::npos ? group.substr(0, bracket) : group;
    };

    auto get_shapes = [](const std::string& group) -> std::vector<std::string> {
      auto open = group.find('[');
      auto close = group.rfind(']');
      if (open == std::string::npos || close == std::string::npos) return {};

      std::vector<std::string> shapes;
      std::string inner = group.substr(open + 1, close - open - 1);
      int start = 0;

      for (int i = 0; i <= inner.size(); ++i) {
        if (i == inner.size() || inner[i] == ',') {
          shapes.push_back(inner.substr(start, i - start));
          start = i + 1;
        }
      }

      return shapes;
    };

    std::string exp_name = get_name(expected);
    std::string got_name = get_name(recieved);

    if (exp_name != got_name) {
      err << "Optimizer name differs: expected '" << exp_name << "', got '" << got_name << "'";
      throw std::runtime_error(err.str());
    }

    auto exp_types = get_types(get_body(expected));
    auto got_types = get_types(get_body(recieved));

    if (exp_types.size() != got_types.size()) {
      err << "Number of parameter types differs: expected " << exp_types.size() << ", got "
          << got_types.size();
      throw std::runtime_error(err.str());
    }

    for (int i = 0; i < exp_types.size(); ++i) {
      if (exp_types[i] == got_types[i]) continue;

      std::string exp_tn = get_type_name(exp_types[i]);
      std::string got_tn = get_type_name(got_types[i]);

      if (exp_tn != got_tn) {
        err << "Parameter group " << i << " dtype differs: expected '" << exp_tn << "', got '"
            << got_tn << "'";
        throw std::runtime_error(err.str());
      }

      auto exp_shapes = get_shapes(exp_types[i]);
      auto got_shapes = get_shapes(got_types[i]);

      if (exp_shapes.size() != got_shapes.size()) {
        err << "Parameter group " << i << " (" << exp_tn << ") count differs: expected "
            << exp_shapes.size() << " parameters, got " << got_shapes.size();
        throw std::runtime_error(err.str());
      }

      for (int j = 0; j < exp_shapes.size(); ++j) {
        if (exp_shapes[j] != got_shapes[j]) {
          err << "Parameter group " << i << " (" << exp_tn << ") parameter " << j
              << " shape differs: expected " << exp_shapes[j] << ", got " << got_shapes[j];
          throw std::runtime_error(err.str());
        }
      }
    }

    throw std::runtime_error("Optimizer type mismatch: expected " + expected + ", got " + recieved);
  }
};
}  // namespace optim
}  // namespace mirage
