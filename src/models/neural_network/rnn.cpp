#include "models/neural_network/rnn.h"

#include <iostream>
#include <stdexcept>

namespace ml {
  namespace nn {

    RNN::RNN()
      : hidden_size_(32) {
    }

    RNN::~RNN() = default;

    void RNN::fit(const std::vector<std::vector<double>>& X,
      const std::vector<int>& y) {
      std::cout << "[RNN] fit() not implemented yet.\n";
    }

    std::vector<int> RNN::predict(const std::vector<std::vector<double>>& X) {
      std::cout << "[RNN] predict() not implemented yet.\n";
      return std::vector<int>(X.size(), 0);
    }

    double RNN::score(const std::vector<std::vector<double>>& X,
      const std::vector<int>& y) {
      std::cout << "[RNN] score() not implemented yet.\n";
      return 0.0;
    }

    void RNN::save_model(const std::string& path) const {
      std::cout << "[RNN] save_model() not implemented yet.\n";
    }

    void RNN::load_model(const std::string& path) {
      std::cout << "[RNN] load_model() not implemented yet.\n";
    }

    void RNN::set_params(const std::vector<double>& params) {
      if (!params.empty()) {
        hidden_size_ = static_cast<int>(params[0]);
      }
    }

    std::vector<double> RNN::get_params() const {
      return { static_cast<double>(hidden_size_) };
    }

    std::vector<double> RNN::forward(const std::vector<double>& input) {
      std::cout << "[RNN] forward() not implemented yet.\n";
      return std::vector<double>(hidden_size_, 0.0);
    }

    void RNN::backward(const std::vector<double>& y_true) {
      std::cout << "[RNN] backward() not implemented yet.\n";
    }

    void RNN::update_weights() {
      std::cout << "[RNN] update_weights() not implemented yet.\n";
    }

  }  // namespace nn
}  // namespace ml
