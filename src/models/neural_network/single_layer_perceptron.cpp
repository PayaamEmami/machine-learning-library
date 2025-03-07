#include "models/neural_network/single_layer_perceptron.h"

#include <iostream>
#include <stdexcept>

namespace ml {
  namespace nn {

    SingleLayerPerceptron::SingleLayerPerceptron()
      : bias_(0.0) {
    }

    SingleLayerPerceptron::~SingleLayerPerceptron() = default;

    void SingleLayerPerceptron::fit(const std::vector<std::vector<double>>& X,
      const std::vector<int>& y) {
      std::cout << "[SingleLayerPerceptron] fit() not implemented yet.\n";
    }

    std::vector<int> SingleLayerPerceptron::predict(const std::vector<std::vector<double>>& X) {
      std::cout << "[SingleLayerPerceptron] predict() not implemented yet.\n";
      return std::vector<int>(X.size(), 0);
    }

    double SingleLayerPerceptron::score(const std::vector<std::vector<double>>& X,
      const std::vector<int>& y) {
      std::cout << "[SingleLayerPerceptron] score() not implemented yet.\n";
      return 0.0;
    }

    void SingleLayerPerceptron::save_model(const std::string& path) const {
      std::cout << "[SingleLayerPerceptron] save_model() not implemented yet.\n";
    }

    void SingleLayerPerceptron::load_model(const std::string& path) {
      std::cout << "[SingleLayerPerceptron] load_model() not implemented yet.\n";
    }

    void SingleLayerPerceptron::set_params(const std::vector<double>& params) {
      if (!params.empty()) {
        weights_ = params;
        bias_ = weights_.back();
        weights_.pop_back();
      }
    }

    std::vector<double> SingleLayerPerceptron::get_params() const {
      std::vector<double> params = weights_;
      params.push_back(bias_);
      return params;
    }

    std::vector<double> SingleLayerPerceptron::forward(const std::vector<double>& input) {
      std::cout << "[SingleLayerPerceptron] forward() not implemented yet.\n";
      return std::vector<double>(input.size(), 0.0);
    }

    void SingleLayerPerceptron::backward(const std::vector<double>& y_true) {
      std::cout << "[SingleLayerPerceptron] backward() not implemented yet.\n";
    }

    void SingleLayerPerceptron::update_weights() {
      std::cout << "[SingleLayerPerceptron] update_weights() not implemented yet.\n";
    }

  }  // namespace nn
}  // namespace ml
