#include "models/knn/knn.h"

#include <iostream>
#include <stdexcept>

namespace ml {

  KNN::KNN()
    : k_(3) {  // Default K
  }

  KNN::~KNN() = default;

  void KNN::fit(const std::vector<std::vector<double>>& X,
    const std::vector<int>& y) {
    // TODO: Implement KNN training (often just storing X and y).
    std::cout << "[KNN] fit() not implemented yet.\n";
  }

  std::vector<int> KNN::predict(const std::vector<std::vector<double>>& X) {
    // TODO: Implement KNN prediction logic.
    std::cout << "[KNN] predict() not implemented yet.\n";
    return std::vector<int>(X.size(), 0);
  }

  double KNN::score(const std::vector<std::vector<double>>& X,
    const std::vector<int>& y) {
    // TODO: Implement an accuracy calculation for KNN.
    std::cout << "[KNN] score() not implemented yet.\n";
    return 0.0;
  }

  void KNN::save_model(const std::string& path) const {
    // TODO: Save model state to file.
    std::cout << "[KNN] save_model() not implemented yet.\n";
  }

  void KNN::load_model(const std::string& path) {
    // TODO: Load model state from file.
    std::cout << "[KNN] load_model() not implemented yet.\n";
  }

  void KNN::set_params(const std::vector<double>& params) {
    if (!params.empty()) {
      // Assuming the first parameter is 'k'
      k_ = static_cast<int>(params[0]);
    }
  }

  std::vector<double> KNN::get_params() const {
    // Return current parameters (here just 'k')
    return { static_cast<double>(k_) };
  }

}  // namespace ml
