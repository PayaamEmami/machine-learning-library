#include "models/tree/decision_tree.h"

#include <iostream>
#include <stdexcept>

namespace ml {

  DecisionTree::DecisionTree()
    : max_depth_(10) {
  }

  DecisionTree::~DecisionTree() = default;

  void DecisionTree::fit(const std::vector<std::vector<double>>& X,
    const std::vector<int>& y) {
    std::cout << "[DecisionTree] fit() not implemented yet.\n";
  }

  std::vector<int> DecisionTree::predict(const std::vector<std::vector<double>>& X) {
    std::cout << "[DecisionTree] predict() not implemented yet.\n";
    return std::vector<int>(X.size(), 0);
  }

  double DecisionTree::score(const std::vector<std::vector<double>>& X,
    const std::vector<int>& y) {
    std::cout << "[DecisionTree] score() not implemented yet.\n";
    return 0.0;
  }

  void DecisionTree::save_model(const std::string& path) const {
    std::cout << "[DecisionTree] save_model() not implemented yet.\n";
  }

  void DecisionTree::load_model(const std::string& path) {
    std::cout << "[DecisionTree] load_model() not implemented yet.\n";
  }

  void DecisionTree::set_params(const std::vector<double>& params) {
    if (!params.empty()) {
      max_depth_ = static_cast<int>(params[0]);
    }
  }

  std::vector<double> DecisionTree::get_params() const {
    return { static_cast<double>(max_depth_) };
  }

}  // namespace ml
