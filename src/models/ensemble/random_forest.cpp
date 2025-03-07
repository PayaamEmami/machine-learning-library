#include "models/ensemble/random_forest.h"

#include <iostream>
#include <stdexcept>

namespace ml {

  RandomForest::RandomForest()
    : n_estimators_(100) {
  }

  RandomForest::~RandomForest() = default;

  void RandomForest::fit(const std::vector<std::vector<double>>& X,
    const std::vector<int>& y) {
    std::cout << "[RandomForest] fit() not implemented yet.\n";
  }

  std::vector<int> RandomForest::predict(const std::vector<std::vector<double>>& X) {
    std::cout << "[RandomForest] predict() not implemented yet.\n";
    return std::vector<int>(X.size(), 0);
  }

  double RandomForest::score(const std::vector<std::vector<double>>& X,
    const std::vector<int>& y) {
    std::cout << "[RandomForest] score() not implemented yet.\n";
    return 0.0;
  }

  void RandomForest::save_model(const std::string& path) const {
    std::cout << "[RandomForest] save_model() not implemented yet.\n";
  }

  void RandomForest::load_model(const std::string& path) {
    std::cout << "[RandomForest] load_model() not implemented yet.\n";
  }

  void RandomForest::set_params(const std::vector<double>& params) {
    if (!params.empty()) {
      n_estimators_ = static_cast<int>(params[0]);
    }
  }

  std::vector<double> RandomForest::get_params() const {
    return { static_cast<double>(n_estimators_) };
  }

}  // namespace ml
