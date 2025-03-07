#include "models/linear/linear_regression.h"

#include <iostream>
#include <stdexcept>

namespace ml {

  LinearRegression::LinearRegression()
    : bias_(0.0) {
    // Default constructor
  }

  LinearRegression::~LinearRegression() = default;

  void LinearRegression::fit(const std::vector<std::vector<double>>& X,
    const std::vector<int>& y) {
    std::cout << "[LinearRegression] fit() not implemented yet.\n";
  }

  std::vector<int> LinearRegression::predict(const std::vector<std::vector<double>>& X) {
    std::cout << "[LinearRegression] predict() not implemented yet.\n";
    // Return a vector of 0s as placeholder
    return std::vector<int>(X.size(), 0);
  }

  double LinearRegression::score(const std::vector<std::vector<double>>& X,
    const std::vector<int>& y) {
    std::cout << "[LinearRegression] score() not implemented yet.\n";
    // Return 0.0 as placeholder
    return 0.0;
  }

  void LinearRegression::save_model(const std::string& path) const {
    std::cout << "[LinearRegression] save_model() not implemented yet.\n";
  }

  void LinearRegression::load_model(const std::string& path) {
    std::cout << "[LinearRegression] load_model() not implemented yet.\n";
  }

  void LinearRegression::set_params(const std::vector<double>& params) {
    // Example param usage:
    // Suppose we interpret the vector as [w1, w2, ..., wN, bias].
    // This is just a placeholder for demonstration.
    if (!params.empty()) {
      weights_ = params;
      // Remove last element as bias if that's the convention
      bias_ = weights_.back();
      weights_.pop_back();
    }
  }

  std::vector<double> LinearRegression::get_params() const {
    // Reconstruct parameter vector as [weights..., bias]
    std::vector<double> params = weights_;
    params.push_back(bias_);
    return params;
  }

}  // namespace ml
