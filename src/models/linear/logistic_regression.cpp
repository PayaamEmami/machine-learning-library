#include "models/linear/logistic_regression.h"

#include <iostream>
#include <stdexcept>

namespace ml {

  LogisticRegression::LogisticRegression()
    : bias_(0.0) {
  }

  LogisticRegression::~LogisticRegression() = default;

  void LogisticRegression::fit(const std::vector<std::vector<double>>& X,
    const std::vector<int>& y) {
    std::cout << "[LogisticRegression] fit() not implemented yet.\n";
  }

  std::vector<int> LogisticRegression::predict(const std::vector<std::vector<double>>& X) {
    std::cout << "[LogisticRegression] predict() not implemented yet.\n";
    return std::vector<int>(X.size(), 0);
  }

  double LogisticRegression::score(const std::vector<std::vector<double>>& X,
    const std::vector<int>& y) {
    std::cout << "[LogisticRegression] score() not implemented yet.\n";
    return 0.0;
  }

  void LogisticRegression::save_model(const std::string& path) const {
    std::cout << "[LogisticRegression] save_model() not implemented yet.\n";
  }

  void LogisticRegression::load_model(const std::string& path) {
    std::cout << "[LogisticRegression] load_model() not implemented yet.\n";
  }

  void LogisticRegression::set_params(const std::vector<double>& params) {
    if (!params.empty()) {
      weights_ = params;
      bias_ = weights_.back();
      weights_.pop_back();
    }
  }

  std::vector<double> LogisticRegression::get_params() const {
    std::vector<double> params = weights_;
    params.push_back(bias_);
    return params;
  }

}  // namespace ml
