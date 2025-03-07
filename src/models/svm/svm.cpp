#include "models/svm/svm.h"

#include <iostream>
#include <stdexcept>

namespace ml {

  SVM::SVM()
    : c_value_(1.0) {
  }

  SVM::~SVM() = default;

  void SVM::fit(const std::vector<std::vector<double>>& X,
    const std::vector<int>& y) {
    std::cout << "[SVM] fit() not implemented yet.\n";
  }

  std::vector<int> SVM::predict(const std::vector<std::vector<double>>& X) {
    std::cout << "[SVM] predict() not implemented yet.\n";
    return std::vector<int>(X.size(), 0);
  }

  double SVM::score(const std::vector<std::vector<double>>& X,
    const std::vector<int>& y) {
    std::cout << "[SVM] score() not implemented yet.\n";
    return 0.0;
  }

  void SVM::save_model(const std::string& path) const {
    std::cout << "[SVM] save_model() not implemented yet.\n";
  }

  void SVM::load_model(const std::string& path) {
    std::cout << "[SVM] load_model() not implemented yet.\n";
  }

  void SVM::set_params(const std::vector<double>& params) {
    if (!params.empty()) {
      c_value_ = params[0];
    }
  }

  std::vector<double> SVM::get_params() const {
    return { c_value_ };
  }

}  // namespace ml
