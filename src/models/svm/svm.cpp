#include "models/svm/svm.h"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <random>

namespace ml {

  SVM::SVM()
    : c_value_(1.0),
      learning_rate_(0.01),
      max_iterations_(1000),
      tolerance_(1e-6),
      bias_(0.0) {
  }

  SVM::~SVM() = default;

  void SVM::fit(const std::vector<std::vector<double>>& X,
    const std::vector<int>& y) {
    if (X.empty() || y.empty()) {
      throw std::invalid_argument("Empty training data");
    }

    // Initialize weights with zeros
    weights_.resize(X[0].size(), 0.0);

    // Training loop
    for (int iter = 0; iter < max_iterations_; ++iter) {
      double prev_norm = 0.0;
      for (const auto& w : weights_) {
        prev_norm += w * w;
      }

      update_weights(X, y);

      // Check convergence
      double curr_norm = 0.0;
      for (const auto& w : weights_) {
        curr_norm += w * w;
      }
      if (std::abs(curr_norm - prev_norm) < tolerance_) {
        break;
      }
    }

    // Store support vectors
    support_vectors_.clear();
    support_vector_labels_.clear();
    for (size_t i = 0; i < X.size(); ++i) {
      double decision = compute_decision_function(X[i]);
      if (std::abs(decision) < 1.0) {  // Points on or within margin
        support_vectors_.push_back(X[i]);
        support_vector_labels_.push_back(y[i]);
      }
    }
  }

  std::vector<int> SVM::predict(const std::vector<std::vector<double>>& X) {
    std::vector<int> predictions;
    predictions.reserve(X.size());

    for (const auto& x : X) {
      double decision = compute_decision_function(x);
      predictions.push_back(decision >= 0 ? 1 : -1);
    }

    return predictions;
  }

  double SVM::score(const std::vector<std::vector<double>>& X,
    const std::vector<int>& y) {
    auto predictions = predict(X);
    int correct = 0;
    
    for (size_t i = 0; i < y.size(); ++i) {
      if (predictions[i] == y[i]) {
        correct++;
      }
    }
    
    return static_cast<double>(correct) / y.size();
  }

  void SVM::save_model(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file for saving model");
    }

    // Save parameters
    file << c_value_ << " " << learning_rate_ << " " << max_iterations_ << " "
         << tolerance_ << " " << bias_ << "\n";

    // Save weights
    file << weights_.size() << "\n";
    for (double w : weights_) {
      file << w << " ";
    }
    file << "\n";

    // Save support vectors
    file << support_vectors_.size() << " " << support_vectors_[0].size() << "\n";
    for (const auto& sv : support_vectors_) {
      for (double val : sv) {
        file << val << " ";
      }
      file << "\n";
    }

    // Save support vector labels
    for (int label : support_vector_labels_) {
      file << label << " ";
    }
    file << "\n";
  }

  void SVM::load_model(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file for loading model");
    }

    // Load parameters
    file >> c_value_ >> learning_rate_ >> max_iterations_ >> tolerance_ >> bias_;

    // Load weights
    size_t n_weights;
    file >> n_weights;
    weights_.resize(n_weights);
    for (double& w : weights_) {
      file >> w;
    }

    // Load support vectors
    size_t n_sv, n_features;
    file >> n_sv >> n_features;
    support_vectors_.resize(n_sv, std::vector<double>(n_features));
    for (auto& sv : support_vectors_) {
      for (double& val : sv) {
        file >> val;
      }
    }

    // Load support vector labels
    support_vector_labels_.resize(n_sv);
    for (int& label : support_vector_labels_) {
      file >> label;
    }
  }

  void SVM::set_params(const std::vector<double>& params) {
    if (params.size() != 4) {
      throw std::invalid_argument("SVM expects exactly 4 parameters (C, learning_rate, max_iterations, tolerance)");
    }
    c_value_ = params[0];
    learning_rate_ = params[1];
    max_iterations_ = static_cast<int>(params[2]);
    tolerance_ = params[3];
  }

  std::vector<double> SVM::get_params() const {
    return {c_value_, learning_rate_, static_cast<double>(max_iterations_), tolerance_};
  }

  double SVM::compute_kernel(const std::vector<double>& x1, const std::vector<double>& x2) const {
    // Linear kernel implementation
    double result = 0.0;
    for (size_t i = 0; i < x1.size(); ++i) {
      result += x1[i] * x2[i];
    }
    return result;
  }

  double SVM::compute_decision_function(const std::vector<double>& x) const {
    double sum = 0.0;
    for (size_t i = 0; i < support_vectors_.size(); ++i) {
      sum += support_vector_labels_[i] * compute_kernel(support_vectors_[i], x);
    }
    return sum + bias_;
  }

  void SVM::update_weights(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
    // Stochastic gradient descent update
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dis(0, X.size() - 1);
    
    size_t idx = dis(gen);
    const auto& x = X[idx];
    int label = y[idx];

    double decision = compute_decision_function(x);
    if (label * decision < 1.0) {  // Hinge loss
      // Update weights
      for (size_t i = 0; i < weights_.size(); ++i) {
        weights_[i] += learning_rate_ * (label * x[i] - c_value_ * weights_[i]);
      }
      // Update bias
      bias_ += learning_rate_ * label;
    }
  }

}  // namespace ml
