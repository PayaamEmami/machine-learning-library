#include "models/knn/knn.h"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>

namespace ml {

  KNN::KNN()
    : k_(5) {  // Default K
  }

  KNN::~KNN() = default;

  void KNN::fit(const std::vector<std::vector<double>>& X,
    const std::vector<int>& y) {
    // KNN is a lazy learner, so we just store the training data
    X_train_ = X;
    y_train_ = y;
  }

  std::vector<int> KNN::predict(const std::vector<std::vector<double>>& X) {
    std::vector<int> predictions;
    predictions.reserve(X.size());

    for (const auto& x : X) {
      // Calculate distances to all training points
      std::vector<std::pair<double, int>> distances;
      for (size_t i = 0; i < X_train_.size(); ++i) {
        double dist = 0.0;
        for (size_t j = 0; j < x.size(); ++j) {
          double diff = x[j] - X_train_[i][j];
          dist += diff * diff;
        }
        distances.emplace_back(std::sqrt(dist), i);
      }

      // Sort by distance and get k nearest neighbors
      std::sort(distances.begin(), distances.end());
      
      // Count class frequencies among k nearest neighbors
      std::vector<int> class_counts(10, 0);  // Assuming max 10 classes
      for (int i = 0; i < k_; ++i) {
        class_counts[y_train_[distances[i].second]]++;
      }

      // Find most frequent class
      int max_count = 0;
      int predicted_class = 0;
      for (int i = 0; i < 10; ++i) {
        if (class_counts[i] > max_count) {
          max_count = class_counts[i];
          predicted_class = i;
        }
      }

      predictions.push_back(predicted_class);
    }

    return predictions;
  }

  double KNN::score(const std::vector<std::vector<double>>& X,
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

  void KNN::save_model(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file for saving model");
    }

    // Save k parameter
    file << k_ << "\n";

    // Save training data
    file << X_train_.size() << " " << X_train_[0].size() << "\n";
    for (const auto& x : X_train_) {
      for (double val : x) {
        file << val << " ";
      }
      file << "\n";
    }

    // Save labels
    for (int label : y_train_) {
      file << label << " ";
    }
    file << "\n";
  }

  void KNN::load_model(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file for loading model");
    }

    // Load k parameter
    file >> k_;

    // Load training data dimensions
    size_t n_samples, n_features;
    file >> n_samples >> n_features;

    // Load training data
    X_train_.resize(n_samples, std::vector<double>(n_features));
    for (auto& x : X_train_) {
      for (double& val : x) {
        file >> val;
      }
    }

    // Load labels
    y_train_.resize(n_samples);
    for (int& label : y_train_) {
      file >> label;
    }
  }

  void KNN::set_params(const std::vector<double>& params) {
    if (params.size() != 1) {
      throw std::invalid_argument("KNN expects exactly one parameter (k)");
    }
    k_ = static_cast<int>(params[0]);
  }

  std::vector<double> KNN::get_params() const {
    return {static_cast<double>(k_)};
  }

}  // namespace ml
