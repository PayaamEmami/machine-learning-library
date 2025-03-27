#include "models/naive_bayes/naive_bayes.h"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <numeric>

namespace ml {

NaiveBayes::NaiveBayes() : smoothing_(1.0) {}

NaiveBayes::~NaiveBayes() = default;

void NaiveBayes::fit(const std::vector<std::vector<double>>& X,
                    const std::vector<int>& y) {
  if (X.empty() || y.empty()) {
    throw std::invalid_argument("Empty training data");
  }

  // Compute class statistics
  compute_class_statistics(X, y);

  // Compute class priors
  std::map<int, int> class_counts;
  for (int label : y) {
    class_counts[label]++;
  }

  int total_samples = y.size();
  for (const auto& count : class_counts) {
    class_priors_[count.first] = static_cast<double>(count.second) / total_samples;
  }
}

std::vector<int> NaiveBayes::predict(const std::vector<std::vector<double>>& X) {
  std::vector<int> predictions;
  predictions.reserve(X.size());

  for (const auto& x : X) {
    double max_posterior = -std::numeric_limits<double>::infinity();
    int predicted_class = 0;

    for (const auto& prior : class_priors_) {
      int class_idx = prior.first;
      double posterior = std::log(prior.second) + compute_likelihood(x, class_idx);
      
      if (posterior > max_posterior) {
        max_posterior = posterior;
        predicted_class = class_idx;
      }
    }

    predictions.push_back(predicted_class);
  }

  return predictions;
}

double NaiveBayes::score(const std::vector<std::vector<double>>& X,
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

void NaiveBayes::save_model(const std::string& path) const {
  std::ofstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for saving model");
  }

  // Save smoothing parameter
  file << smoothing_ << "\n";

  // Save class priors
  file << class_priors_.size() << "\n";
  for (const auto& prior : class_priors_) {
    file << prior.first << " " << prior.second << "\n";
  }

  // Save feature means
  file << feature_means_.size() << " " << feature_means_[0].size() << "\n";
  for (const auto& means : feature_means_) {
    for (double mean : means) {
      file << mean << " ";
    }
    file << "\n";
  }

  // Save feature variances
  for (const auto& vars : feature_vars_) {
    for (double var : vars) {
      file << var << " ";
    }
    file << "\n";
  }
}

void NaiveBayes::load_model(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for loading model");
  }

  // Load smoothing parameter
  file >> smoothing_;

  // Load class priors
  size_t n_classes;
  file >> n_classes;
  class_priors_.clear();
  for (size_t i = 0; i < n_classes; ++i) {
    int class_idx;
    double prior;
    file >> class_idx >> prior;
    class_priors_[class_idx] = prior;
  }

  // Load feature means
  size_t n_classes_means, n_features;
  file >> n_classes_means >> n_features;
  feature_means_.resize(n_classes_means, std::vector<double>(n_features));
  for (auto& means : feature_means_) {
    for (double& mean : means) {
      file >> mean;
    }
  }

  // Load feature variances
  feature_vars_.resize(n_classes_means, std::vector<double>(n_features));
  for (auto& vars : feature_vars_) {
    for (double& var : vars) {
      file >> var;
    }
  }
}

void NaiveBayes::set_params(const std::vector<double>& params) {
  if (params.size() != 1) {
    throw std::invalid_argument("Naive Bayes expects exactly one parameter (smoothing)");
  }
  smoothing_ = params[0];
}

std::vector<double> NaiveBayes::get_params() const {
  return {smoothing_};
}

double NaiveBayes::compute_likelihood(const std::vector<double>& x, int class_idx) const {
  double log_likelihood = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    double mean = feature_means_[class_idx][i];
    double var = feature_vars_[class_idx][i] + smoothing_;
    
    // Gaussian likelihood
    log_likelihood -= 0.5 * std::log(2 * M_PI * var);
    log_likelihood -= 0.5 * std::pow(x[i] - mean, 2) / var;
  }
  return log_likelihood;
}

void NaiveBayes::compute_class_statistics(const std::vector<std::vector<double>>& X,
                                        const std::vector<int>& y) {
  // Find unique classes
  std::map<int, std::vector<size_t>> class_indices;
  for (size_t i = 0; i < y.size(); ++i) {
    class_indices[y[i]].push_back(i);
  }

  // Initialize feature statistics for each class
  size_t n_features = X[0].size();
  feature_means_.resize(class_indices.size(), std::vector<double>(n_features, 0.0));
  feature_vars_.resize(class_indices.size(), std::vector<double>(n_features, 0.0));

  // Compute means and variances for each class
  int class_idx = 0;
  for (const auto& indices : class_indices) {
    // Compute means
    for (size_t i = 0; i < n_features; ++i) {
      double sum = 0.0;
      for (size_t j : indices.second) {
        sum += X[j][i];
      }
      feature_means_[class_idx][i] = sum / indices.second.size();
    }

    // Compute variances
    for (size_t i = 0; i < n_features; ++i) {
      double sum_sq_diff = 0.0;
      for (size_t j : indices.second) {
        double diff = X[j][i] - feature_means_[class_idx][i];
        sum_sq_diff += diff * diff;
      }
      feature_vars_[class_idx][i] = sum_sq_diff / indices.second.size();
    }

    ++class_idx;
  }
}

}  // namespace ml
