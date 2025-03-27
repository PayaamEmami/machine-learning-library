#include "models/tree/decision_tree.h"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <map>
#include <random>

namespace ml {

  DecisionTree::DecisionTree()
    : max_depth_(5),
      min_samples_split_(2),
      min_samples_leaf_(1) {
  }

  DecisionTree::~DecisionTree() = default;

  void DecisionTree::fit(const std::vector<std::vector<double>>& X,
    const std::vector<int>& y) {
    if (X.empty() || y.empty()) {
      throw std::invalid_argument("Empty training data");
    }

    root_ = build_tree(X, y, 0);
  }

  std::vector<int> DecisionTree::predict(const std::vector<std::vector<double>>& X) {
    std::vector<int> predictions;
    predictions.reserve(X.size());

    for (const auto& x : X) {
      predictions.push_back(predict_single(x));
    }

    return predictions;
  }

  double DecisionTree::score(const std::vector<std::vector<double>>& X,
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

  void DecisionTree::save_model(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file for saving model");
    }

    // Save parameters
    file << max_depth_ << " " << min_samples_split_ << " " << min_samples_leaf_ << "\n";

    // Save tree structure
    save_node(file, root_.get());
  }

  void DecisionTree::load_model(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file for loading model");
    }

    // Load parameters
    file >> max_depth_ >> min_samples_split_ >> min_samples_leaf_;

    // Load tree structure
    load_node(file, root_);
  }

  void DecisionTree::set_params(const std::vector<double>& params) {
    if (params.size() != 3) {
      throw std::invalid_argument("Decision Tree expects exactly 3 parameters (max_depth, min_samples_split, min_samples_leaf)");
    }
    max_depth_ = static_cast<int>(params[0]);
    min_samples_split_ = static_cast<int>(params[1]);
    min_samples_leaf_ = static_cast<int>(params[2]);
  }

  std::vector<double> DecisionTree::get_params() const {
    return {static_cast<double>(max_depth_),
            static_cast<double>(min_samples_split_),
            static_cast<double>(min_samples_leaf_)};
  }

  std::unique_ptr<TreeNode> DecisionTree::build_tree(const std::vector<std::vector<double>>& X,
                                                   const std::vector<int>& y,
                                                   int depth) {
    auto node = std::make_unique<TreeNode>();

    // Check stopping conditions
    if (depth >= max_depth_ || X.size() < min_samples_split_) {
      node->is_leaf = true;
      // Set leaf value to majority class
      std::map<int, int> class_counts;
      for (int label : y) {
        class_counts[label]++;
      }
      int max_count = 0;
      for (const auto& count : class_counts) {
        if (count.second > max_count) {
          max_count = count.second;
          node->value = count.first;
        }
      }
      return node;
    }

    // Find best split
    auto [feature_idx, threshold] = find_best_split(X, y);
    if (feature_idx == -1) {
      node->is_leaf = true;
      // Set leaf value to majority class
      std::map<int, int> class_counts;
      for (int label : y) {
        class_counts[label]++;
      }
      int max_count = 0;
      for (const auto& count : class_counts) {
        if (count.second > max_count) {
          max_count = count.second;
          node->value = count.first;
        }
      }
      return node;
    }

    // Split data
    std::vector<std::vector<double>> X_left, X_right;
    std::vector<int> y_left, y_right;
    for (size_t i = 0; i < X.size(); ++i) {
      if (X[i][feature_idx] <= threshold) {
        X_left.push_back(X[i]);
        y_left.push_back(y[i]);
      } else {
        X_right.push_back(X[i]);
        y_right.push_back(y[i]);
      }
    }

    // Check if split is valid
    if (X_left.size() < min_samples_leaf_ || X_right.size() < min_samples_leaf_) {
      node->is_leaf = true;
      // Set leaf value to majority class
      std::map<int, int> class_counts;
      for (int label : y) {
        class_counts[label]++;
      }
      int max_count = 0;
      for (const auto& count : class_counts) {
        if (count.second > max_count) {
          max_count = count.second;
          node->value = count.first;
        }
      }
      return node;
    }

    // Set node parameters and build children
    node->feature_index = feature_idx;
    node->threshold = threshold;
    node->left = build_tree(X_left, y_left, depth + 1);
    node->right = build_tree(X_right, y_right, depth + 1);

    return node;
  }

  std::pair<int, double> DecisionTree::find_best_split(const std::vector<std::vector<double>>& X,
                                                     const std::vector<int>& y) {
    int best_feature = -1;
    double best_threshold = 0.0;
    double best_gini = std::numeric_limits<double>::infinity();

    for (size_t feature_idx = 0; feature_idx < X[0].size(); ++feature_idx) {
      // Get unique values for this feature
      std::vector<double> unique_values;
      for (const auto& x : X) {
        unique_values.push_back(x[feature_idx]);
      }
      std::sort(unique_values.begin(), unique_values.end());
      unique_values.erase(std::unique(unique_values.begin(), unique_values.end()),
                        unique_values.end());

      // Try each threshold
      for (double threshold : unique_values) {
        std::vector<int> y_left, y_right;
        for (size_t i = 0; i < X.size(); ++i) {
          if (X[i][feature_idx] <= threshold) {
            y_left.push_back(y[i]);
          } else {
            y_right.push_back(y[i]);
          }
        }

        if (y_left.empty() || y_right.empty()) {
          continue;
        }

        // Compute weighted Gini impurity
        double gini_left = compute_gini(y_left);
        double gini_right = compute_gini(y_right);
        double weighted_gini = (y_left.size() * gini_left + y_right.size() * gini_right) / y.size();

        if (weighted_gini < best_gini) {
          best_gini = weighted_gini;
          best_feature = feature_idx;
          best_threshold = threshold;
        }
      }
    }

    return {best_feature, best_threshold};
  }

  double DecisionTree::compute_gini(const std::vector<int>& y) {
    std::map<int, int> class_counts;
    for (int label : y) {
      class_counts[label]++;
    }

    double gini = 1.0;
    for (const auto& count : class_counts) {
      double p = static_cast<double>(count.second) / y.size();
      gini -= p * p;
    }

    return gini;
  }

  int DecisionTree::predict_single(const std::vector<double>& x) const {
    const TreeNode* current = root_.get();
    
    while (!current->is_leaf) {
      if (x[current->feature_index] <= current->threshold) {
        current = current->left.get();
      } else {
        current = current->right.get();
      }
    }
    
    return current->value;
  }

  void DecisionTree::save_node(std::ofstream& file, const TreeNode* node) const {
    if (!node) {
      file << "null\n";
      return;
    }

    file << node->is_leaf << " "
         << node->feature_index << " "
         << node->threshold << " "
         << node->value << "\n";

    if (!node->is_leaf) {
      save_node(file, node->left.get());
      save_node(file, node->right.get());
    }
  }

  void DecisionTree::load_node(std::ifstream& file, std::unique_ptr<TreeNode>& node) {
    bool is_leaf;
    file >> is_leaf;

    if (is_leaf) {
      node = std::make_unique<TreeNode>();
      node->is_leaf = true;
      file >> node->feature_index >> node->threshold >> node->value;
      return;
    }

    node = std::make_unique<TreeNode>();
    file >> node->feature_index >> node->threshold >> node->value;
    node->is_leaf = false;

    load_node(file, node->left);
    load_node(file, node->right);
  }

}  // namespace ml
