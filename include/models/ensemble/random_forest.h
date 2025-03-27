#ifndef MODELS_ENSEMBLE_RANDOM_FOREST_H_
#define MODELS_ENSEMBLE_RANDOM_FOREST_H_

#include "models/base_model.h"
#include "models/tree/decision_tree.h"
#include <vector>
#include <memory>
#include <random>

namespace ml {

  class RandomForest : public BaseModel {
  public:
    RandomForest();
    ~RandomForest() override;

    void fit(const std::vector<std::vector<double>>& X,
      const std::vector<int>& y) override;

    std::vector<int> predict(const std::vector<std::vector<double>>& X) override;

    double score(const std::vector<std::vector<double>>& X,
      const std::vector<int>& y) override;

    void save_model(const std::string& path) const override;

    void load_model(const std::string& path) override;

    void set_params(const std::vector<double>& params) override;

    std::vector<double> get_params() const override;

  private:
    // Model parameters
    int n_estimators_;  // Number of trees in the forest
    int max_features_;  // Number of features to consider for each split
    int max_depth_;     // Maximum depth of each tree
    int min_samples_split_;  // Minimum samples required to split a node
    int min_samples_leaf_;   // Minimum samples required in a leaf node

    // Model state
    std::vector<std::unique_ptr<DecisionTree>> trees_;  // Collection of decision trees
    std::vector<std::vector<size_t>> feature_indices_;  // Feature indices used by each tree

    // Helper methods
    std::vector<size_t> bootstrap_indices(size_t n_samples);
    std::vector<size_t> random_feature_subset(size_t n_features);
    std::vector<int> predict_single(const std::vector<double>& x) const;
  };

}  // namespace ml

#endif  // MODELS_ENSEMBLE_RANDOM_FOREST_H_
