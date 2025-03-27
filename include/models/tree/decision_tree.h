#ifndef MODELS_TREE_DECISION_TREE_H_
#define MODELS_TREE_DECISION_TREE_H_

#include "models/base_model.h"
#include <vector>
#include <memory>

namespace ml {

  struct TreeNode {
    int feature_index;  // Index of feature to split on
    double threshold;   // Threshold value for the split
    int value;         // Class value (for leaf nodes)
    bool is_leaf;      // Whether this is a leaf node
    std::unique_ptr<TreeNode> left;   // Left child
    std::unique_ptr<TreeNode> right;  // Right child

    TreeNode() : feature_index(-1), threshold(0.0), value(0), is_leaf(false) {}
  };

  class DecisionTree : public BaseModel {
  public:
    DecisionTree();
    ~DecisionTree() override;

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
    int max_depth_;  // Maximum depth of the tree
    int min_samples_split_;  // Minimum samples required to split a node
    int min_samples_leaf_;   // Minimum samples required in a leaf node

    // Model state
    std::unique_ptr<TreeNode> root_;  // Root node of the tree

    // Helper methods
    std::unique_ptr<TreeNode> build_tree(const std::vector<std::vector<double>>& X,
                                       const std::vector<int>& y,
                                       int depth);
    std::pair<int, double> find_best_split(const std::vector<std::vector<double>>& X,
                                         const std::vector<int>& y);
    double compute_gini(const std::vector<int>& y);
    int predict_single(const std::vector<double>& x) const;
    void save_node(std::ofstream& file, const TreeNode* node) const;
    void load_node(std::ifstream& file, std::unique_ptr<TreeNode>& node);
  };

}  // namespace ml

#endif  // MODELS_TREE_DECISION_TREE_H_
