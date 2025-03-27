#ifndef MODELS_SVM_SVM_H_
#define MODELS_SVM_SVM_H_

#include "models/base_model.h"
#include <vector>

namespace ml {

  class SVM : public BaseModel {
  public:
    SVM();
    ~SVM() override;

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
    // SVM parameters
    double c_value_;  // Regularization parameter
    double learning_rate_;  // Learning rate for gradient descent
    int max_iterations_;  // Maximum number of iterations
    double tolerance_;  // Convergence tolerance

    // Model state
    std::vector<double> weights_;  // Weight vector
    double bias_;  // Bias term
    std::vector<std::vector<double>> support_vectors_;  // Support vectors
    std::vector<int> support_vector_labels_;  // Labels of support vectors

    // Helper methods
    double compute_kernel(const std::vector<double>& x1, const std::vector<double>& x2) const;
    double compute_decision_function(const std::vector<double>& x) const;
    void update_weights(const std::vector<std::vector<double>>& X, const std::vector<int>& y);
  };

}  // namespace ml

#endif  // MODELS_SVM_SVM_H_
