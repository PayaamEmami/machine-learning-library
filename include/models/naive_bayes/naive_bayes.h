#ifndef MODELS_NAIVE_BAYES_NAIVE_BAYES_H_
#define MODELS_NAIVE_BAYES_NAIVE_BAYES_H_

#include "models/base_model.h"
#include <vector>
#include <map>

namespace ml {

  class NaiveBayes : public BaseModel {
  public:
    NaiveBayes();
    ~NaiveBayes() override;

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
    double smoothing_;  // Laplace smoothing parameter
    std::map<int, double> class_priors_;  // Prior probabilities for each class
    std::vector<std::vector<double>> feature_means_;  // Mean of each feature for each class
    std::vector<std::vector<double>> feature_vars_;   // Variance of each feature for each class

    // Helper methods
    double compute_likelihood(const std::vector<double>& x, int class_idx) const;
    void compute_class_statistics(const std::vector<std::vector<double>>& X,
                                const std::vector<int>& y);
  };

}  // namespace ml

#endif  // MODELS_NAIVE_BAYES_NAIVE_BAYES_H_
