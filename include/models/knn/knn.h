#ifndef MODELS_KNN_KNN_H_
#define MODELS_KNN_KNN_H_

#include "models/base_model.h"
#include <vector>

namespace ml {

  class KNN : public BaseModel {
  public:
    KNN();
    ~KNN() override;

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
    int k_;  // Number of nearest neighbors
    int max_class_;  // Maximum class label in training data
    std::vector<std::vector<double>> X_train_;  // Training features
    std::vector<int> y_train_;  // Training labels
  };

}  // namespace ml

#endif  // MODELS_KNN_KNN_H_
