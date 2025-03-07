#ifndef MODELS_ENSEMBLE_GRADIENT_BOOSTING_H_
#define MODELS_ENSEMBLE_GRADIENT_BOOSTING_H_

#include "models/base_model.h"
#include <vector>

namespace ml {

  class GradientBoosting : public BaseModel {
  public:
    GradientBoosting();
    ~GradientBoosting() override;

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
    int n_estimators_;
    double learning_rate_;
  };

}  // namespace ml

#endif  // MODELS_ENSEMBLE_GRADIENT_BOOSTING_H_
