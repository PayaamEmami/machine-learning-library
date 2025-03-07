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
    double c_value_;  // Example regularization parameter
  };

}  // namespace ml

#endif  // MODELS_SVM_SVM_H_
