#ifndef MODELS_BASE_MODEL_H_
#define MODELS_BASE_MODEL_H_

#include <vector>
#include <string>

namespace ml {

  // BaseModel is an abstract class that defines the core interface
  // for machine learning models.
  class BaseModel {
  public:
    virtual ~BaseModel() = default;

    // Train the model with features X and labels y.
    virtual void fit(const std::vector<std::vector<double>>& X,
      const std::vector<int>& y) = 0;

    // Predict labels for given features X.
    virtual std::vector<int> predict(const std::vector<std::vector<double>>& X) = 0;

    // Evaluate model performance on test data (e.g., accuracy).
    virtual double score(const std::vector<std::vector<double>>& X,
      const std::vector<int>& y) = 0;

    // Save the model to the specified path.
    virtual void save_model(const std::string& path) const = 0;

    // Load the model from the specified path.
    virtual void load_model(const std::string& path) = 0;

    // Set model parameters.
    virtual void set_params(const std::vector<double>& params) = 0;

    // Get current model parameters.
    virtual std::vector<double> get_params() const = 0;
  };

}  // namespace ml

#endif  // MODELS_BASE_MODEL_H_
