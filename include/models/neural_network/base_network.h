#ifndef MODELS_NEURAL_NETWORK_BASE_NETWORK_H_
#define MODELS_NEURAL_NETWORK_BASE_NETWORK_H_

#include "models/base_model.h"
#include <vector>

namespace ml {
  namespace nn {

    // BaseNetwork extends BaseModel with additional methods needed
    // for neural networks (forward, backward, update_weights).
    class BaseNetwork : public ml::BaseModel {
    public:
      virtual ~BaseNetwork() = default;

      // Forward pass for a single sample (or a mini-batch).
      virtual std::vector<double> forward(const std::vector<double>& input) = 0;

      // Backward pass to compute gradients based on the true labels.
      virtual void backward(const std::vector<double>& y_true) = 0;

      // Update weights after computing gradients.
      virtual void update_weights() = 0;
    };

  }  // namespace nn
}  // namespace ml

#endif  // MODELS_NEURAL_NETWORK_BASE_NETWORK_H_
