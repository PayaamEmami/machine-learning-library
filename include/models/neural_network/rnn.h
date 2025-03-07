#ifndef MODELS_NEURAL_NETWORK_RNN_H_
#define MODELS_NEURAL_NETWORK_RNN_H_

#include "models/neural_network/base_network.h"
#include <vector>

namespace ml {
  namespace nn {

    class RNN : public BaseNetwork {
    public:
      RNN();
      ~RNN() override;

      // BaseModel methods
      void fit(const std::vector<std::vector<double>>& X,
        const std::vector<int>& y) override;

      std::vector<int> predict(const std::vector<std::vector<double>>& X) override;

      double score(const std::vector<std::vector<double>>& X,
        const std::vector<int>& y) override;

      void save_model(const std::string& path) const override;

      void load_model(const std::string& path) override;

      void set_params(const std::vector<double>& params) override;

      std::vector<double> get_params() const override;

      // BaseNetwork methods
      std::vector<double> forward(const std::vector<double>& input) override;
      void backward(const std::vector<double>& y_true) override;
      void update_weights() override;

    private:
      // Placeholder for RNN parameters
      int hidden_size_;
    };

  }  // namespace nn
}  // namespace ml

#endif  // MODELS_NEURAL_NETWORK_RNN_H_
