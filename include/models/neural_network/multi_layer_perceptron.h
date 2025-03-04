#ifndef MODELS_NEURAL_NETWORK_MULTI_LAYER_PERCEPTRON_H_
#define MODELS_NEURAL_NETWORK_MULTI_LAYER_PERCEPTRON_H_

#include "models/neural_network/base_network.h"

namespace models::neural_network
{
  class MultiLayerPerceptron : public BaseNetwork
  {
  public:
    MultiLayerPerceptron();
    ~MultiLayerPerceptron() override;
  };
}

#endif // MODELS_NEURAL_NETWORK_MULTI_LAYER_PERCEPTRON_H_
