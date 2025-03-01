#ifndef MODELS_NEURAL_NETWORK_SINGLE_LAYER_PERCEPTRON_H_
#define MODELS_NEURAL_NETWORK_SINGLE_LAYER_PERCEPTRON_H_

#include "models/neural_network/base_network.h"

namespace models {
  namespace neural_network {

    class SingleLayerPerceptron : public BaseNetwork {
    public:
      SingleLayerPerceptron();
      ~SingleLayerPerceptron();
    };

  } // namespace neural_network
} // namespace models

#endif // MODELS_NEURAL_NETWORK_SINGLE_LAYER_PERCEPTRON_H_
