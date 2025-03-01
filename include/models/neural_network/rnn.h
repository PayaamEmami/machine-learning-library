#ifndef MODELS_NEURAL_NETWORK_RNN_H_
#define MODELS_NEURAL_NETWORK_RNN_H_

#include "models/neural_network/base_network.h"

namespace models {
  namespace neural_network {

    class RNN : public BaseNetwork {
    public:
      RNN();
      ~RNN();
    };

  } // namespace neural_network
} // namespace models

#endif // MODELS_NEURAL_NETWORK_RNN_H_
