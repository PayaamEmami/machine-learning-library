#ifndef MODELS_NEURAL_NETWORK_TRANSFORMER_H_
#define MODELS_NEURAL_NETWORK_TRANSFORMER_H_

#include "models/neural_network/base_network.h"

namespace models::neural_network
{
  class Transformer : public BaseNetwork
  {
  public:
    Transformer();
    ~Transformer() override;
  };
}

#endif // MODELS_NEURAL_NETWORK_TRANSFORMER_H_
