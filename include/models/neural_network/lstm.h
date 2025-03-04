#ifndef MODELS_NEURAL_NETWORK_LSTM_H_
#define MODELS_NEURAL_NETWORK_LSTM_H_

#include "models/neural_network/base_network.h"

namespace models::neural_network
{
  class LSTM : public BaseNetwork
  {
  public:
    LSTM();
    ~LSTM() override;
  };
}

#endif // MODELS_NEURAL_NETWORK_LSTM_H_
