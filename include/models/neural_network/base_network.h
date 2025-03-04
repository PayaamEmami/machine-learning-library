#ifndef MODELS_NEURAL_NETWORK_BASE_NETWORK_H_
#define MODELS_NEURAL_NETWORK_BASE_NETWORK_H_

#include "models/base_model.h"

namespace models::neural_network
{
  class BaseNetwork : public BaseModel
  {
  public:
    BaseNetwork();
    ~BaseNetwork() override;
  };
}

#endif // MODELS_NEURAL_NETWORK_BASE_NETWORK_H_
