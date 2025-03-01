#ifndef MODELS_NEURAL_NETWORK_BASE_NETWORK_H_
#define MODELS_NEURAL_NETWORK_BASE_NETWORK_H_

#include "models/base_model.h"

namespace models {
  namespace neural_network {

    class BaseNetwork : public models::BaseModel {
    public:
      BaseNetwork();
      virtual ~BaseNetwork();
    };

  } // namespace neural_network
} // namespace models

#endif // MODELS_NEURAL_NETWORK_BASE_NETWORK_H_
