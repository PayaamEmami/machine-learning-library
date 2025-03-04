#ifndef MODELS_ENSEMBLE_RANDOM_FOREST_H_
#define MODELS_ENSEMBLE_RANDOM_FOREST_H_

#include "models/base_model.h"

namespace models::ensemble
{
  class RandomForest : public BaseModel
  {
  public:
    RandomForest();
    ~RandomForest() override;
  };
}

#endif // MODELS_ENSEMBLE_RANDOM_FOREST_H_
