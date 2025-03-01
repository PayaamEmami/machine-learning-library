#ifndef MODELS_ENSEMBLE_RANDOM_FOREST_H_
#define MODELS_ENSEMBLE_RANDOM_FOREST_H_

#include "models/base_model.h"

namespace models {
  namespace ensemble {

    class RandomForest : public models::BaseModel {
    public:
      RandomForest();
      ~RandomForest();
    };

  } // namespace ensemble
} // namespace models

#endif // MODELS_ENSEMBLE_RANDOM_FOREST_H_
