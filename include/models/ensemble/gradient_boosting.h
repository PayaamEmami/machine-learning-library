#ifndef MODELS_ENSEMBLE_GRADIENT_BOOSTING_H_
#define MODELS_ENSEMBLE_GRADIENT_BOOSTING_H_

#include "models/base_model.h"

namespace models {
  namespace ensemble {

    class GradientBoosting : public models::BaseModel {
    public:
      GradientBoosting();
      ~GradientBoosting();
    };

  } // namespace ensemble
} // namespace models

#endif // MODELS_ENSEMBLE_GRADIENT_BOOSTING_H_
