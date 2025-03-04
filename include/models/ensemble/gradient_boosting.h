#ifndef MODELS_ENSEMBLE_GRADIENT_BOOSTING_H_
#define MODELS_ENSEMBLE_GRADIENT_BOOSTING_H_

#include "models/base_model.h"

namespace models::ensemble
{
  class GradientBoosting : public BaseModel
  {
  public:
    GradientBoosting();
    ~GradientBoosting() override;
  };
}

#endif // MODELS_ENSEMBLE_GRADIENT_BOOSTING_H_
