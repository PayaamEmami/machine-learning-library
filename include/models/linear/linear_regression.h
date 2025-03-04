#ifndef MODELS_LINEAR_LINEAR_REGRESSION_H_
#define MODELS_LINEAR_LINEAR_REGRESSION_H_

#include "models/base_model.h"

namespace models::linear
{
  class LinearRegression : public BaseModel
  {
  public:
    LinearRegression();
    ~LinearRegression() override;
  };
}

#endif // MODELS_LINEAR_LINEAR_REGRESSION_H_
