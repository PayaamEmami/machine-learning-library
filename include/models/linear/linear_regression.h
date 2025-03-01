#ifndef MODELS_LINEAR_LINEAR_REGRESSION_H_
#define MODELS_LINEAR_LINEAR_REGRESSION_H_

#include "models/base_model.h"

namespace models {
  namespace linear {

    class LinearRegression : public models::BaseModel {
    public:
      LinearRegression();
      ~LinearRegression();
    };

  } // namespace linear
} // namespace models

#endif // MODELS_LINEAR_LINEAR_REGRESSION_H_
