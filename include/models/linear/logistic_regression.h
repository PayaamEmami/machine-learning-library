#ifndef MODELS_LINEAR_LOGISTIC_REGRESSION_H_
#define MODELS_LINEAR_LOGISTIC_REGRESSION_H_

#include "models/base_model.h"

namespace models {
  namespace linear {

    class LogisticRegression : public models::BaseModel {
    public:
      LogisticRegression();
      ~LogisticRegression();
    };

  } // namespace linear
} // namespace models

#endif // MODELS_LINEAR_LOGISTIC_REGRESSION_H_
