#ifndef MODELS_LINEAR_LOGISTIC_REGRESSION_H_
#define MODELS_LINEAR_LOGISTIC_REGRESSION_H_

#include "models/base_model.h"

namespace models::linear
{
  class LogisticRegression : public BaseModel
  {
  public:
    LogisticRegression();
    ~LogisticRegression() override;
  };
}

#endif // MODELS_LINEAR_LOGISTIC_REGRESSION_H_
