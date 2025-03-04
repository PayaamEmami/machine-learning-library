#ifndef MODELS_NAIVE_BAYES_NAIVE_BAYES_H_
#define MODELS_NAIVE_BAYES_NAIVE_BAYES_H_

#include "models/base_model.h"

namespace models::naive_bayes
{
  class NaiveBayes : public BaseModel
  {
  public:
    NaiveBayes();
    ~NaiveBayes() override;
  };
}

#endif // MODELS_NAIVE_BAYES_NAIVE_BAYES_H_
