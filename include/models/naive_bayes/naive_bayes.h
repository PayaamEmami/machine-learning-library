#ifndef MODELS_NAIVE_BAYES_NAIVE_BAYES_H_
#define MODELS_NAIVE_BAYES_NAIVE_BAYES_H_

#include "models/base_model.h"

namespace models {
  namespace naive_bayes {

    class NaiveBayes : public models::BaseModel {
    public:
      NaiveBayes();
      ~NaiveBayes();
    };

  } // namespace naive_bayes
} // namespace models

#endif // MODELS_NAIVE_BAYES_NAIVE_BAYES_H_
