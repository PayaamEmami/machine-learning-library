#ifndef MODELS_KNN_KNN_H_
#define MODELS_KNN_KNN_H_

#include "models/base_model.h"

namespace models::knn
{
  class KNN : public BaseModel
  {
  public:
    KNN();
    ~KNN() override;
  };
}

#endif // MODELS_KNN_KNN_H_
