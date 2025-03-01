#ifndef MODELS_KNN_KNN_H_
#define MODELS_KNN_KNN_H_

#include "models/base_model.h"

namespace models {
  namespace knn {

    class KNN : public models::BaseModel {
    public:
      KNN();
      ~KNN();
    };

  } // namespace knn
} // namespace models

#endif // MODELS_KNN_KNN_H_
