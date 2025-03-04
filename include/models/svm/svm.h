#ifndef MODELS_SVM_SVM_H_
#define MODELS_SVM_SVM_H_

#include "models/base_model.h"

namespace models::svm
{
  class SVM : public BaseModel
  {
  public:
    SVM();
    ~SVM() override;
  };
}

#endif // MODELS_SVM_SVM_H_
