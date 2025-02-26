#pragma once
#include "base_network.h"
#include "matrix.h"
#include <vector>

namespace neural_network
{
  class multi_layer_perceptron : public base_network
  {
  public:
    multi_layer_perceptron();
    ~multi_layer_perceptron();

  private:
    std::vector<matrix> weights_;
    std::vector<matrix> biases_;
  };

}