#pragma once
#include "base_network.h"
#include "matrix.h"

namespace neural_network
{
  class single_layer_perceptron : public base_network
  {
  public:
    single_layer_perceptron(int input_size, int output_size);
    ~single_layer_perceptron();

  private:
    matrix weights;
    matrix biases;
  };
}
