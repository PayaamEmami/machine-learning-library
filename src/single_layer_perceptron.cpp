#include "single_layer_perceptron.h"
#include <iostream>

namespace neural_network
{
  single_layer_perceptron::single_layer_perceptron(int input_size, int output_size) :
    weights(input_size, output_size), biases(1, output_size)
  {
    std::cout << "SLP constructed!\n";
  }

  single_layer_perceptron::~single_layer_perceptron()
  {
    std::cout << "SLP destroyed!\n";
  }
}
