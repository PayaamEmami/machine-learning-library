#include "single_layer_perceptron.h"
#include <iostream>

namespace neural_network
{
  single_layer_perceptron::single_layer_perceptron(int input_size, int output_size) :
    weights_(input_size, output_size), biases_(1, output_size)
  {
    std::cout << "SLP constructed!\n";
  }

  single_layer_perceptron::~single_layer_perceptron()
  {
    std::cout << "SLP destroyed!\n";
  }
}
