#include "multi_layer_perceptron.h"
#include <iostream>

namespace neural_network
{
  multi_layer_perceptron::multi_layer_perceptron()
  {
    std::cout << "MLP constructed!\n";
  }

  multi_layer_perceptron::~multi_layer_perceptron()
  {
    std::cout << "MLP destroyed!\n";
  }
}
