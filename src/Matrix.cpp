#include "matrix.h"
#include <iostream>

namespace neural_network
{
  matrix::matrix()
  {
    std::cout << "matrix constructed!\n";
  }

  matrix::matrix(int input_size, int output_size)
  {
  }

  matrix::~matrix()
  {
    std::cout << "matrix destroyed!\n";
  }
}
