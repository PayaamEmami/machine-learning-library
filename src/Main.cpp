#include <iostream>
#include "matrix.h"

int main()
{
  std::cout << "Hello world\n";

  neural_network::matrix matrix1({ {1, 2}, {3, 4} });
  neural_network::matrix matrix2({ { 5, 6 }, { 7, 8 } });
  neural_network::matrix result = matrix1 * matrix2;

  std::cout << "Matrix 1:\n";
  matrix1.print();

  std::cout << "Matrix 2:\n";
  matrix2.print();

  std::cout << "Result from matrix multiplication:\n";
  result.print();

  std::cout << "Press Enter to exit...";
  std::cin.get();

  return 0;
}
