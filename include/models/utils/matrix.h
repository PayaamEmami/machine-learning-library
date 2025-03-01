#pragma once
#include <vector>

namespace models
{
  namespace utils
  {
    class matrix
    {
    public:
      matrix(int rows, int columns);
      matrix(const std::vector<std::vector<double>>& values);
      matrix operator*(const matrix& operand) const;
      void print() const;
    private:
      int rows_, columns_;
      std::vector<std::vector<double>> data_;
    };
  }
}
