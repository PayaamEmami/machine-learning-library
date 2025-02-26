#include "matrix.h"
#include <stdexcept>
#include <iostream>

namespace neural_network
{
  matrix::matrix(const int rows, const int columns) : rows_(rows), columns_(columns)
  {
    data_.resize(rows, std::vector<double>(columns, 0.0));
  }

  matrix::matrix(const std::vector<std::vector<double>>& values)
  {
    rows_ = values.size();
    columns_ = values[0].size();
    data_ = values;
  }

  matrix matrix::operator*(const matrix& operand) const
  {
    if (columns_ != operand.rows_)
    {
      throw std::invalid_argument("Failed to execute matrix multiplication, matrix dimensions do not match.");
    }

    matrix result(rows_, operand.columns_);

    for (int i = 0; i < rows_; ++i)
    {
      for (int j = 0; j < operand.columns_; ++j)
      {
        for (int k = 0; k < columns_; ++k)
        {
          result.data_[i][j] += data_[i][k] * operand.data_[k][j];
        }
      }
    }

    return result;
  }

  void matrix::print() const
  {
    for (const auto& row : data_)
    {
      for (double val : row)
      {
        std::cout << val << " ";
      }
      std::cout << "\n";
    }
  }
}
