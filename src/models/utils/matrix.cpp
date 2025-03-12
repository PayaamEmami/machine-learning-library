#include "models/utils/matrix.h"
#include <stdexcept>
#include <iostream>

namespace models::utils
{
  matrix::matrix(int rows, int columns) : rows_(rows), columns_(columns)
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

  matrix matrix::transpose() const
  {
    matrix result(columns_, rows_);
    for (int i = 0; i < rows_; ++i)
    {
      for (int j = 0; j < columns_; ++j)
      {
        result.set(j, i, data_[i][j]);
      }
    }
    return result;
  }

  matrix matrix::inverse() const
  {
    if (rows_ != columns_)
    {
      throw std::invalid_argument("Matrix must be square to compute inverse.");
    }

    int n = rows_;
    matrix aug(n, 2 * n);

    // Initialize augmented matrix [A | I]
    for (int i = 0; i < n; ++i)
    {
      for (int j = 0; j < n; ++j)
      {
        aug.set(i, j, data_[i][j]);
      }
      aug.set(i, n + i, 1.0); // Identity matrix
    }

    // Gaussian elimination
    for (int i = 0; i < n; ++i)
    {
      double diag = aug.get(i, i);
      if (diag == 0)
      {
        throw std::runtime_error("Matrix is singular and cannot be inverted.");
      }
      for (int j = 0; j < 2 * n; ++j)
      {
        aug.set(i, j, aug.get(i, j) / diag);
      }

      for (int k = 0; k < n; ++k)
      {
        if (k == i) continue;
        double factor = aug.get(k, i);
        for (int j = 0; j < 2 * n; ++j)
        {
          aug.set(k, j, aug.get(k, j) - factor * aug.get(i, j));
        }
      }
    }

    // Extract inverse matrix
    matrix inv(n, n);
    for (int i = 0; i < n; ++i)
    {
      for (int j = 0; j < n; ++j)
      {
        inv.set(i, j, aug.get(i, j + n));
      }
    }

    return inv;
  }

  std::vector<double> matrix::solve(const std::vector<double>& b) const
  {
    if (rows_ != columns_)
    {
      throw std::invalid_argument("Matrix must be square to solve Ax = b.");
    }

    matrix invA = inverse();
    std::vector<double> x(rows_);
    for (int i = 0; i < rows_; ++i)
    {
      x[i] = 0;
      for (int j = 0; j < columns_; ++j)
      {
        x[i] += invA.get(i, j) * b[j];
      }
    }
    return x;
  }

  double matrix::get(int row, int col) const
  {
    if (row >= rows_ || col >= columns_)
    {
      throw std::out_of_range("Matrix index out of bounds.");
    }
    return data_[row][col];
  }

  void matrix::set(int row, int col, double value)
  {
    if (row >= rows_ || col >= columns_)
    {
      throw std::out_of_range("Matrix index out of bounds.");
    }
    data_[row][col] = value;
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
