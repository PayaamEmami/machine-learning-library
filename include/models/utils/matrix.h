#ifndef MATRIX_H
#define MATRIX_H

#include <vector>

namespace models::utils
{
  class matrix
  {
  public:
    matrix(int rows, int columns);
    matrix(const std::vector<std::vector<double>>& values);

    matrix operator*(const matrix& operand) const;
    matrix transpose() const;
    matrix inverse() const;
    std::vector<double> solve(const std::vector<double>& b) const;

    double get(int row, int col) const; // Safe accessor
    void set(int row, int col, double value); // Setter method

    void print() const;

  private:
    int rows_, columns_;
    std::vector<std::vector<double>> data_;
  };
}

#endif // MATRIX_H
