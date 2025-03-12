#include "models/linear/linear_regression.h"
#include "models/utils/matrix.h"
#include <iostream>

namespace ml
{
  LinearRegression::LinearRegression() : bias_(0.0)
  {
  }

  LinearRegression::~LinearRegression() = default;

  void LinearRegression::fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y)
  {
    int rows = X.size();
    int cols = X[0].size();

    // Convert input to matrix class
    models::utils::matrix X_mat(rows, cols + 1);
    models::utils::matrix y_mat(rows, 1);

    for (int i = 0; i < rows; ++i)
    {
      for (int j = 0; j < cols; ++j)
      {
        X_mat.set(i, j, X[i][j]);
      }
      X_mat.set(i, cols, 1.0); // Bias term
      y_mat.set(i, 0, y[i]);
    }

    // Compute (X^T X)
    models::utils::matrix Xt = X_mat.transpose();
    models::utils::matrix XtX = Xt * X_mat;

    // Compute (X^T y)
    models::utils::matrix Xty = Xt * y_mat;

    // Solve for w
    std::vector<double> y_vector(rows);
    for (int i = 0; i < rows; ++i)
    {
      y_vector[i] = Xty.get(i, 0);
    }

    std::vector<double> w = XtX.solve(y_vector);

    // Store parameters
    weights_.assign(w.begin(), w.end() - 1);
    bias_ = w.back();
  }

  std::vector<int> LinearRegression::predict(const std::vector<std::vector<double>>& X)
  {
    std::vector<int> predictions(X.size(), 0);
    for (size_t i = 0; i < X.size(); ++i)
    {
      double pred = bias_;
      for (size_t j = 0; j < X[i].size(); ++j)
      {
        pred += X[i][j] * weights_[j];
      }
      predictions[i] = static_cast<int>(pred);
    }
    return predictions;
  }
} // namespace ml
