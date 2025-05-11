#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <numeric>

namespace mll::math {

// ============================
//  Matrix – 2-D dense array
// ============================

template <typename T>
class Matrix {
 public:
  using value_type = T;

  Matrix() noexcept : rows_(0), cols_(0), data_(nullptr) {}

  Matrix(std::size_t rows, std::size_t cols, const T &value = T{})
      : rows_(rows), cols_(cols), data_(new T[rows * cols]) {
    std::fill(data_, data_ + rows * cols, value);
  }

  static Matrix zeros(std::size_t rows, std::size_t cols) {
    return Matrix(rows, cols, T{});
  }

  static Matrix identity(std::size_t n) {
    Matrix I(n, n, T{});
    for (std::size_t i = 0; i < n; ++i) I(i, i) = T{1};
    return I;
  }

  // −−− Rule of Five −−−
  Matrix(const Matrix &other)
      : rows_(other.rows_), cols_(other.cols_), data_(nullptr) {
    if (other.data_) {
      data_ = new T[rows_ * cols_];
      std::copy(other.data_, other.data_ + rows_ * cols_, data_);
    }
  }

  Matrix(Matrix &&other) noexcept
      : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {
    other.rows_ = other.cols_ = 0;
    other.data_ = nullptr;
  }

  Matrix &operator=(Matrix other) noexcept {
    swap(other);
    return *this;
  }

  ~Matrix() { delete[] data_; }

  void swap(Matrix &other) noexcept {
    std::swap(rows_, other.rows_);
    std::swap(cols_, other.cols_);
    std::swap(data_, other.data_);
  }

  // −−− Element access −−−
  T &operator()(std::size_t r, std::size_t c) {
    assert(r < rows_ && c < cols_);
    return data_[r * cols_ + c];
  }

  const T &operator()(std::size_t r, std::size_t c) const {
    assert(r < rows_ && c < cols_);
    return data_[r * cols_ + c];
  }

  T *data() noexcept { return data_; }
  const T *data() const noexcept { return data_; }

  std::size_t rows() const noexcept { return rows_; }
  std::size_t cols() const noexcept { return cols_; }
  std::size_t size() const noexcept { return rows_ * cols_; }
  bool empty() const noexcept { return size() == 0; }

  // −−− Arithmetic (element wise / scalar) −−−
  Matrix operator+(const Matrix &rhs) const {
    assert(rows_ == rhs.rows_ && cols_ == rhs.cols_);
    Matrix out(rows_, cols_);
    for (std::size_t i = 0; i < size(); ++i)
      out.data_[i] = data_[i] + rhs.data_[i];
    return out;
  }

  Matrix &operator+=(const Matrix &rhs) {
    assert(rows_ == rhs.rows_ && cols_ == rhs.cols_);
    for (std::size_t i = 0; i < size(); ++i) data_[i] += rhs.data_[i];
    return *this;
  }

  Matrix operator*(T scalar) const {
    Matrix out(rows_, cols_);
    for (std::size_t i = 0; i < size(); ++i) out.data_[i] = data_[i] * scalar;
    return out;
  }

  Matrix &operator*=(T scalar) {
    for (std::size_t i = 0; i < size(); ++i) data_[i] *= scalar;
    return *this;
  }

  // −−− Matrix utilities −−−
  Matrix transpose() const {
    Matrix t(cols_, rows_);
    for (std::size_t r = 0; r < rows_; ++r)
      for (std::size_t c = 0; c < cols_; ++c) t(c, r) = (*this)(r, c);
    return t;
  }

  T sum() const {
    T s = T{};
    for (std::size_t i = 0; i < size(); ++i) s += data_[i];
    return s;
  }
  T mean() const { return size() ? sum() / static_cast<T>(size()) : T{}; }

 private:
  std::size_t rows_{0}, cols_{0};
  T *data_{nullptr};
};

// Overload for naive O(n³) matrix multiplication
template <typename T>
Matrix<T> operator*(const Matrix<T> &A, const Matrix<T> &B) {
  assert(A.cols() == B.rows());
  Matrix<T> C(A.rows(), B.cols(), T{});
  for (std::size_t i = 0; i < A.rows(); ++i) {
    for (std::size_t k = 0; k < A.cols(); ++k) {
      T aik = A(i, k);
      for (std::size_t j = 0; j < B.cols(); ++j) {
        C(i, j) += aik * B(k, j);
      }
    }
  }
  return C;
}

}  // namespace mll::math
