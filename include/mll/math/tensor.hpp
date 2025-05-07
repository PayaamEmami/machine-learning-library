#pragma once

#include <vector>
#include <cstddef>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <initializer_list>
#include <stdexcept>

namespace mll::math {

// ============================
//  Tensor – N-D dense array
// ============================

template<typename T>
class Tensor {
public:
    using value_type = T;

    Tensor() noexcept : size_(0), data_(nullptr) {}

    explicit Tensor(const std::vector<std::size_t> &shape, const T &value = T{})
        : shape_(shape), strides_(shape.size()), size_(calc_size(shape)) {
        compute_strides();
        data_ = new T[size_];
        std::fill(data_, data_ + size_, value);
    }

    // −−− Rule of Five −−−
    Tensor(const Tensor &other)
        : shape_(other.shape_), strides_(other.strides_), size_(other.size_), data_(nullptr) {
        if (other.data_) {
            data_ = new T[size_];
            std::copy(other.data_, other.data_ + size_, data_);
        }
    }

    Tensor(Tensor &&other) noexcept
        : shape_(std::move(other.shape_)),
          strides_(std::move(other.strides_)),
          size_(other.size_),
          data_(other.data_) {
        other.size_ = 0;
        other.data_ = nullptr;
    }

    Tensor &operator=(Tensor other) noexcept {
        swap(other);
        return *this;
    }

    ~Tensor() { delete[] data_; }

    void swap(Tensor &other) noexcept {
        shape_.swap(other.shape_);
        strides_.swap(other.strides_);
        std::swap(size_, other.size_);
        std::swap(data_, other.data_);
    }

    // −−− Element access −−−
    T &operator()(std::initializer_list<std::size_t> indices) {
        return data_[flat_index(indices)];
    }

    const T &operator()(std::initializer_list<std::size_t> indices) const {
        return data_[flat_index(indices)];
    }

    // −−− Meta data −−−
    const std::vector<std::size_t> &shape() const noexcept { return shape_; }
    const std::vector<std::size_t> &strides() const noexcept { return strides_; }
    std::size_t ndim() const noexcept { return shape_.size(); }
    std::size_t size() const noexcept { return size_; }

    T *data() noexcept { return data_; }
    const T *data() const noexcept { return data_; }

    // −−− Arithmetic −−−
    Tensor operator+(const Tensor &rhs) const {
        assert(shape_ == rhs.shape_);
        Tensor out(shape_);
        for (std::size_t i = 0; i < size_; ++i) out.data_[i] = data_[i] + rhs.data_[i];
        return out;
    }

    Tensor &operator+=(const Tensor &rhs) {
        assert(shape_ == rhs.shape_);
        for (std::size_t i = 0; i < size_; ++i) data_[i] += rhs.data_[i];
        return *this;
    }

    Tensor operator*(T scalar) const {
        Tensor out(shape_);
        for (std::size_t i = 0; i < size_; ++i) out.data_[i] = data_[i] * scalar;
        return out;
    }

    Tensor &operator*=(T scalar) {
        for (std::size_t i = 0; i < size_; ++i) data_[i] *= scalar;
        return *this;
    }

    T sum() const {
        T s = T{};
        for (std::size_t i = 0; i < size_; ++i) s += data_[i];
        return s;
    }

    // −−− Shape helper −−−
    Tensor reshape(const std::vector<std::size_t> &new_shape) const {
        if (calc_size(new_shape) != size_) throw std::runtime_error("reshape size mismatch");
        Tensor view(*this);          // copy meta & data pointer
        view.shape_ = new_shape;
        view.compute_strides();
        return view;                // shares memory
    }

private:
    // −−− helpers −−−

    static std::size_t calc_size(const std::vector<std::size_t> &shape) {
        return std::accumulate(shape.begin(), shape.end(), std::size_t{1}, std::multiplies<std::size_t>());
    }

    void compute_strides() {
        if (shape_.empty()) return;
        strides_[shape_.size() - 1] = 1;
        for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i)
            strides_[i] = strides_[i + 1] * shape_[i + 1];
    }

    std::size_t flat_index(std::initializer_list<std::size_t> indices) const {
        if (indices.size() != shape_.size()) throw std::out_of_range("incorrect number of indices");
        std::size_t offset = 0;
        auto it_stride = strides_.cbegin();
        auto it_shape = shape_.cbegin();
        auto it_idx = indices.begin();
        for (; it_idx != indices.end(); ++it_idx, ++it_stride, ++it_shape) {
            if (*it_idx >= *it_shape) throw std::out_of_range("index out of bounds");
            offset += *it_idx * *it_stride;
        }
        return offset;
    }

    // −−− data members −−−
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::size_t size_;
    T *data_;
};

} // namespace mll::math
