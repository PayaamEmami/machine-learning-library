#pragma once

#include <vector>
#include <memory>
#include <stdexcept>
#include "activation.h"

namespace ml {
namespace nn {
namespace layer {

// Base layer interface
class Layer {
public:
    virtual ~Layer() = default;
    virtual std::vector<double> forward(const std::vector<double>& input) = 0;
    virtual std::vector<double> backward(const std::vector<double>& grad_output) = 0;
    virtual void update_weights(const std::vector<double>& grad_weights, const std::vector<double>& grad_bias) = 0;
    virtual std::vector<double> get_weights() const = 0;
    virtual std::vector<double> get_bias() const = 0;
    virtual void set_weights(const std::vector<double>& weights) = 0;
    virtual void set_bias(const std::vector<double>& bias) = 0;
    virtual std::vector<double> get_grad_weights() const = 0;
    virtual std::vector<double> get_grad_bias() const = 0;
};

// Dense (Fully Connected) Layer
class Dense : public Layer {
public:
    Dense(int input_size, int output_size, std::shared_ptr<activation::ActivationFunction> activation = nullptr)
        : input_size_(input_size), output_size_(output_size), activation_(activation) {
        // Initialize weights with Xavier/Glorot initialization
        weights_.resize(input_size * output_size);
        bias_.resize(output_size);

        double scale = std::sqrt(2.0 / (input_size + output_size));
        for (size_t i = 0; i < weights_.size(); ++i) {
            weights_[i] = (rand() / double(RAND_MAX) * 2.0 - 1.0) * scale;
        }
        for (size_t i = 0; i < bias_.size(); ++i) {
            bias_[i] = 0.0;
        }
    }

    std::vector<double> forward(const std::vector<double>& input) override {
        if (input.size() != input_size_) {
            throw std::invalid_argument("Input size mismatch");
        }

        // Compute linear transformation
        std::vector<double> output(output_size_);
        for (int i = 0; i < output_size_; ++i) {
            output[i] = bias_[i];
            for (int j = 0; j < input_size_; ++j) {
                output[i] += input[j] * weights_[j * output_size_ + i];
            }
        }

        // Apply activation if specified
        if (activation_) {
            output = activation_->forward(output);
        }

        return output;
    }

    std::vector<double> backward(const std::vector<double>& grad_output) override {
        if (grad_output.size() != output_size_) {
            throw std::invalid_argument("Gradient output size mismatch");
        }

        // Compute gradients with respect to weights and bias
        std::vector<double> grad_weights(weights_.size());
        std::vector<double> grad_bias(bias_.size());

        // If activation is present, compute its gradient
        std::vector<double> grad_activation = grad_output;
        if (activation_) {
            grad_activation = activation_->backward(grad_output);
        }

        // Compute gradients
        for (int i = 0; i < output_size_; ++i) {
            grad_bias[i] = grad_activation[i];
            for (int j = 0; j < input_size_; ++j) {
                grad_weights[j * output_size_ + i] = last_input_[j] * grad_activation[i];
            }
        }

        // Store gradients for weight update
        last_grad_weights_ = grad_weights;
        last_grad_bias_ = grad_bias;

        // Compute gradient with respect to input
        std::vector<double> grad_input(input_size_);
        for (int i = 0; i < input_size_; ++i) {
            grad_input[i] = 0.0;
            for (int j = 0; j < output_size_; ++j) {
                grad_input[i] += weights_[i * output_size_ + j] * grad_activation[j];
            }
        }

        return grad_input;
    }

    void update_weights(const std::vector<double>& grad_weights, const std::vector<double>& grad_bias) override {
        if (grad_weights.size() != weights_.size() || grad_bias.size() != bias_.size()) {
            throw std::invalid_argument("Gradient size mismatch");
        }

        for (size_t i = 0; i < weights_.size(); ++i) {
            weights_[i] -= grad_weights[i];
        }
        for (size_t i = 0; i < bias_.size(); ++i) {
            bias_[i] -= grad_bias[i];
        }
    }

    std::vector<double> get_weights() const override { return weights_; }
    std::vector<double> get_bias() const override { return bias_; }
    void set_weights(const std::vector<double>& weights) override { weights_ = weights; }
    void set_bias(const std::vector<double>& bias) override { bias_ = bias; }

    std::vector<double> get_grad_weights() const override { return last_grad_weights_; }
    std::vector<double> get_grad_bias() const override { return last_grad_bias_; }

private:
    int input_size_;
    int output_size_;
    std::shared_ptr<activation::ActivationFunction> activation_;
    std::vector<double> weights_;
    std::vector<double> bias_;
    std::vector<double> last_input_;
    std::vector<double> last_grad_weights_;
    std::vector<double> last_grad_bias_;
};

// Dropout Layer
class Dropout : public Layer {
public:
    explicit Dropout(double rate = 0.5) : rate_(rate), training_(true) {}

    std::vector<double> forward(const std::vector<double>& input) override {
        if (training_) {
            mask_.resize(input.size());
            for (size_t i = 0; i < input.size(); ++i) {
                mask_[i] = (rand() / double(RAND_MAX)) > rate_;
            }
        }

        std::vector<double> output(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = training_ ? input[i] * mask_[i] / (1.0 - rate_) : input[i];
        }
        return output;
    }

    std::vector<double> backward(const std::vector<double>& grad_output) override {
        std::vector<double> grad_input(grad_output.size());
        for (size_t i = 0; i < grad_output.size(); ++i) {
            grad_input[i] = training_ ? grad_output[i] * mask_[i] / (1.0 - rate_) : grad_output[i];
        }
        return grad_input;
    }

    void update_weights(const std::vector<double>&, const std::vector<double>&) override {}
    std::vector<double> get_weights() const override { return {}; }
    std::vector<double> get_bias() const override { return {}; }
    void set_weights(const std::vector<double>&) override {}
    void set_bias(const std::vector<double>&) override {}

    void set_training(bool training) { training_ = training; }

    std::vector<double> get_grad_weights() const override { return {}; }
    std::vector<double> get_grad_bias() const override { return {}; }

private:
    double rate_;
    bool training_;
    std::vector<double> mask_;
};

} // namespace layer
} // namespace nn
} // namespace ml