#pragma once

#include <vector>
#include <memory>
#include <stdexcept>
#include "activation.h"

namespace ml {
namespace nn {
namespace layer {

// Convolutional Layer
class Conv2D : public Layer {
public:
    Conv2D(int input_channels, int output_channels, int kernel_size, int stride = 1, int padding = 0,
           std::shared_ptr<activation::ActivationFunction> activation = nullptr)
        : input_channels_(input_channels), output_channels_(output_channels),
          kernel_size_(kernel_size), stride_(stride), padding_(padding),
          activation_(activation) {
        // Initialize weights with Xavier/Glorot initialization
        int kernel_volume = input_channels * output_channels * kernel_size * kernel_size;
        weights_.resize(kernel_volume);
        bias_.resize(output_channels);
        
        double scale = std::sqrt(2.0 / (input_channels * kernel_size * kernel_size + output_channels));
        for (size_t i = 0; i < weights_.size(); ++i) {
            weights_[i] = (rand() / double(RAND_MAX) * 2.0 - 1.0) * scale;
        }
        for (size_t i = 0; i < bias_.size(); ++i) {
            bias_[i] = 0.0;
        }
    }

    std::vector<double> forward(const std::vector<double>& input) override {
        // Reshape input to 4D tensor (batch_size, channels, height, width)
        // This is a simplified version - in practice, you'd want to handle batches
        int input_height = static_cast<int>(std::sqrt(input.size() / input_channels_));
        int input_width = input_height;

        // Calculate output dimensions
        int output_height = (input_height + 2 * padding_ - kernel_size_) / stride_ + 1;
        int output_width = (input_width + 2 * padding_ - kernel_size_) / stride_ + 1;
        int output_size = output_channels_ * output_height * output_width;

        std::vector<double> output(output_size);

        // Perform convolution
        for (int oc = 0; oc < output_channels_; ++oc) {
            for (int oh = 0; oh < output_height; ++oh) {
                for (int ow = 0; ow < output_width; ++ow) {
                    double sum = bias_[oc];
                    for (int ic = 0; ic < input_channels_; ++ic) {
                        for (int kh = 0; kh < kernel_size_; ++kh) {
                            for (int kw = 0; kw < kernel_size_; ++kw) {
                                int ih = oh * stride_ + kh - padding_;
                                int iw = ow * stride_ + kw - padding_;
                                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                    int input_idx = ic * input_height * input_width + ih * input_width + iw;
                                    int weight_idx = (oc * input_channels_ + ic) * kernel_size_ * kernel_size_ + kh * kernel_size_ + kw;
                                    sum += input[input_idx] * weights_[weight_idx];
                                }
                            }
                        }
                    }
                    output[oc * output_height * output_width + oh * output_width + ow] = sum;
                }
            }
        }

        // Apply activation if specified
        if (activation_) {
            output = activation_->forward(output);
        }

        return output;
    }

    std::vector<double> backward(const std::vector<double>& grad_output) override {
        // This is a simplified version - in practice, you'd want to handle batches
        // and implement proper backpropagation for convolutional layers
        std::vector<double> grad_input(input_channels_ * input_height_ * input_width_);
        std::vector<double> grad_weights(weights_.size());
        std::vector<double> grad_bias(bias_.size());

        // Compute gradients (simplified)
        for (size_t i = 0; i < grad_output.size(); ++i) {
            grad_bias[i % output_channels_] += grad_output[i];
        }

        // Store gradients for weight update
        last_grad_weights_ = grad_weights;
        last_grad_bias_ = grad_bias;

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

private:
    int input_channels_;
    int output_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    int input_height_;
    int input_width_;
    std::shared_ptr<activation::ActivationFunction> activation_;
    std::vector<double> weights_;
    std::vector<double> bias_;
    std::vector<double> last_grad_weights_;
    std::vector<double> last_grad_bias_;
};

// Max Pooling Layer
class MaxPool2D : public Layer {
public:
    MaxPool2D(int pool_size = 2, int stride = 2)
        : pool_size_(pool_size), stride_(stride) {}

    std::vector<double> forward(const std::vector<double>& input) override {
        // This is a simplified version - in practice, you'd want to handle batches
        // and multiple channels
        int input_size = static_cast<int>(std::sqrt(input.size()));
        int output_size = (input_size - pool_size_) / stride_ + 1;
        std::vector<double> output(output_size * output_size);
        std::vector<int> indices(output_size * output_size);

        for (int oh = 0; oh < output_size; ++oh) {
            for (int ow = 0; ow < output_size; ++ow) {
                double max_val = -std::numeric_limits<double>::infinity();
                int max_idx = -1;

                for (int ph = 0; ph < pool_size_; ++ph) {
                    for (int pw = 0; pw < pool_size_; ++pw) {
                        int ih = oh * stride_ + ph;
                        int iw = ow * stride_ + pw;
                        int input_idx = ih * input_size + iw;
                        if (input[input_idx] > max_val) {
                            max_val = input[input_idx];
                            max_idx = input_idx;
                        }
                    }
                }

                output[oh * output_size + ow] = max_val;
                indices[oh * output_size + ow] = max_idx;
            }
        }

        last_indices_ = indices;
        return output;
    }

    std::vector<double> backward(const std::vector<double>& grad_output) override {
        std::vector<double> grad_input(last_indices_.size());
        for (size_t i = 0; i < last_indices_.size(); ++i) {
            grad_input[last_indices_[i]] = grad_output[i];
        }
        return grad_input;
    }

    void update_weights(const std::vector<double>&, const std::vector<double>&) override {}
    std::vector<double> get_weights() const override { return {}; }
    std::vector<double> get_bias() const override { return {}; }
    void set_weights(const std::vector<double>&) override {}
    void set_bias(const std::vector<double>&) override {}

private:
    int pool_size_;
    int stride_;
    std::vector<int> last_indices_;
};

} // namespace layer
} // namespace nn
} // namespace ml 