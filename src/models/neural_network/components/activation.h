#pragma once

#include <vector>
#include <cmath>

namespace ml {
namespace nn {
namespace activation {

// Base activation function interface
class ActivationFunction {
public:
    virtual ~ActivationFunction() = default;
    virtual std::vector<double> forward(const std::vector<double>& input) const = 0;
    virtual std::vector<double> backward(const std::vector<double>& input) const = 0;
};

// ReLU activation
class ReLU : public ActivationFunction {
public:
    std::vector<double> forward(const std::vector<double>& input) const override {
        std::vector<double> output(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = std::max(0.0, input[i]);
        }
        return output;
    }

    std::vector<double> backward(const std::vector<double>& input) const override {
        std::vector<double> grad(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            grad[i] = input[i] > 0.0 ? 1.0 : 0.0;
        }
        return grad;
    }
};

// Sigmoid activation
class Sigmoid : public ActivationFunction {
public:
    std::vector<double> forward(const std::vector<double>& input) const override {
        std::vector<double> output(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = 1.0 / (1.0 + std::exp(-input[i]));
        }
        return output;
    }

    std::vector<double> backward(const std::vector<double>& input) const override {
        std::vector<double> output = forward(input);
        std::vector<double> grad(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            grad[i] = output[i] * (1.0 - output[i]);
        }
        return grad;
    }
};

// Tanh activation
class Tanh : public ActivationFunction {
public:
    std::vector<double> forward(const std::vector<double>& input) const override {
        std::vector<double> output(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = std::tanh(input[i]);
        }
        return output;
    }

    std::vector<double> backward(const std::vector<double>& input) const override {
        std::vector<double> output = forward(input);
        std::vector<double> grad(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            grad[i] = 1.0 - output[i] * output[i];
        }
        return grad;
    }
};

// Softmax activation
class Softmax : public ActivationFunction {
public:
    std::vector<double> forward(const std::vector<double>& input) const override {
        std::vector<double> output(input.size());
        double max_val = *std::max_element(input.begin(), input.end());
        double sum = 0.0;
        
        // Subtract max for numerical stability
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = std::exp(input[i] - max_val);
            sum += output[i];
        }
        
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] /= sum;
        }
        return output;
    }

    std::vector<double> backward(const std::vector<double>& input) const override {
        std::vector<double> output = forward(input);
        std::vector<double> grad(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            grad[i] = output[i] * (1.0 - output[i]);
        }
        return grad;
    }
};

} // namespace activation
} // namespace nn
} // namespace ml 