#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>

namespace ml {
namespace nn {
namespace loss {

// Base loss function interface
class LossFunction {
public:
    virtual ~LossFunction() = default;
    virtual double compute(const std::vector<double>& predictions, const std::vector<double>& targets) const = 0;
    virtual std::vector<double> gradient(const std::vector<double>& predictions, const std::vector<double>& targets) const = 0;
};

// Mean Squared Error loss
class MSE : public LossFunction {
public:
    double compute(const std::vector<double>& predictions, const std::vector<double>& targets) const override {
        if (predictions.size() != targets.size()) {
            throw std::invalid_argument("Predictions and targets must have the same size");
        }

        double sum = 0.0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            double diff = predictions[i] - targets[i];
            sum += diff * diff;
        }
        return sum / predictions.size();
    }

    std::vector<double> gradient(const std::vector<double>& predictions, const std::vector<double>& targets) const override {
        if (predictions.size() != targets.size()) {
            throw std::invalid_argument("Predictions and targets must have the same size");
        }

        std::vector<double> grad(predictions.size());
        for (size_t i = 0; i < predictions.size(); ++i) {
            grad[i] = 2.0 * (predictions[i] - targets[i]) / predictions.size();
        }
        return grad;
    }
};

// Binary Cross Entropy loss
class BinaryCrossEntropy : public LossFunction {
public:
    double compute(const std::vector<double>& predictions, const std::vector<double>& targets) const override {
        if (predictions.size() != targets.size()) {
            throw std::invalid_argument("Predictions and targets must have the same size");
        }

        double sum = 0.0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            // Add small epsilon to avoid log(0)
            double pred = std::max(1e-7, std::min(1.0 - 1e-7, predictions[i]));
            sum += targets[i] * std::log(pred) + (1.0 - targets[i]) * std::log(1.0 - pred);
        }
        return -sum / predictions.size();
    }

    std::vector<double> gradient(const std::vector<double>& predictions, const std::vector<double>& targets) const override {
        if (predictions.size() != targets.size()) {
            throw std::invalid_argument("Predictions and targets must have the same size");
        }

        std::vector<double> grad(predictions.size());
        for (size_t i = 0; i < predictions.size(); ++i) {
            // Add small epsilon to avoid division by zero
            double pred = std::max(1e-7, std::min(1.0 - 1e-7, predictions[i]));
            grad[i] = (pred - targets[i]) / predictions.size();
        }
        return grad;
    }
};

// Categorical Cross Entropy loss
class CategoricalCrossEntropy : public LossFunction {
public:
    double compute(const std::vector<double>& predictions, const std::vector<double>& targets) const override {
        if (predictions.size() != targets.size()) {
            throw std::invalid_argument("Predictions and targets must have the same size");
        }

        double sum = 0.0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            // Add small epsilon to avoid log(0)
            double pred = std::max(1e-7, std::min(1.0 - 1e-7, predictions[i]));
            sum += targets[i] * std::log(pred);
        }
        return -sum / predictions.size();
    }

    std::vector<double> gradient(const std::vector<double>& predictions, const std::vector<double>& targets) const override {
        if (predictions.size() != targets.size()) {
            throw std::invalid_argument("Predictions and targets must have the same size");
        }

        std::vector<double> grad(predictions.size());
        for (size_t i = 0; i < predictions.size(); ++i) {
            // Add small epsilon to avoid division by zero
            double pred = std::max(1e-7, std::min(1.0 - 1e-7, predictions[i]));
            grad[i] = (pred - targets[i]) / predictions.size();
        }
        return grad;
    }
};

} // namespace loss
} // namespace nn
} // namespace ml 