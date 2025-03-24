#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>

namespace ml {
namespace nn {
namespace optimizer {

// Base optimizer interface
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void update(std::vector<double>& weights, const std::vector<double>& gradients) = 0;
};

// Stochastic Gradient Descent
class SGD : public Optimizer {
public:
    explicit SGD(double learning_rate = 0.01) : learning_rate_(learning_rate) {}

    void update(std::vector<double>& weights, const std::vector<double>& gradients) override {
        if (weights.size() != gradients.size()) {
            throw std::invalid_argument("Weights and gradients must have the same size");
        }

        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] -= learning_rate_ * gradients[i];
        }
    }

private:
    double learning_rate_;
};

// Adam optimizer
class Adam : public Optimizer {
public:
    Adam(double learning_rate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
        : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0) {}

    void update(std::vector<double>& weights, const std::vector<double>& gradients) override {
        if (weights.size() != gradients.size()) {
            throw std::invalid_argument("Weights and gradients must have the same size");
        }

        t_++;
        if (m_.empty()) {
            m_.resize(weights.size(), 0.0);
            v_.resize(weights.size(), 0.0);
        }

        double mt = m_[0] / (1.0 - std::pow(beta1_, t_));
        double vt = v_[0] / (1.0 - std::pow(beta2_, t_));

        for (size_t i = 0; i < weights.size(); ++i) {
            // Update biased first moment estimate
            m_[i] = beta1_ * m_[i] + (1.0 - beta1_) * gradients[i];
            // Update biased second raw moment estimate
            v_[i] = beta2_ * v_[i] + (1.0 - beta2_) * gradients[i] * gradients[i];

            // Compute bias-corrected first moment estimate
            mt = m_[i] / (1.0 - std::pow(beta1_, t_));
            // Compute bias-corrected second raw moment estimate
            vt = v_[i] / (1.0 - std::pow(beta2_, t_));

            // Update parameters
            weights[i] -= learning_rate_ * mt / (std::sqrt(vt) + epsilon_);
        }
    }

private:
    double learning_rate_;
    double beta1_;
    double beta2_;
    double epsilon_;
    int t_;
    std::vector<double> m_;  // First moment vector
    std::vector<double> v_;  // Second moment vector
};

// RMSprop optimizer
class RMSprop : public Optimizer {
public:
    RMSprop(double learning_rate = 0.001, double decay_rate = 0.9, double epsilon = 1e-8)
        : learning_rate_(learning_rate), decay_rate_(decay_rate), epsilon_(epsilon) {}

    void update(std::vector<double>& weights, const std::vector<double>& gradients) override {
        if (weights.size() != gradients.size()) {
            throw std::invalid_argument("Weights and gradients must have the same size");
        }

        if (cache_.empty()) {
            cache_.resize(weights.size(), 0.0);
        }

        for (size_t i = 0; i < weights.size(); ++i) {
            cache_[i] = decay_rate_ * cache_[i] + (1.0 - decay_rate_) * gradients[i] * gradients[i];
            weights[i] -= learning_rate_ * gradients[i] / (std::sqrt(cache_[i]) + epsilon_);
        }
    }

private:
    double learning_rate_;
    double decay_rate_;
    double epsilon_;
    std::vector<double> cache_;
};

} // namespace optimizer
} // namespace nn
} // namespace ml 