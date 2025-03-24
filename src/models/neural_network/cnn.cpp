#include "models/neural_network/cnn.h"
#include <stdexcept>

namespace ml {
namespace nn {

// Empty implementation file since all methods are defined in the header
// This file is kept for consistency with the project structure

CNN::CNN() : input_shape_({0, 0, 0}) {}

void CNN::add_layer(std::shared_ptr<layer::Layer> layer) {
    layers_.push_back(layer);
}

void CNN::set_loss_function(std::shared_ptr<loss::LossFunction> loss) {
    loss_function_ = loss;
}

void CNN::set_optimizer(std::shared_ptr<optimizer::Optimizer> optimizer) {
    optimizer_ = optimizer;
}

std::vector<double> CNN::forward(const std::vector<double>& input) {
    if (input_shape_[0] == 0) {
        input_shape_ = {1, static_cast<int>(std::sqrt(input.size() / 3)), 3};
    }

    std::vector<double> current = input;
    for (const auto& layer : layers_) {
        current = layer->forward(current);
    }
    return current;
}

std::vector<double> CNN::backward(const std::vector<double>& grad_output) {
    std::vector<double> current_grad = grad_output;
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
        current_grad = (*it)->backward(current_grad);
    }
    return current_grad;
}

void CNN::train_step(const std::vector<double>& input, const std::vector<double>& target) {
    // Forward pass
    std::vector<double> output = forward(input);
    
    // Compute loss and its gradient
    double loss = loss_function_->compute(output, target);
    std::vector<double> grad_output = loss_function_->gradient(output, target);
    
    // Backward pass
    backward(grad_output);
    
    // Update weights using optimizer
    for (auto& layer : layers_) {
        optimizer_->update(layer);
    }
}

void CNN::fit(const std::vector<std::vector<double>>& inputs, 
              const std::vector<std::vector<double>>& targets,
              int epochs, int batch_size) {
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Input and target sizes must match");
    }

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double epoch_loss = 0.0;
        for (size_t i = 0; i < inputs.size(); i += batch_size) {
            // Process batch
            for (size_t j = 0; j < batch_size && (i + j) < inputs.size(); ++j) {
                train_step(inputs[i + j], targets[i + j]);
                epoch_loss += loss_function_->compute(forward(inputs[i + j]), targets[i + j]);
            }
        }
        epoch_loss /= inputs.size();
        // You might want to log or store the epoch loss here
    }
}

std::vector<double> CNN::predict(const std::vector<double>& input) {
    return forward(input);
}

} // namespace nn
} // namespace ml 