#include "../../../include/models/neural_network/autoencoder.h"
#include <stdexcept>

namespace ml {
namespace nn {

Autoencoder::Autoencoder() {}

void Autoencoder::add_encoder_layer(std::shared_ptr<layer::Layer> layer) {
    encoder_layers_.push_back(layer);
}

void Autoencoder::add_decoder_layer(std::shared_ptr<layer::Layer> layer) {
    decoder_layers_.push_back(layer);
}

void Autoencoder::set_loss_function(std::shared_ptr<loss::LossFunction> loss) {
    loss_function_ = loss;
}

void Autoencoder::set_optimizer(std::shared_ptr<optimizer::Optimizer> optimizer) {
    optimizer_ = optimizer;
}

std::vector<double> Autoencoder::encode(const std::vector<double>& input) {
    std::vector<double> current = input;
    for (const auto& layer : encoder_layers_) {
        current = layer->forward(current);
    }
    return current;
}

std::vector<double> Autoencoder::decode(const std::vector<double>& encoded) {
    std::vector<double> current = encoded;
    for (const auto& layer : decoder_layers_) {
        current = layer->forward(current);
    }
    return current;
}

std::vector<double> Autoencoder::forward(const std::vector<double>& input) {
    return decode(encode(input));
}

std::vector<double> Autoencoder::backward(const std::vector<double>& grad_output) {
    // Backward pass through decoder
    std::vector<double> current_grad = grad_output;
    for (auto it = decoder_layers_.rbegin(); it != decoder_layers_.rend(); ++it) {
        current_grad = (*it)->backward(current_grad);
    }

    // Backward pass through encoder
    for (auto it = encoder_layers_.rbegin(); it != encoder_layers_.rend(); ++it) {
        current_grad = (*it)->backward(current_grad);
    }

    return current_grad;
}

void Autoencoder::train_step(const std::vector<double>& input) {
    // Forward pass
    std::vector<double> reconstructed = forward(input);

    // Compute reconstruction loss and its gradient
    double loss = loss_function_->compute(reconstructed, input);
    std::vector<double> grad_output = loss_function_->gradient(reconstructed, input);

    // Backward pass
    backward(grad_output);

    // Update weights using optimizer
    for (auto& layer : encoder_layers_) {
        std::vector<double> weights = layer->get_weights();
        std::vector<double> bias = layer->get_bias();
        optimizer_->update(weights, layer->get_grad_weights());
        layer->set_weights(weights);
        layer->set_bias(bias);
    }
    for (auto& layer : decoder_layers_) {
        std::vector<double> weights = layer->get_weights();
        std::vector<double> bias = layer->get_bias();
        optimizer_->update(weights, layer->get_grad_weights());
        layer->set_weights(weights);
        layer->set_bias(bias);
    }
}

void Autoencoder::fit(const std::vector<std::vector<double>>& inputs,
                     int epochs, int batch_size) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double epoch_loss = 0.0;
        for (size_t i = 0; i < inputs.size(); i += batch_size) {
            // Process batch
            for (size_t j = 0; j < batch_size && (i + j) < inputs.size(); ++j) {
                train_step(inputs[i + j]);
                epoch_loss += loss_function_->compute(forward(inputs[i + j]), inputs[i + j]);
            }
        }
        epoch_loss /= inputs.size();
        // You might want to log or store the epoch loss here
    }
}

double Autoencoder::evaluate(const std::vector<std::vector<double>>& inputs) {
    double total_loss = 0.0;
    for (const auto& input : inputs) {
        std::vector<double> reconstructed = forward(input);
        total_loss += loss_function_->compute(reconstructed, input);
    }
    return total_loss / inputs.size();
}

} // namespace nn
} // namespace ml