#include "../../../include/models/neural_network/gan.h"
#include <stdexcept>
#include <random>

namespace ml {
namespace nn {

GAN::GAN() : latent_dim_(100) {}

void GAN::add_generator_layer(std::shared_ptr<layer::Layer> layer) {
    generator_layers_.push_back(layer);
}

void GAN::add_discriminator_layer(std::shared_ptr<layer::Layer> layer) {
    discriminator_layers_.push_back(layer);
}

void GAN::set_generator_optimizer(std::shared_ptr<optimizer::Optimizer> optimizer) {
    generator_optimizer_ = optimizer;
}

void GAN::set_discriminator_optimizer(std::shared_ptr<optimizer::Optimizer> optimizer) {
    discriminator_optimizer_ = optimizer;
}

std::vector<double> GAN::generate_noise(int batch_size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> noise(batch_size * latent_dim_);
    for (size_t i = 0; i < noise.size(); ++i) {
        noise[i] = dist(gen);
    }
    return noise;
}

std::vector<double> GAN::generate(const std::vector<double>& noise) {
    std::vector<double> current = noise;
    for (const auto& layer : generator_layers_) {
        current = layer->forward(current);
    }
    return current;
}

std::vector<double> GAN::discriminate(const std::vector<double>& input) {
    std::vector<double> current = input;
    for (const auto& layer : discriminator_layers_) {
        current = layer->forward(current);
    }
    return current;
}

void GAN::train_discriminator(const std::vector<double>& real_data,
                            const std::vector<double>& fake_data) {
    // Train on real data
    std::vector<double> real_output = discriminate(real_data);
    std::vector<double> real_target(real_output.size(), 1.0);
    std::vector<double> real_grad = discriminator_layers_.back()->backward(real_target);

    // Train on fake data
    std::vector<double> fake_output = discriminate(fake_data);
    std::vector<double> fake_target(fake_output.size(), 0.0);
    std::vector<double> fake_grad = discriminator_layers_.back()->backward(fake_target);

    // Update discriminator weights
    for (auto& layer : discriminator_layers_) {
        discriminator_optimizer_->update(layer);
    }
}

void GAN::train_generator(const std::vector<double>& fake_data) {
    // Generate fake data
    std::vector<double> fake_output = discriminate(fake_data);
    std::vector<double> target(fake_output.size(), 1.0);  // Try to fool discriminator

    // Backward pass through discriminator
    std::vector<double> grad = discriminator_layers_.back()->backward(target);

    // Backward pass through generator
    for (auto it = generator_layers_.rbegin(); it != generator_layers_.rend(); ++it) {
        grad = (*it)->backward(grad);
    }

    // Update generator weights
    for (auto& layer : generator_layers_) {
        generator_optimizer_->update(layer);
    }
}

void GAN::train_step(const std::vector<double>& real_data, int batch_size) {
    // Generate fake data
    std::vector<double> noise = generate_noise(batch_size);
    std::vector<double> fake_data = generate(noise);

    // Train discriminator
    train_discriminator(real_data, fake_data);

    // Train generator
    train_generator(fake_data);
}

void GAN::fit(const std::vector<std::vector<double>>& real_data,
              int epochs, int batch_size) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < real_data.size(); i += batch_size) {
            // Get batch of real data
            std::vector<double> batch;
            for (size_t j = 0; j < batch_size && (i + j) < real_data.size(); ++j) {
                batch.insert(batch.end(), real_data[i + j].begin(), real_data[i + j].end());
            }

            // Train on batch
            train_step(batch, batch_size);
        }
        // You might want to log or store the epoch progress here
    }
}

std::vector<double> GAN::generate_samples(int num_samples) {
    std::vector<double> noise = generate_noise(num_samples);
    return generate(noise);
}

} // namespace nn
} // namespace ml