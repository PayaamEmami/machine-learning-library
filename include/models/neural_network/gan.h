#pragma once

#include <vector>
#include <memory>
#include <stdexcept>
#include "components/layer.h"
#include "components/activation.h"
#include "components/loss.h"
#include "components/optimizer.h"

namespace ml {
namespace nn {

class GAN {
public:
    GAN();
    ~GAN() = default;

    // Generator methods
    void add_generator_layer(std::shared_ptr<layer::Layer> layer);
    void set_generator_optimizer(std::shared_ptr<optimizer::Optimizer> optimizer);

    // Discriminator methods
    void add_discriminator_layer(std::shared_ptr<layer::Layer> layer);
    void set_discriminator_optimizer(std::shared_ptr<optimizer::Optimizer> optimizer);

    // Forward pass through generator
    std::vector<double> generate(const std::vector<double>& noise);

    // Forward pass through discriminator
    std::vector<double> discriminate(const std::vector<double>& input);

    // Training step
    void train_step(const std::vector<double>& real_data, int batch_size);

    // Train the GAN
    void fit(const std::vector<std::vector<double>>& real_data,
            int epochs, int batch_size);

    // Generate samples
    std::vector<double> generate_samples(int num_samples);

    std::vector<double> generate_noise(int batch_size);

private:
    // Train discriminator
    void train_discriminator(const std::vector<double>& real_data, 
                           const std::vector<double>& fake_data);

    // Train generator
    void train_generator(const std::vector<double>& fake_data);

    std::vector<std::shared_ptr<layer::Layer>> generator_layers_;
    std::vector<std::shared_ptr<layer::Layer>> discriminator_layers_;
    std::shared_ptr<optimizer::Optimizer> generator_optimizer_;
    std::shared_ptr<optimizer::Optimizer> discriminator_optimizer_;
    int latent_dim_;
};

} // namespace nn
} // namespace ml 