#pragma once

#include <vector>
#include <memory>
#include <stdexcept>
#include "components/layer.h"
#include "components/loss.h"
#include "components/optimizer.h"

namespace ml {
namespace nn {

class Autoencoder {
public:
    Autoencoder();
    void add_encoder_layer(std::shared_ptr<layer::Layer> layer);
    void add_decoder_layer(std::shared_ptr<layer::Layer> layer);
    void set_loss_function(std::shared_ptr<loss::LossFunction> loss);
    void set_optimizer(std::shared_ptr<optimizer::Optimizer> optimizer);
    
    std::vector<double> encode(const std::vector<double>& input);
    std::vector<double> decode(const std::vector<double>& encoded);
    std::vector<double> forward(const std::vector<double>& input);
    std::vector<double> backward(const std::vector<double>& grad_output);
    void train_step(const std::vector<double>& input);
    void fit(const std::vector<std::vector<double>>& inputs,
            int epochs, int batch_size);
    double evaluate(const std::vector<std::vector<double>>& inputs);

private:
    std::vector<std::shared_ptr<layer::Layer>> encoder_layers_;
    std::vector<std::shared_ptr<layer::Layer>> decoder_layers_;
    std::shared_ptr<loss::LossFunction> loss_function_;
    std::shared_ptr<optimizer::Optimizer> optimizer_;
};

} // namespace nn
} // namespace ml 