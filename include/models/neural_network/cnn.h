#pragma once

#include <vector>
#include <memory>
#include <stdexcept>
#include "components/layer.h"
#include "components/loss.h"
#include "components/optimizer.h"

namespace ml {
namespace nn {

class CNN {
public:
    CNN();
    void add_layer(std::shared_ptr<layer::Layer> layer);
    void set_loss_function(std::shared_ptr<loss::LossFunction> loss);
    void set_optimizer(std::shared_ptr<optimizer::Optimizer> optimizer);
    
    std::vector<double> forward(const std::vector<double>& input);
    std::vector<double> backward(const std::vector<double>& grad_output);
    void train_step(const std::vector<double>& input, const std::vector<double>& target);
    void fit(const std::vector<std::vector<double>>& inputs, 
            const std::vector<std::vector<double>>& targets,
            int epochs, int batch_size);
    std::vector<double> predict(const std::vector<double>& input);

private:
    std::vector<std::shared_ptr<layer::Layer>> layers_;
    std::shared_ptr<loss::LossFunction> loss_function_;
    std::shared_ptr<optimizer::Optimizer> optimizer_;
    std::vector<int> input_shape_;
};

} // namespace nn
} // namespace ml 