#ifndef NN_H
#define NN_H

#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <memory>
#include <map>

#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include "layer.h"

using json = nlohmann::json;

struct CommonOptions {
  CommonOptions(size_t num_epochs, std::string to_save_path, size_t mbatch_size, torch::Device device)
  : _num_epochs(num_epochs)
  , _to_save_path(to_save_path)
  , _mbatch_size(mbatch_size)
  , _device(device) {} 
  
  size_t _num_epochs;
  std::string _to_save_path;
  size_t _mbatch_size;
  torch::Device _device;
};

/*
enum class LayerType: uint8_t {
  LINEAR,

  LOSS
};

enum class LossType: uint8_t {
  MAE,
  MSE
};

enum class NonlinearityType: uint8_t {
  LINEAR,
  SIGMOID,
  TANH,
  RELU
};
*/

class NN: public torch::nn::Module {
private:
  std::map<std::string, std::shared_ptr<BaseLayer>> _layers;
  std::vector<std::string>                 _layer_names;
  std::map<std::string, torch::Tensor>     _layer_inputs;
  std::map<std::string, torch::Tensor>     _layer_outputs;
  json                                     _net_configuration;

public:
  NN(json& nn_arch_cfg); // TODO: make final layer
  torch::Tensor forward(torch::Tensor X);

  std::vector<torch::Tensor> parameters() {
    std::vector<torch::Tensor> all_parameters;

    for (const auto& layer_pair : _layers) {
      auto layer_parameters = layer_pair.second->parameters();
      all_parameters.insert(all_parameters.end(),
                            layer_parameters.begin(),
                            layer_parameters.end());
    }

    return all_parameters;
  }
};

///
// FABRIC FUNCTION
///
std::shared_ptr<BaseLayer> getLayer(const std::string& layerName, json& layerJson);

void print_arch(const std::shared_ptr<NN>& model); // TODO: enhance with layer types
size_t count_model_params(const std::shared_ptr<NN>& model); // TODO: Edit
CommonOptions loadCommonOptions(const std::string& nn_cfg_path);
std::shared_ptr<NN> loadModel(const std::string& nn_cfg_path);
std::shared_ptr<torch::optim::Optimizer> loadOptimizer(const std::string& nn_cfg_path, std::shared_ptr<NN>& model);
std::shared_ptr<BaseLayer> loadCriterion(const std::string& nn_cfg_path);


#endif // NN_H