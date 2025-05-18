#ifndef NN_H
#define NN_H

#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>
#include <map>

#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include "layer.h"

#include "label_gen/label_gen.h"
#include "label_gen/cls_label_gen.h"

using json = nlohmann::json;

struct CommonOptions {
  CommonOptions(size_t num_epochs, std::string to_save_path, std::string start_from_epoch, int train_part, int test_part, size_t mbatch_size, torch::Device device)
  : _num_epochs(num_epochs)
  , _to_save_path(to_save_path)
  , _start_from_epoch(start_from_epoch)
  , _train_part(train_part)
  , _test_part(test_part)
  , _mbatch_size(mbatch_size)
  , _device(device) {} 
  
  size_t _num_epochs;
  std::string _to_save_path;
  std::string _start_from_epoch;
  int _train_part;
  int _test_part;
  size_t _mbatch_size;
  torch::Device _device;
};


class NetImpl : public torch::nn::Module {
private:
  std::map<std::string, std::shared_ptr<torch::nn::Module>> _tmp_layers;
  /* ARCH */
  std::vector<std::string>                          _tensor_names;
  //std::map<std::string, std::string>                _layer_nonlinearities;
  std::map<std::string, std::vector<std::string>>   _layer_lists;
  std::vector<std::string>                          _layer_names;
  /* ARCH */
  json                                              _net_configuration;

private:
  std::vector<std::string> findInputsByOutput(const std::string& output, json& nn_arch_cfg) const;
  std::string findLayerByOutput(const std::string& output, json& nn_arch_cfg) const;
  //torch::Tensor calculate_value(std::map<std::string, std::vector<std::string>>& graph, const std::string& vertex, torch::Tensor input_tensor);
public:
  void print_modules() {
    // Iterate over all submodules
    for (const auto& named_module : this->named_modules()) {
      std::cout << "Module name: " << named_module.key() << std::endl;
      std::cout << "Module type: " << named_module.value()->name() << std::endl;
    }
  }

public:
  NetImpl(json& nn_arch_cfg);
  torch::Tensor forward(torch::Tensor X);
  //std::vector<torch::Tensor> parameters();
};
TORCH_MODULE(Net);

//std::shared_ptr<BaseLayer> getLayer(const std::string& layerName, json& layerJson);

void print_arch(const std::shared_ptr<Net>& model);
void print_model_weights(const std::shared_ptr<Net>& model);
void print_weights(const torch::nn::Module& module);
void print_to_terminal(const std::string& message);
size_t count_model_params(const torch::nn::Module& module);
size_t count_parameters(Net& model);
CommonOptions loadCommonOptions(const std::string& nn_cfg_path);
Net loadModel(const std::string& nn_cfg_path);
std::shared_ptr<torch::optim::Optimizer> loadOptimizer(const std::string& nn_cfg_path, Net& model);
std::shared_ptr<torch::nn::Module> loadCriterion(const std::string& nn_cfg_path);
std::shared_ptr<BaseLabelGenerator> loadLabelGenerator(const std::string& nn_cfg_path);


#endif // NN_H