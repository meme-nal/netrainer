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


class NN: public torch::nn::Module {
private:
  std::map<std::string, std::shared_ptr<BaseLayer>> _layers;
  std::vector<std::string>                          _layer_names;
  std::map<std::string, torch::Tensor>              _layer_inputs;
  std::map<std::string, torch::Tensor>              _layer_outputs;
  json                                              _net_configuration;

  //bool                                              _trainMode;

public:
  NN(json& nn_arch_cfg);
  torch::Tensor forward(torch::Tensor X);
  std::vector<torch::Tensor> parameters();

  //void train() { _trainMode = true; }
  //void eval() { _trainMode = false; }
};

std::shared_ptr<BaseLayer> getLayer(const std::string& layerName, json& layerJson);

void print_arch(const std::shared_ptr<NN>& model);
void print_to_terminal(const std::string& message);
size_t count_model_params(const std::shared_ptr<NN>& model);
CommonOptions loadCommonOptions(const std::string& nn_cfg_path);
std::shared_ptr<NN> loadModel(const std::string& nn_cfg_path);
std::shared_ptr<torch::optim::Optimizer> loadOptimizer(const std::string& nn_cfg_path, std::shared_ptr<NN>& model);
std::shared_ptr<BaseLayer> loadCriterion(const std::string& nn_cfg_path);
std::shared_ptr<BaseLabelGenerator> loadLabelGenerator(const std::string& nn_cfg_path);


#endif // NN_H