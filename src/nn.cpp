#include "nn.h"

CommonOptions loadCommonOptions(const std::string& nn_cfg_path) {
  json nn_cfg_json;
  std::ifstream nn_cfg_file(nn_cfg_path);

  if (!nn_cfg_file.is_open()) {
    std::cerr << "Could not open the file: " << nn_cfg_path << std::endl;
  }

  nn_cfg_file >> nn_cfg_json;
  nn_cfg_file.close();

  return CommonOptions(
    nn_cfg_json["num_epochs"].get<size_t>(),
    nn_cfg_json["to_save_path"].get<std::string>(),
    nn_cfg_json["mbatch_size"].get<size_t>(),
    torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);  
}

std::shared_ptr<torch::optim::Optimizer> loadOptimizer(const std::string& nn_cfg_path, std::shared_ptr<NN>& model) {
  json nn_cfg_json;
  std::ifstream nn_cfg_file(nn_cfg_path);

  if (!nn_cfg_file.is_open()) {
    std::cerr << "Could not open the file: " << nn_cfg_path << std::endl;
  }

  nn_cfg_file >> nn_cfg_json;
  nn_cfg_file.close();

  float lr = nn_cfg_json["optimizer"]["lr"].get<float>();
  std::string optimType = nn_cfg_json["optimizer"]["type"].get<std::string>();

  if (optimType == "SGD") {
    return std::make_shared<torch::optim::SGD>(model->parameters(), lr);
  } else if (optimType == "Adam") {
    return std::make_shared<torch::optim::Adam>(model->parameters(), lr);
  } else {
    std::cerr << "Incorrect Optimizer type!\n";
  }
}

std::shared_ptr<BaseLayer> loadCriterion(const std::string& nn_cfg_path) {
  json nn_cfg_json;
  std::ifstream nn_cfg_file(nn_cfg_path);

  if (!nn_cfg_file.is_open()) {
    std::cerr << "Could not open the file: " << nn_cfg_path << std::endl;
  }

  nn_cfg_file >> nn_cfg_json;
  nn_cfg_file.close();

  std::string costType = nn_cfg_json["arch"]["final"]["cost"];

  if (costType == "MSE") { return std::make_shared<MSELossLayer>(); } 
  else if (costType == "MAE") { return std::make_shared<MAELossLayer>(); }
  else if (costType == "CrossEntropy") { return std::make_shared<CrossEntropyLossLayer>(); }
  else {
    std::cerr << "Incorrect criterion type!\n";
  }
}

std::shared_ptr<BaseLayer> getLayer(const std::string& layerName, json& layerJson) {
  // COMMON LAYERS
  if (layerJson["type"] == "dense") { return std::make_shared<DenseLayer>(layerName, layerJson); }
  else if (layerJson["type"] == "conv") { return std::make_shared<ConvLayer>(layerName, layerJson); }
  else if (layerJson["type"] == "pooling") { return std::make_shared<PoolingLayer>(layerName, layerJson); }
  
  // AUXILIARY LAYERS
  else if (layerJson["type"] == "flatten") { return std::make_shared<FlattenLayer>(layerName, layerJson); }
  else if (layerJson["type"] == "reshape") { return std::make_shared<ReshapeLayer>(layerName, layerJson); }
  
  // LOSS LAYERS

  // SPECIAL LAYERS
  else if (layerJson["type"] == "bn") { return std::make_shared<BatchNormLayer>(layerName, layerJson); }

  else {
    std::cout << "Incorrect layer type\n";
  }
}

NN::NN(json& nn_arch_cfg) {
  _net_configuration = nn_arch_cfg;
  size_t num_layers = nn_arch_cfg.size();

  // find first layer
  for (auto& layer : nn_arch_cfg.items()) {
    if (layer.value()["input"] == "main_data") {
      _layer_names.push_back(layer.key());
    }
  }

  // get remain layers without final
  for (size_t li {0}; li < num_layers - 2; ++li) {
    for (auto& layer : nn_arch_cfg.items()) {
      if (layer.value()["input"] == nn_arch_cfg[_layer_names[li]]["output"]) {
        _layer_names.push_back(layer.key());
      }
    }
  }

  for (size_t li {0}; li < num_layers - 1; ++li) {
    _layers[_layer_names[li]] = getLayer(_layer_names[li], nn_arch_cfg[_layer_names[li]]);
  }
}

torch::Tensor NN::forward(torch::Tensor input) {
  size_t num_layers = _net_configuration.size();

  _layer_inputs.clear();
  _layer_outputs.clear();

  // first out
  _layer_inputs[_net_configuration[_layer_names[0]]["input"]] = input;
  torch::Tensor out = _layers[_layer_names[0]]->forward(input).to(torch::kCPU);
  _layer_outputs[_net_configuration[_layer_names[0]]["output"]] = out;


  // other layers
  for (size_t li {1}; li < num_layers - 1; ++li) {
    torch::Tensor T = _layer_outputs[_net_configuration[_layer_names[li - 1]]["output"]].to(torch::kCPU);
    _layer_inputs[_net_configuration[_layer_names[li]]["input"]] = T;
    torch::Tensor T_out = _layers[_layer_names[li]]->forward(T).to(torch::kCPU);
    _layer_outputs[_net_configuration[_layer_names[li]]["output"]] = T_out;
  }

  return _layer_outputs[_net_configuration[_layer_names.back()]["output"]];
}

void print_arch(const std::shared_ptr<NN>& model) {
  std::cout << "====== Model arch ======\n";
  for (const auto& param : model->named_parameters()) {
    std::cout << param.key() << " : " << param.value().sizes() << "\n\n";
  }
  std::cout << '\n';
}

std::vector<torch::Tensor> NN::parameters() {
  std::vector<torch::Tensor> all_parameters;

  for (const auto& layer_pair : _layers) {
    auto layer_parameters = layer_pair.second->parameters();
    all_parameters.insert(all_parameters.end(),
                          layer_parameters.begin(),
                          layer_parameters.end());
  }

  return all_parameters;
}

size_t count_model_params(const std::shared_ptr<NN>& model) {
  size_t num_params = 0;
  for (const auto& param : model->parameters()) {
    num_params += param.numel();
  }
  return num_params;
}

std::shared_ptr<NN> loadModel(const std::string& nn_cfg_path) {
  json nn_cfg_json;
  std::ifstream nn_cfg_file(nn_cfg_path);

  if (!nn_cfg_file.is_open()) {
    std::cerr << "Could not open the file: " << nn_cfg_path << std::endl;
  }

  nn_cfg_file >> nn_cfg_json;
  nn_cfg_file.close();

  std::cout << "Running on "<< (torch::cuda::is_available() ? "CUDA" : "CPU") << "\n\n";

  std::shared_ptr<NN> model = std::make_shared<NN>(nn_cfg_json["arch"]);
  return model;
}

std::shared_ptr<BaseLabelGenerator> loadLabelGenerator(const std::string& nn_cfg_path) {
  json nn_cfg_json;
  std::ifstream nn_cfg_file(nn_cfg_path);

  if (!nn_cfg_file.is_open()) {
    std::cerr << "Could not open the file: " << nn_cfg_path << std::endl;
  }

  nn_cfg_file >> nn_cfg_json;
  nn_cfg_file.close();

  std::string labelType = nn_cfg_json["label_generator"]["type"];

  if (labelType == "default") { return nullptr; }
  else if (labelType == "ClassLabelGenerator") { return std::make_shared<ClassLabelGenerator>(nn_cfg_json["label_generator"]); }
  else {
    std::cout << "label generator type is not specified\n";
  }
}
