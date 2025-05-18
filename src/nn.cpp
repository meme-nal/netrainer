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
    nn_cfg_json["start_from_epoch"].get<std::string>(),
    nn_cfg_json["train_part"].get<int>(),
    nn_cfg_json["test_part"].get<int>(),
    nn_cfg_json["mbatch_size"].get<size_t>(),
    torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);  
}


std::shared_ptr<torch::optim::Optimizer> loadOptimizer(const std::string& nn_cfg_path, Net& model) {
  json nn_cfg_json;
  std::ifstream nn_cfg_file(nn_cfg_path);

  if (!nn_cfg_file.is_open()) {
    std::cerr << "Could not open the file: " << nn_cfg_path << std::endl;
  }

  nn_cfg_file >> nn_cfg_json;
  nn_cfg_file.close();

  float lr = nn_cfg_json["optimizer"]["lr"].get<float>();
  std::string optimType = nn_cfg_json["optimizer"]["type"].get<std::string>();

       if (optimType == "SGD")  { return std::make_shared<torch::optim::SGD>(model->parameters(), lr); }
  else if (optimType == "Adam") { return std::make_shared<torch::optim::Adam>(model->parameters(), lr); }
  else {
    throw std::invalid_argument("Incorrect Optimizer type");
  }
}


std::shared_ptr<torch::nn::Module> loadCriterion(const std::string& nn_cfg_path) {
  json nn_cfg_json;
  std::ifstream nn_cfg_file(nn_cfg_path);

  if (!nn_cfg_file.is_open()) {
    std::cerr << "Could not open the file: " << nn_cfg_path << std::endl;
  }

  nn_cfg_file >> nn_cfg_json;
  nn_cfg_file.close();

  std::string costType = nn_cfg_json["arch"]["final"]["cost"];

       if (costType == "MSE")          { return std::make_shared<MSELossLayer>(); } 
  else if (costType == "MAE")          { return std::make_shared<MAELossLayer>(); }
  else if (costType == "CrossEntropy") { return std::make_shared<CrossEntropyLossLayer>(); }
  else {
    throw std::invalid_argument("Incorrect criterion type");
  }
}

// TODO: implement in NetImpl::NetImpl(json&) function
/*
std::shared_ptr<BaseLayer> getLayer(const std::string& layerName, json& layerJson) {
  // COMMON LAYERS //
       if (layerJson["type"] == "dense")   { return std::make_shared<DenseLayer>(layerName, layerJson); }
  else if (layerJson["type"] == "conv")    { return std::make_shared<ConvLayer>(layerName, layerJson); }
  else if (layerJson["type"] == "pooling") { return std::make_shared<PoolingLayer>(layerName, layerJson); }
  
  // AUXILIARY LAYERS //
  else if (layerJson["type"] == "upsample") { return std::make_shared<UpsampleLayer>(layerName, layerJson); }
  else if (layerJson["type"] == "flatten") { return std::make_shared<FlattenLayer>(layerName, layerJson); }
  else if (layerJson["type"] == "reshape") { return std::make_shared<ReshapeLayer>(layerName, layerJson); }
  else if (layerJson["type"] == "crop") { return std::make_shared<CropLayer>(layerName, layerJson); }
  else if (layerJson["type"] == "concat") { return std::make_shared<ConcatLayer>(layerName, layerJson); }
  else if (layerJson["type"] == "arith")   { return std::make_shared<ArithmeticLayer>(layerName, layerJson); }
  
  // SPECIAL LAYERS //
  else if (layerJson["type"] == "bn")      { return std::make_shared<BatchNormLayer>(layerName, layerJson); }
  else if (layerJson["type"] == "dropout") { return std::make_shared<DropoutLayer>(layerName, layerJson); }

  else {
    throw std::invalid_argument("Incorrect layer type");
  }
}
*/


std::vector<std::string> NetImpl::findInputsByOutput(const std::string& output, json& nn_arch_cfg) const {
  for (auto& layer : nn_arch_cfg.items()) {
    if (layer.key() != "final") {
      if (layer.value()["output"].get<std::string>() == output) {
        return layer.value()["inputs"].get<std::vector<std::string>>();
      }
    }
  }
}


std::string NetImpl::findLayerByOutput(const std::string& output, json& nn_arch_cfg) const {
  for (auto& layer : nn_arch_cfg.items()) {
    if (layer.key() != "final") {
      if (layer.value()["output"].get<std::string>() == output) {
        return layer.key();
      }
    }
  }
}


// TODO: implement forward pass through graph architecture
NetImpl::NetImpl(json& nn_arch_cfg) {
  _net_configuration = nn_arch_cfg;

  /*
  // create all layers
  for (auto& layer : nn_arch_cfg.items()) {
    if (layer.key() != "final") {
      _layers[layer.key()] = getLayer(layer.key(), nn_arch_cfg[layer.key()]);
    }
  }
  */



  /* TMP */
  for (auto& layer : nn_arch_cfg.items()) {
    json& layerJson = nn_arch_cfg[layer.key()];
    std::string layerName = layer.key();

    if (layer.key() != "final") {
      if (layerJson["type"] == "dense") {
        auto linear = std::make_shared<DenseLayer>(layerName, layerJson);
        register_module(layerName, linear);
        _tmp_layers[layerName] = linear;
      } else if (layerJson["type"] == "conv") {
        auto conv = std::make_shared<ConvLayer>(layerName, layerJson);
        register_module(layerName, conv);
        _tmp_layers[layerName] = conv;
      } else if (layerJson["type"] == "flatten") {
        auto flatten = std::make_shared<FlattenLayer>(layerName, layerJson);
        register_module(layerName, flatten);
        _tmp_layers[layerName] = flatten;
      } else if (layerJson["type"] == "pooling") {
        auto pooling = std::make_shared<PoolingLayer>(layerName, layerJson);
        register_module(layerName, pooling);
        _tmp_layers[layerName] = pooling;
      } else if (layerJson["type"] == "crop") {

        auto crop = std::make_shared<CropLayer>(layerName, layerJson);
        register_module(layerName, crop);
        _tmp_layers[layerName] = crop;
      }
      
      /*
      if (layerJson["type"] == "dense") {
        DenseLayer linear(DenseLayerImpl(layerName, layerJson));
        register_module(layerName, linear);
        _tmp_layers[layerName] = std::make_shared<torch::nn::Module>(linear);

      }
      */
    }

    //register_module("layers", layers);
  }

  /* TMP */

  // find all unique tensor names
  //_tensor_names.push_back("main_data");
  
  for (auto& layer : nn_arch_cfg.items()) {
    if (layer.key() != "final") {
      _tensor_names.push_back(layer.value()["output"]);
    }
  }

  
  for (size_t i {0}; i < _tensor_names.size(); ++i) {
    std::vector<std::string> tmp_inputs = findInputsByOutput(_tensor_names[i], nn_arch_cfg);
    for (size_t j {0}; j < tmp_inputs.size(); ++j) {
      _layer_lists[_tensor_names[i]].push_back(tmp_inputs[j]);
    }
  }

  for (const auto& list : _layer_lists) {
    _layer_names.push_back(findLayerByOutput(list.first, nn_arch_cfg));
  }

  //_tensor_names.push_back("main_data");

  /*
  for (const auto& list : _layer_lists) {
    std::cout << list.first << ": ";
    for (const auto& lr : list.second) {
      std::cout << lr << " ";
    }
    std::cout << "\n";
  }
  */
}

// TODO: implement forward pass through graph arch
/*
torch::Tensor NNImpl::calculate_value(std::map<std::string, std::vector<std::string>>& graph, const std::string& vertex, torch::Tensor input_tensor) {
  if (vertex == "main_data") {
    return input_tensor;
  }

  std::vector<std::string> neighbors = graph[vertex];
  std::vector<torch::Tensor> incoming_tensors;

  for (const std::string& neighbor : neighbors) {
    torch::Tensor outgoing_tensor = calculate_value(graph, neighbor, input_tensor);
    incoming_tensors.push_back(outgoing_tensor);
  }

  //return _tmp_layers[findLayerByOutput(vertex, _net_configuration)]->forward(incoming_tensors);
  return _tmp_layers[findLayerByOutput(vertex, _net_configuration)]->forward(incoming_tensors);
}
*/


torch::Tensor NetImpl::forward(torch::Tensor input) {
  //for (const auto& lname : _layers) {
  //  std::cout << lname.first << '\n';
  //}
  //return calculate_value(_layer_lists, "prediction", input);
  
  // Plain implementation
  for (const std::string& layerName : _layer_names) {
    std::vector<torch::Tensor> tmp_inputs = {input};

    if (auto linear = _tmp_layers[layerName]->as<DenseLayer>()) {
      input = linear->forward(tmp_inputs);
    } else if (auto conv = _tmp_layers[layerName]->as<ConvLayer>()) {
      input = conv->forward(tmp_inputs);
    } else if (auto flatten = _tmp_layers[layerName]->as<FlattenLayer>()) {
      input = flatten->forward(tmp_inputs);
    } else if (auto pooling = _tmp_layers[layerName]->as<PoolingLayer>()) {
      input = pooling->forward(tmp_inputs);
    } else if (auto crop = _tmp_layers[layerName]->as<CropLayer>()) {
      input = crop->forward(tmp_inputs);
    }
  }

  return input;
}


void print_arch(const std::shared_ptr<NetImpl>& model) {
  std::cout << "====== Model arch ======\n";
  for (const auto& param : model->named_parameters()) {
    std::cout << param.key() << " : " << param.value().sizes() << "\n\n";
  }
  std::cout << '\n';
}


void print_weights(const torch::nn::Module& module) {
  for (const auto& param : module.named_parameters()) {
    const auto& name = param.key();
    const auto& tensor = param.value();

    std::cout << "Parameter: " << name << std::endl;
    // Check if parameter is a weight matrix (e.g., contains 'weight' in name)
    if (name.find("weight") != std::string::npos) {
      std::cout << "Weight matrix:\n" << tensor << std::endl;
    }
  }

  for (const auto& child : module.children()) {
    print_weights(*child);
  }
}


void print_model_weights(const std::shared_ptr<NetImpl>& model) {
  for (const auto& param : model->named_parameters()) {
    const auto& name = param.key();
    const auto& tensor = param.value();

    std::cout << "Parameter: " << name << std::endl;
    // Check if parameter is a weight matrix (e.g., contains 'weight' in name)
    if (name.find("weight") != std::string::npos) {
      std::cout << "Weight matrix:\n" << tensor << std::endl;
    }
  }

  for (const auto& child : model->children()) {
    print_weights(*child);
  }
}


void print_to_terminal(const std::string& message) {
  int fd = open("/dev/tty", O_WRONLY);
  if (fd == -1) {
    throw std::runtime_error("Can't open the terminal");
  }

  write(fd, message.c_str(), message.size());
  write(fd, "\n", 1);

  close(fd);
}

/*
std::vector<torch::Tensor> NN::parameters() {
  std::vector<torch::Tensor> all_parameters;

  for (const auto& layer_pair : _tmp_layers) {
    auto layer_parameters = layer_pair.second.ptr()->parameters();
    all_parameters.insert(all_parameters.end(),
                          layer_parameters.begin(),
                          layer_parameters.end());
  }

  return all_parameters;
}
*/


size_t count_model_params(const torch::nn::Module& module) {
  size_t num_params = 0;
  for (const auto& param : module.parameters()) {
    num_params += param.numel();
  }
  return num_params;
}


size_t count_parameters(Net& model) {
  size_t total_params = 0;

  //for (const auto& param : module.named_parameters(/*recurse=*/false)) {
  //  total_params += param.value().numel();
  //}

  // Recursively count in submodules
  for (const auto& child : model->named_children()) {
    total_params += count_model_params(*child.value());
  }

  return total_params;
}


Net loadModel(const std::string& nn_cfg_path) {
  json nn_cfg_json;
  std::ifstream nn_cfg_file(nn_cfg_path);

  if (!nn_cfg_file.is_open()) {
    std::cerr << "Could not open the file: " << nn_cfg_path << std::endl;
  }

  nn_cfg_file >> nn_cfg_json;
  nn_cfg_file.close();

  std::cout << "Running on "<< (torch::cuda::is_available() ? "CUDA" : "CPU") << "\n\n";

  Net model = Net(nn_cfg_json["arch"]);
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

       if (labelType == "default")             { return nullptr; }
  else if (labelType == "ClassLabelGenerator") { return std::make_shared<ClassLabelGenerator>(nn_cfg_json["label_generator"]); }
  else {
    throw std::invalid_argument("label generator type is not specified");
  }
}
