#ifndef LAYER_H
#define LAYER_H

#include <torch/torch.h>
#include <memory>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
using json = nlohmann::json;


// COMMON LAYERS
class DenseLayer : public torch::nn::Module {
public:
  DenseLayer(std::string layerName, json layerJson)
    : linear(nullptr) {
    _nonlinearityType = layerJson["nonlinearity"];
    _winit = layerJson["winit"]["type"];
    //linear = register_module(layerName, torch::nn::Linear(torch::nn::LinearOptions(layerJson["shape"][0], layerJson["shape"][1]).bias(layerJson["bias"].get<bool>())));
    linear = register_module("linear", torch::nn::Linear(torch::nn::LinearOptions(layerJson["shape"][0], layerJson["shape"][1]).bias(layerJson["bias"].get<bool>())));
    
    /* WEIGHTS INITIALIZATION */
    if (_winit == "uniform") {
      torch::nn::init::uniform_(linear->weight, layerJson["winit"]["low"].get<double>(), layerJson["winit"]["high"].get<double>());
      if (layerJson["bias"].get<bool>()) {
        //torch::nn::init::uniform_(linear->bias, layerJson["winit"]["low"].get<double>(), layerJson["winit"]["high"].get<double>());
        torch::nn::init::constant_(linear->bias, 1.0);
      }
    }
    else if (_winit == "normal") {
      torch::nn::init::normal_(linear->weight, layerJson["winit"]["mean"].get<double>(), layerJson["winit"]["std"].get<double>());
      if (layerJson["bias"].get<bool>()) {
        //torch::nn::init::normal_(linear->bias, layerJson["winit"]["mean"].get<double>(), layerJson["winit"]["std"].get<double>());
        torch::nn::init::constant_(linear->bias, 1.0);
      }
    }
    else if (_winit == "constant") {
      torch::nn::init::constant_(linear->weight, layerJson["winit"]["value"].get<double>());
      if (layerJson["bias"].get<bool>()) {
        //torch::nn::init::constant_(linear->bias, layerJson["winit"]["value"].get<double>());
        torch::nn::init::constant_(linear->bias, 1.0);
      }
    }
    else if (_winit == "ones") {
      torch::nn::init::ones_(linear->weight);
      if (layerJson["bias"].get<bool>()) {
        //torch::nn::init::ones_(linear->bias);
        torch::nn::init::constant_(linear->bias, 1.0);
      }
    }
    else if (_winit == "zeros") {
      torch::nn::init::zeros_(linear->weight);
      if (layerJson["bias"].get<bool>()) {
        //torch::nn::init::zeros_(linear->bias);
        torch::nn::init::constant_(linear->bias, 1.0);
      }
    }
    else if (_winit == "eye") {
      torch::nn::init::eye_(linear->weight);
      if (layerJson["bias"].get<bool>()) {
        //torch::nn::init::eye_(linear->bias);
        torch::nn::init::constant_(linear->bias, 1.0);
      }
    }
    else if (_winit == "dirac") {
      torch::nn::init::dirac_(linear->weight);
      if (layerJson["bias"].get<bool>()) {
        //torch::nn::init::dirac_(linear->bias);
        torch::nn::init::constant_(linear->bias, 1.0);
      }
    }
    else if (_winit == "xavier_uniform") {
      torch::nn::init::xavier_uniform_(linear->weight);
      if (layerJson["bias"].get<bool>()) {
        //torch::nn::init::xavier_uniform_(linear->bias);
        torch::nn::init::constant_(linear->bias, 1.0);
      }
    }
    else if (_winit == "xavier_normal") {
      torch::nn::init::xavier_normal_(linear->weight);
      if (layerJson["bias"].get<bool>()) {
        //torch::nn::init::xavier_normal_(linear->bias);
        torch::nn::init::constant_(linear->bias, 1.0);
      }
    }
    else if (_winit == "he_uniform") {
      if (layerJson["winit"]["mode"] == "fan_in") {
        torch::nn::init::kaiming_uniform_(linear->weight, layerJson["winit"]["negative_slope"].get<double>(), torch::kFanIn, torch::kReLU);
      } else if (layerJson["winit"]["mode"] == "fan_out") {
        torch::nn::init::kaiming_uniform_(linear->weight, layerJson["winit"]["negative_slope"].get<double>(), torch::kFanOut, torch::kReLU);
      } else {
        throw std::invalid_argument("Incorrect mode in he_uniform weights initialization\nExpected: 'fan_in' or 'fan_out'");
      }

      if (layerJson["bias"].get<bool>()) {
        torch::nn::init::constant_(linear->bias, 1.0);
        /*
        if (layerJson["winit"]["mode"] == "fan_in") {
          torch::nn::init::kaiming_uniform_(linear->bias, layerJson["winit"]["negative_slope"].get<double>(), torch::kFanIn, torch::kReLU);
        } else if (layerJson["winit"]["mode"] == "fan_out") {
          torch::nn::init::kaiming_uniform_(linear->bias, layerJson["winit"]["negative_slope"].get<double>(), torch::kFanOut, torch::kReLU);
        } else {
          throw std::invalid_argument("Incorrect mode in he_uniform weights initialization\nExpected: 'fan_in' or 'fan_out'");
        }
        */
      }
    }
    else if (_winit == "he_normal") {
      if (layerJson["winit"]["mode"] == "fan_in") {
        torch::nn::init::kaiming_normal_(linear->weight, layerJson["winit"]["negative_slope"].get<double>(), torch::kFanIn, torch::kReLU);
      } else if (layerJson["winit"]["mode"] == "fan_out") {
        torch::nn::init::kaiming_normal_(linear->weight, layerJson["winit"]["negative_slope"].get<double>(), torch::kFanOut, torch::kReLU);
      } else {
        throw std::invalid_argument("Incorrect mode in he_normal weights initialization\nExpected: 'fan_in' or 'fan_out'"); 
      }

      if (layerJson["bias"].get<bool>()) {
        torch::nn::init::constant_(linear->bias, 1.0);
        /*
        if (layerJson["winit"]["mode"] == "fan_in") {
          torch::nn::init::kaiming_normal_(linear->bias, layerJson["winit"]["negative_slope"].get<double>(), torch::kFanIn, torch::kReLU);
        } else if (layerJson["winit"]["mode"] == "fan_out") {
          torch::nn::init::kaiming_normal_(linear->bias, layerJson["winit"]["negative_slope"].get<double>(), torch::kFanOut, torch::kReLU);
        } else {
          throw std::invalid_argument("Incorrect mode in he_normal weights initialization\nExpected: 'fan_in' or 'fan_out'"); 
        }
        */
      }
    }
    else if (_winit == "orthogonal") {
      torch::nn::init::orthogonal_(linear->weight);
      if (layerJson["bias"].get<bool>()) {
        //torch::nn::init::orthogonal_(linear->bias);
        torch::nn::init::constant_(linear->bias, 1.0);
      }
    }
    else if (_winit == "sparse") {
      torch::nn::init::sparse_(linear->weight, layerJson["winit"]["sparsity"].get<double>(), layerJson["winit"]["std"]);
      if (layerJson["bias"].get<bool>()) {
        //torch::nn::init::sparse_(linear->bias, layerJson["winit"]["sparsity"].get<double>(), layerJson["winit"]["std"]);
        torch::nn::init::constant_(linear->bias, 1.0);
      }
    }
    else {
      throw std::invalid_argument("Incorrect weights initialization type\nExpected types:\n'uniform'\n'normal'\n'constant'\n'ones'\n'zeros'\n'eye'\n'dirac'\n'xavier_uniform'\n'xavier_normal'\n'he_uniform'\n'he_normal'\n'orthogonal'\n'sparse'");
    }    
  }

  torch::Tensor forward(std::vector<torch::Tensor> inputs) {
         if (_nonlinearityType == "linear")    { return linear->forward(inputs[0]); }
    else if (_nonlinearityType == "sigmoid")   { return torch::sigmoid(linear->forward(inputs[0])); }
    else if (_nonlinearityType == "swish")     { return torch::silu(linear->forward(inputs[0])); }
    else if (_nonlinearityType == "tanh")      { return torch::tanh(linear->forward(inputs[0])); }
    else if (_nonlinearityType == "relu")      { return torch::relu(linear->forward(inputs[0])); }
    else if (_nonlinearityType == "elu")       { return torch::elu(linear->forward(inputs[0])); } 
    else if (_nonlinearityType == "leakyRelu") { return torch::leaky_relu(linear->forward(inputs[0])); } 
    else if (_nonlinearityType == "selu")      { return torch::selu(linear->forward(inputs[0])); } 
    else if (_nonlinearityType == "celu")      { return torch::celu(linear->forward(inputs[0])); } 
    else if (_nonlinearityType == "gelu")      { return torch::gelu(linear->forward(inputs[0])); } 
    else if (_nonlinearityType == "softmax")   { return torch::softmax(linear->forward(inputs[0]), 1); }
    else {
      throw std::invalid_argument("Incorrect nonlinearity type in dense layer\nExpected types:\n'linear'\n'sigmoid'\n'swish'\n'tanh'\n'relu'\n'elu'\n'leakyRelu'\n'selu'\n'celu'\n'gelu'\n'softmax'");
    }
  }

  static std::string getType() {
    return "dense";
  }

private:
  std::string _nonlinearityType;
  std::string _winit;
  torch::nn::Linear linear;
};


class ConvLayer : public torch::nn::Module {
public:
  ConvLayer(std::string layerName, json layerJson)
    : conv(nullptr) {
    _kernel = layerJson["kernel"].get<std::vector<int>>();
    _stride = layerJson["stride"].get<std::vector<int>>();
    _padding = layerJson["padding"].get<std::vector<int>>();
    _dilation = layerJson["dilation"].get<std::vector<int>>();
    _nonlinearityType = layerJson["nonlinearity"];
    _winit = layerJson["winit"]["type"];

    conv = register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(layerJson["in_channels"], layerJson["out_channels"], {_kernel[0], _kernel[1]})
                                                                           .stride({_stride[0], _stride[1]})
                                                                           .padding({_padding[0], _padding[1]})
                                                                           .dilation({_dilation[0], _dilation[1]})
                                                                           .bias(layerJson["bias"].get<bool>())
                                                                           .groups(layerJson["groups"])));
    /* WEIGHTS INITIALIZATION */
    if (_winit == "uniform") {
      torch::nn::init::uniform_(conv->weight, layerJson["winit"]["low"].get<double>(), layerJson["winit"]["high"].get<double>());
      if (layerJson["bias"].get<bool>()) {
        //torch::nn::init::uniform_(conv->bias, layerJson["winit"]["low"].get<double>(), layerJson["winit"]["high"].get<double>());
        torch::nn::init::constant_(conv->bias, 1.0);
      }
    }
    else if (_winit == "normal") {
      torch::nn::init::normal_(conv->weight, layerJson["winit"]["mean"].get<double>(), layerJson["winit"]["std"].get<double>());
      if (layerJson["bias"].get<bool>()) {
        //torch::nn::init::normal_(conv->bias, layerJson["winit"]["mean"].get<double>(), layerJson["winit"]["std"].get<double>());
        torch::nn::init::constant_(conv->bias, 1.0);
      }
    }
    else if (_winit == "constant") {
      torch::nn::init::constant_(conv->weight, layerJson["winit"]["value"].get<double>());
      if (layerJson["bias"].get<bool>()) {
        //torch::nn::init::constant_(conv->bias, layerJson["winit"]["value"].get<double>());
        torch::nn::init::constant_(conv->bias, 1.0);
      }
    }
    else if (_winit == "ones") {
      torch::nn::init::ones_(conv->weight);
      if (layerJson["bias"].get<bool>()) {
        //torch::nn::init::ones_(conv->bias);
        torch::nn::init::constant_(conv->bias, 1.0);
      }
    }
    else if (_winit == "zeros") {
      torch::nn::init::zeros_(conv->weight);
      if (layerJson["bias"].get<bool>()) {
        //torch::nn::init::zeros_(conv->bias);
        torch::nn::init::constant_(conv->bias, 1.0);
      }
    }
    else if (_winit == "eye") {
      torch::nn::init::eye_(conv->weight);
      if (layerJson["bias"].get<bool>()) {
        //torch::nn::init::eye_(conv->bias);
        torch::nn::init::constant_(conv->bias, 1.0);
      }
    }
    else if (_winit == "dirac") {
      torch::nn::init::dirac_(conv->weight);
      if (layerJson["bias"].get<bool>()) {
        //torch::nn::init::dirac_(conv->bias);
        torch::nn::init::constant_(conv->bias, 1.0);
      }
    }
    else if (_winit == "xavier_uniform") {
      torch::nn::init::xavier_uniform_(conv->weight);
      if (layerJson["bias"].get<bool>()) {
        //torch::nn::init::xavier_uniform_(conv->bias);
        torch::nn::init::constant_(conv->bias, 1.0);
      }
    }
    else if (_winit == "xavier_normal") {
      torch::nn::init::xavier_normal_(conv->weight);
      if (layerJson["bias"].get<bool>()) {
        //torch::nn::init::xavier_normal_(conv->bias);
        torch::nn::init::constant_(conv->bias, 1.0);
      }
    }
    else if (_winit == "he_uniform") {
      if (layerJson["winit"]["mode"] == "fan_in") {
        torch::nn::init::kaiming_uniform_(conv->weight, layerJson["winit"]["negative_slope"].get<double>(), torch::kFanIn, torch::kReLU);
      } else if (layerJson["winit"]["mode"] == "fan_out") {
        torch::nn::init::kaiming_uniform_(conv->weight, layerJson["winit"]["negative_slope"].get<double>(), torch::kFanOut, torch::kReLU);
      } else {
        throw std::invalid_argument("Incorrect mode in he_uniform weights initialization\nExpected: 'fan_in' or 'fan_out'");
      }

      if (layerJson["bias"].get<bool>()) {
        torch::nn::init::constant_(conv->bias, 1.0);
        /*
        if (layerJson["winit"]["mode"] == "fan_in") {
          torch::nn::init::kaiming_uniform_(conv->bias, layerJson["winit"]["negative_slope"].get<double>(), torch::kFanIn, torch::kReLU);
        } else if (layerJson["winit"]["mode"] == "fan_out") {
          torch::nn::init::kaiming_uniform_(conv->bias, layerJson["winit"]["negative_slope"].get<double>(), torch::kFanOut, torch::kReLU);
        } else {
          throw std::invalid_argument("Incorrect mode in he_uniform weights initialization\nExpected: 'fan_in' or 'fan_out'");
        }
        */
      }
    }
    else if (_winit == "he_normal") {
      if (layerJson["winit"]["mode"] == "fan_in") {
        torch::nn::init::kaiming_normal_(conv->weight, layerJson["winit"]["negative_slope"].get<double>(), torch::kFanIn, torch::kReLU);
      } else if (layerJson["winit"]["mode"] == "fan_out") {
        torch::nn::init::kaiming_normal_(conv->weight, layerJson["winit"]["negative_slope"].get<double>(), torch::kFanOut, torch::kReLU);
      } else {
        throw std::invalid_argument("Incorrect mode in he_normal weights initialization\nExpected: 'fan_in' or 'fan_out'"); 
      }

      if (layerJson["bias"].get<bool>()) {
        torch::nn::init::constant_(conv->bias, 1.0);
        /*
        if (layerJson["winit"]["mode"] == "fan_in") {
          torch::nn::init::kaiming_normal_(conv->bias, layerJson["winit"]["negative_slope"].get<double>(), torch::kFanIn, torch::kReLU);
        } else if (layerJson["winit"]["mode"] == "fan_out") {
          torch::nn::init::kaiming_normal_(conv->bias, layerJson["winit"]["negative_slope"].get<double>(), torch::kFanOut, torch::kReLU);
        } else {
          throw std::invalid_argument("Incorrect mode in he_normal weights initialization\nExpected: 'fan_in' or 'fan_out'"); 
        }
        */
      }
    }
    else if (_winit == "orthogonal") {
      torch::nn::init::orthogonal_(conv->weight);
      if (layerJson["bias"].get<bool>()) {
        //torch::nn::init::orthogonal_(conv->bias);
        torch::nn::init::constant_(conv->bias, 1.0);
      }
    }
    else if (_winit == "sparse") {
      torch::nn::init::sparse_(conv->weight, layerJson["winit"]["sparsity"].get<double>(), layerJson["winit"]["std"]);
      if (layerJson["bias"].get<bool>()) {
        //torch::nn::init::sparse_(conv->bias, layerJson["winit"]["sparsity"].get<double>(), layerJson["winit"]["std"]);
        torch::nn::init::constant_(conv->bias, 1.0);
      }
    }
    else {
      throw std::invalid_argument("Incorrect weights initialization type\nExpected types:\n'uniform'\n'normal'\n'constant'\n'ones'\n'zeros'\n'eye'\n'dirac'\n'xavier_uniform'\n'xavier_normal'\n'he_uniform'\n'he_normal'\n'orthogonal'\n'sparse'");
    }
  }

  torch::Tensor forward(std::vector<torch::Tensor> inputs) {
         if (_nonlinearityType == "linear")    { return conv->forward(inputs[0]); }
    else if (_nonlinearityType == "sigmoid")   { return torch::sigmoid(conv->forward(inputs[0])); }
    else if (_nonlinearityType == "swish")     { return torch::silu(conv->forward(inputs[0])); }
    else if (_nonlinearityType == "tanh")      { return torch::tanh(conv->forward(inputs[0])); }
    else if (_nonlinearityType == "relu")      { return torch::relu(conv->forward(inputs[0])); }
    else if (_nonlinearityType == "elu")       { return torch::elu(conv->forward(inputs[0])); } 
    else if (_nonlinearityType == "leakyRelu") { return torch::leaky_relu(conv->forward(inputs[0])); } 
    else if (_nonlinearityType == "selu")      { return torch::selu(conv->forward(inputs[0])); } 
    else if (_nonlinearityType == "celu")      { return torch::celu(conv->forward(inputs[0])); } 
    else if (_nonlinearityType == "gelu")      { return torch::gelu(conv->forward(inputs[0])); } 
    else if (_nonlinearityType == "softmax")   { return torch::softmax(conv->forward(inputs[0]), 1); }
    else {
      throw std::invalid_argument("Incorrect nonlinearity type in conv layer\nExpected types:\n'linear'\n'sigmoid'\n'swish'\n'tanh'\n'relu'\n'elu'\n'leakyRelu'\n'selu'\n'celu'\n'gelu'\n'softmax'");
    }
  }

  static std::string getType() {
    return "conv";
  }

private:
  std::vector<int> _kernel;
  std::vector<int> _stride;
  std::vector<int> _padding;
  std::vector<int> _dilation;
  std::string _nonlinearityType;
  std::string _winit;
  torch::nn::Conv2d conv;
};


class PoolingLayer : public torch::nn::Module {
public:
  PoolingLayer(std::string layerName, json layerJson)
    : max_pool(nullptr)
    , avg_pool(nullptr)
    , global_max_pool(nullptr)
    , global_avg_pool(nullptr) {
    _kernel = layerJson["kernel"].get<std::vector<int>>();
    _stride = layerJson["stride"].get<std::vector<int>>();
    _subtype = layerJson["subtype"].get<std::string>();
    _nonlinearityType = layerJson["nonlinearity"];

    if (_subtype == "max") {
      if (_kernel[0] == -1 && _kernel[1] == -1) {
        global_max_pool = register_module("global_max_pooling", torch::nn::AdaptiveMaxPool2d(torch::nn::AdaptiveMaxPool2dOptions({1, 1})));
      } else {
        max_pool = register_module("max_pooling", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({_kernel[0], _kernel[1]}).stride({_stride[0], _stride[1]})));
      }
      
    } else if (_subtype == "avg") {
      if (_kernel[0] == -1 && _kernel[1] == -1) {
        global_avg_pool = register_module("global_avg_pooling", torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
      } else {
        avg_pool = register_module("avg_pooling", torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions({_kernel[0], _kernel[1]}).stride({_stride[0], _stride[1]})));
      }
      
    }
  }

  torch::Tensor forward(std::vector<torch::Tensor> inputs) {
    if (_subtype == "max") {
      if (_kernel[0] == -1 && _kernel[1] == -1) {
          if (_nonlinearityType == "linear")    { return global_max_pool->forward(inputs[0]); }
        else if (_nonlinearityType == "sigmoid")   { return torch::sigmoid(global_max_pool->forward(inputs[0])); }
        else if (_nonlinearityType == "swish")     { return torch::silu(global_max_pool->forward(inputs[0])); }
        else if (_nonlinearityType == "tanh")      { return torch::tanh(global_max_pool->forward(inputs[0])); }
        else if (_nonlinearityType == "relu")      { return torch::relu(global_max_pool->forward(inputs[0])); }
        else if (_nonlinearityType == "elu")       { return torch::elu(global_max_pool->forward(inputs[0])); } 
        else if (_nonlinearityType == "leakyRelu") { return torch::leaky_relu(global_max_pool->forward(inputs[0])); } 
        else if (_nonlinearityType == "selu")      { return torch::selu(global_max_pool->forward(inputs[0])); } 
        else if (_nonlinearityType == "celu")      { return torch::celu(global_max_pool->forward(inputs[0])); } 
        else if (_nonlinearityType == "gelu")      { return torch::gelu(global_max_pool->forward(inputs[0])); } 
        else if (_nonlinearityType == "softmax")   { return torch::softmax(global_max_pool->forward(inputs[0]), 1); }
        else {
          throw std::invalid_argument("Incorrect nonlinearity type in pooling layer\nExpected types:\n'linear'\n'sigmoid'\n'swish'\n'tanh'\n'relu'\n'elu'\n'leakyRelu'\n'selu'\n'celu'\n'gelu'\n'softmax'");
        }
      } else {
          if (_nonlinearityType == "linear")    { return max_pool->forward(inputs[0]); }
        else if (_nonlinearityType == "sigmoid")   { return torch::sigmoid(max_pool->forward(inputs[0])); }
        else if (_nonlinearityType == "swish")     { return torch::silu(max_pool->forward(inputs[0])); }
        else if (_nonlinearityType == "tanh")      { return torch::tanh(max_pool->forward(inputs[0])); }
        else if (_nonlinearityType == "relu")      { return torch::relu(max_pool->forward(inputs[0])); }
        else if (_nonlinearityType == "elu")       { return torch::elu(max_pool->forward(inputs[0])); } 
        else if (_nonlinearityType == "leakyRelu") { return torch::leaky_relu(max_pool->forward(inputs[0])); } 
        else if (_nonlinearityType == "selu")      { return torch::selu(max_pool->forward(inputs[0])); } 
        else if (_nonlinearityType == "celu")      { return torch::celu(max_pool->forward(inputs[0])); } 
        else if (_nonlinearityType == "gelu")      { return torch::gelu(max_pool->forward(inputs[0])); } 
        else if (_nonlinearityType == "softmax")   { return torch::softmax(max_pool->forward(inputs[0]), 1); }
        else {
          throw std::invalid_argument("Incorrect nonlinearity type in pooling layer\nExpected types:\n'linear'\n'sigmoid'\n'swish'\n'tanh'\n'relu'\n'elu'\n'leakyRelu'\n'selu'\n'celu'\n'gelu'\n'softmax'");
        }
      }
    } else if (_subtype == "avg") {
        if (_kernel[0] == -1 && _kernel[1] == -1) {
          if (_nonlinearityType == "linear")    { return global_avg_pool->forward(inputs[0]); }
        else if (_nonlinearityType == "sigmoid")   { return torch::sigmoid(global_avg_pool->forward(inputs[0])); }
        else if (_nonlinearityType == "swish")     { return torch::silu(global_avg_pool->forward(inputs[0])); }
        else if (_nonlinearityType == "tanh")      { return torch::tanh(global_avg_pool->forward(inputs[0])); }
        else if (_nonlinearityType == "relu")      { return torch::relu(global_avg_pool->forward(inputs[0])); }
        else if (_nonlinearityType == "elu")       { return torch::elu(global_avg_pool->forward(inputs[0])); } 
        else if (_nonlinearityType == "leakyRelu") { return torch::leaky_relu(global_avg_pool->forward(inputs[0])); } 
        else if (_nonlinearityType == "selu")      { return torch::selu(global_avg_pool->forward(inputs[0])); } 
        else if (_nonlinearityType == "celu")      { return torch::celu(global_avg_pool->forward(inputs[0])); } 
        else if (_nonlinearityType == "gelu")      { return torch::gelu(global_avg_pool->forward(inputs[0])); } 
        else if (_nonlinearityType == "softmax")   { return torch::softmax(global_avg_pool->forward(inputs[0]).squeeze({2, 3}), 1); }
        else {
          throw std::invalid_argument("Incorrect nonlinearity type in pooling layer\nExpected types:\n'linear'\n'sigmoid'\n'swish'\n'tanh'\n'relu'\n'elu'\n'leakyRelu'\n'selu'\n'celu'\n'gelu'\n'softmax'");
        }
      } else {
          if (_nonlinearityType == "linear")    { return avg_pool->forward(inputs[0]); }
        else if (_nonlinearityType == "sigmoid")   { return torch::sigmoid(avg_pool->forward(inputs[0])); }
        else if (_nonlinearityType == "swish")     { return torch::silu(avg_pool->forward(inputs[0])); }
        else if (_nonlinearityType == "tanh")      { return torch::tanh(avg_pool->forward(inputs[0])); }
        else if (_nonlinearityType == "relu")      { return torch::relu(avg_pool->forward(inputs[0])); }
        else if (_nonlinearityType == "elu")       { return torch::elu(avg_pool->forward(inputs[0])); } 
        else if (_nonlinearityType == "leakyRelu") { return torch::leaky_relu(avg_pool->forward(inputs[0])); } 
        else if (_nonlinearityType == "selu")      { return torch::selu(avg_pool->forward(inputs[0])); } 
        else if (_nonlinearityType == "celu")      { return torch::celu(avg_pool->forward(inputs[0])); } 
        else if (_nonlinearityType == "gelu")      { return torch::gelu(avg_pool->forward(inputs[0])); } 
        else if (_nonlinearityType == "softmax")   { return torch::softmax(avg_pool->forward(inputs[0]).squeeze({2, 3}), 1); }
        else {
          throw std::invalid_argument("Incorrect nonlinearity type in pooling layer\nExpected types:\n'linear'\n'sigmoid'\n'swish'\n'tanh'\n'relu'\n'elu'\n'leakyRelu'\n'selu'\n'celu'\n'gelu'\n'softmax'");
        }
      }    
    } else {
      throw std::invalid_argument("Incorrect pooling type\nExpected types: 'max' or 'avg'");
    }
  }

  static std::string getType() {
    return "pooling";
  }

private:
  std::vector<int> _kernel;
  std::vector<int> _stride;
  std::string _subtype;
  std::string _nonlinearityType;
  torch::nn::MaxPool2d max_pool;
  torch::nn::AvgPool2d avg_pool;

  // global poolings
  torch::nn::AdaptiveMaxPool2d global_max_pool;
  torch::nn::AdaptiveAvgPool2d global_avg_pool;
};


// AUXILIARY LAYERS
class UpsampleLayer : public torch::nn::Module {
public:
  UpsampleLayer(std::string layerName, json layerJson) {
    _mode = layerJson["mode"].get<std::string>();
    _target_size = layerJson["target_size"].get<std::vector<int64_t>>();
  }

  torch::Tensor forward(std::vector<torch::Tensor> inputs) {
         if (_mode == "nearest") { return torch::nn::functional::interpolate(inputs[0], torch::nn::functional::InterpolateFuncOptions().size(_target_size).mode(torch::kNearest).align_corners(true)); } 
    else if (_mode == "linear") { return torch::nn::functional::interpolate(inputs[0], torch::nn::functional::InterpolateFuncOptions().size(_target_size).mode(torch::kLinear).align_corners(true)); }
    else if (_mode == "bilinear") { return torch::nn::functional::interpolate(inputs[0], torch::nn::functional::InterpolateFuncOptions().size(_target_size).mode(torch::kBilinear).align_corners(true)); }
    else if (_mode == "bicubic") { return torch::nn::functional::interpolate(inputs[0], torch::nn::functional::InterpolateFuncOptions().size(_target_size).mode(torch::kBicubic).align_corners(true)); }
    else if (_mode == "trilinear") { return torch::nn::functional::interpolate(inputs[0], torch::nn::functional::InterpolateFuncOptions().size(_target_size).mode(torch::kTrilinear).align_corners(true)); }
    else if (_mode == "area") { return torch::nn::functional::interpolate(inputs[0], torch::nn::functional::InterpolateFuncOptions().size(_target_size).mode(torch::kArea).align_corners(true)); }
    else if (_mode == "nearest-exact") { return torch::nn::functional::interpolate(inputs[0], torch::nn::functional::InterpolateFuncOptions().size(_target_size).mode(torch::kNearestExact).align_corners(true)); }
    else {
      throw std::invalid_argument("Incorrect upsampling mode\nExpected modes:\n'nearest'\n'linear'\n'bilinear'\n'bicubic'\n'trilinear'\n'area'\n'nearest-exact'");
    }
  }

  static std::string getType() {
    return "upsample";
  }

private:
  std::string          _mode;
  std::vector<int64_t> _target_size;
};


class FlattenLayer : public torch::nn::Module {
public:
  FlattenLayer(std::string layerName, json layerJson)
    : flatten(nullptr) {
    flatten = torch::nn::Flatten();
  }

  torch::Tensor forward(std::vector<torch::Tensor> inputs) {
    return flatten->forward(inputs[0]);
  }

  static std::string getType() {
    return "flatten";
  }

private:
  torch::nn::Flatten flatten;
};


class ReshapeLayer : public torch::nn::Module {
public:
  ReshapeLayer(std::string layerName, json layerJson) {
    _shape = layerJson["shape"].get<std::vector<int>>();
  }

  torch::Tensor forward(std::vector<torch::Tensor> inputs) {
    return inputs[0].view({-1, _shape[0], _shape[1], _shape[2]});
  }

  static std::string getType() {
    return "reshape";
  }

private:
  std::vector<int> _shape;
};


class CropLayer : public torch::nn::Module {
public:
  CropLayer(std::string layerName, json layerJson) {
    _channels = layerJson["channels"].get<std::vector<int>>();
  }

  torch::Tensor forward(std::vector<torch::Tensor> inputs) {
    int start = _channels[0];
    int end = _channels[1];

    if (start > end) {
      throw std::invalid_argument("first index in channels must be less than last");
    } else if (end > inputs[0].sizes()[1]) {
      throw std::invalid_argument("last index in channels is out of range");
    }

    torch::Tensor croppedTensor = inputs[0].slice(1, start, end);

    return croppedTensor; 
  }

  static std::string getType() {
    return "crop";
  }

private:
  std::vector<int> _channels;
};


class ConcatLayer : public torch::nn::Module {
public:
  ConcatLayer(std::string layerName, json layerJson) {}

  torch::Tensor forward(std::vector<torch::Tensor> inputs) {
    if (inputs.size() == 1) {
      throw std::invalid_argument("inputs must have at least 2 items");
    }

    torch::Tensor result = torch::cat({inputs[0], inputs[1]}, 1);

    if (inputs.size() > 2) {
      for (size_t i {2}; i < inputs.size(); ++i) {
        result = torch::cat({result, inputs[i]}, 1);
      }
    } else {
      return result;
    }
  }

  static std::string getType() {
    return "concat";
  }
};


class ArithmeticLayer : public torch::nn::Module {
public:
  ArithmeticLayer(std::string layerName, json layerJson) {
    _op_type = layerJson["op_type"].get<std::string>();
  }

  torch::Tensor forward(std::vector<torch::Tensor> inputs) {
    if (_op_type == "sum") {
      torch::Tensor sum = inputs[0];
      for (size_t i {1}; i < inputs.size(); ++i) {
        sum = sum + inputs[i];
      }
      return sum;
    } else if (_op_type == "mul") {
      torch::Tensor mul = inputs[0];
      for (size_t i {1}; i < inputs.size(); ++i) {
        mul = mul * inputs[i];
      }
      return mul;
    } else {
      throw std::invalid_argument("Incorrect operation type in arith layer\nExpected types: 'sum' or 'mul'");
    }
  }

  static std::string getType() {
    return "arith";
  }

private:
  std::string _op_type;
};


// LOSS LAYERS
class MAELossLayer : public torch::nn::Module {
public:
  MAELossLayer() {};

  torch::Tensor forward(std::vector<torch::Tensor> inputs) {
    return mae_loss(inputs[0], inputs[1]); // inputs[0] - prediction, inputs[1] - label
  }

private:
  torch::nn::L1Loss mae_loss {nullptr};
};


class MSELossLayer : public torch::nn::Module {
public:
  MSELossLayer() {};

  torch::Tensor forward(std::vector<torch::Tensor> inputs) {
    return mse_loss(inputs[0], inputs[1]); // inputs[0] - prediction, inputs[1] - label
  }

private:
  torch::nn::MSELoss mse_loss {nullptr};
};


class CrossEntropyLossLayer : public torch::nn::Module {
public:
  CrossEntropyLossLayer() {};

  torch::Tensor forward(std::vector<torch::Tensor> inputs) {
    return ce_loss(inputs[0], inputs[1]); // inputs[0] - prediction, inputs[1] - label
  }

private:
  torch::nn::CrossEntropyLoss ce_loss;
};


// SPECIAL LAYERS
class BatchNormLayer : public torch::nn::Module {
public:
  BatchNormLayer(std::string layerName, json layerJson)
    : _dims(1)
    , _bn1d(nullptr)
    , _bn2d(nullptr) {
    _dims = layerJson["dims"].get<size_t>();

         if (_dims == 1) { _bn1d = torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(layerJson["channels"].get<size_t>())); }
    else if (_dims == 2) { _bn2d = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(layerJson["channels"].get<size_t>())); }
    else {
      throw std::invalid_argument("Incorrect dims number in batchnorm layer\nExpected dims: 1 or 2");
    }
  };

  torch::Tensor forward(std::vector<torch::Tensor> inputs) {
         if (_dims == 1) { return _bn1d->forward(inputs[0]); }
    else if (_dims == 2) { return _bn2d->forward(inputs[0]); }
    else {
      throw std::invalid_argument("Incorrect dims number in batchnorm layer\nExpected dims: 1 or 2");
    }
  }

private:
  size_t _dims;
  torch::nn::BatchNorm1d _bn1d;
  torch::nn::BatchNorm2d _bn2d;
};


class DropoutLayer : public torch::nn::Module {
public:
  DropoutLayer(std::string layerName, json layerJson)
    : _prob(0.5f)
    , _dims(1)
    , dropout1d(nullptr)
    , dropout2d(nullptr) {
    _prob = layerJson["prob"].get<float>();
    _dims = layerJson["dims"].get<size_t>();
    
         if (_dims == 1) { dropout1d = torch::nn::Dropout(torch::nn::DropoutOptions(_prob)); }
    else if (_dims == 2) { dropout2d = torch::nn::Dropout2d(torch::nn::Dropout2dOptions(_prob)); }
    else {
      throw std::invalid_argument("Incorrect dims number in dropout layer\nExpected dims: 1 or 2");
    }
  }

  torch::Tensor forward(std::vector<torch::Tensor> inputs) {
         if (_dims == 1) { return dropout1d->forward(inputs[0]); }
    else if (_dims == 2) { return dropout2d->forward(inputs[0]); }
    else {
      throw std::invalid_argument("Incorrect dims number in dropout layer\nExpected dims: 1 or 2");
    }
  }

private:
  float _prob;
  size_t _dims;
  torch::nn::Dropout dropout1d;
  torch::nn::Dropout2d dropout2d;
};

#endif // LAYER_H