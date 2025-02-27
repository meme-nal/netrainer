#ifndef LAYER_H
#define LAYER_H

#include <torch/torch.h>
#include <memory>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

class BaseLayer : public torch::nn::Module {
public:
  virtual ~BaseLayer() = default;
  virtual torch::Tensor forward(torch::Tensor input, torch::Tensor label = torch::tensor(0)) = 0;
};

// COMMON LAYERS
class DenseLayer : public BaseLayer {
public:
  DenseLayer(const std::string& layerName, json& layerJson) {
    _nonlinearityType = layerJson["nonlinearity"];
    linear = register_module(layerName, torch::nn::Linear(torch::nn::LinearOptions(layerJson["shape"][0], layerJson["shape"][1]).bias(layerJson["bias"].get<bool>())));
  }

  torch::Tensor forward(torch::Tensor input, torch::Tensor label) override {
    if (_nonlinearityType == "Linear") { return linear->forward(input); }
    else if (_nonlinearityType == "Sigmoid") { return torch::sigmoid(linear->forward(input)); }
    else if (_nonlinearityType == "Swish") { return torch::silu(linear->forward(input)); }
    else if (_nonlinearityType == "Tanh") { return torch::tanh(linear->forward(input)); }
    else if (_nonlinearityType == "ReLU") { return torch::relu(linear->forward(input)); }
    else if (_nonlinearityType == "ELU") { return torch::elu(linear->forward(input)); } 
    else if (_nonlinearityType == "LeakyReLU") { return torch::leaky_relu(linear->forward(input)); } 
    else if (_nonlinearityType == "SELU") { return torch::selu(linear->forward(input)); } 
    else if (_nonlinearityType == "CELU") { return torch::celu(linear->forward(input)); } 
    else if (_nonlinearityType == "GELU") { return torch::gelu(linear->forward(input)); } 
    else if (_nonlinearityType == "Softmax") { return torch::softmax(linear->forward(input), 1); }
    else {
      std::cout << "Incorrect nonlinearity type!\n";
    }
  }

private:
  std::string _nonlinearityType;
  torch::nn::Linear linear {nullptr};
};


class ConvLayer : public BaseLayer {
public:
  ConvLayer(const std::string& layerName, json& layerJson) {
    _kernel = layerJson["kernel"].get<std::vector<int>>();
    _stride = layerJson["stride"].get<std::vector<int>>();
    _padding = layerJson["padding"].get<std::vector<int>>();
    _dilation = layerJson["dilation"].get<std::vector<int>>();
    _nonlinearityType = layerJson["nonlinearity"];

    conv = register_module(layerName, torch::nn::Conv2d(torch::nn::Conv2dOptions(layerJson["in_channels"], layerJson["out_channels"], {_kernel[0], _kernel[1]})
                                                                           .stride({_stride[0], _stride[1]})
                                                                           .padding({_padding[0], _padding[1]})
                                                                           .dilation({_dilation[0], _dilation[1]})
                                                                           .bias(layerJson["bias"].get<bool>())
                                                                           .groups(layerJson["groups"])));
  }

  torch::Tensor forward(torch::Tensor input, torch::Tensor label) override {
    if (_nonlinearityType == "Linear") { return conv->forward(input); }
    else if (_nonlinearityType == "Sigmoid") { return torch::sigmoid(conv->forward(input)); }
    else if (_nonlinearityType == "Swish") { return torch::silu(conv->forward(input)); }
    else if (_nonlinearityType == "Tanh") { return torch::tanh(conv->forward(input)); }
    else if (_nonlinearityType == "ReLU") { return torch::relu(conv->forward(input)); }
    else if (_nonlinearityType == "ELU") { return torch::elu(conv->forward(input)); } 
    else if (_nonlinearityType == "LeakyReLU") { return torch::leaky_relu(conv->forward(input)); } 
    else if (_nonlinearityType == "SELU") { return torch::selu(conv->forward(input)); } 
    else if (_nonlinearityType == "CELU") { return torch::celu(conv->forward(input)); } 
    else if (_nonlinearityType == "GELU") { return torch::gelu(conv->forward(input)); } 
    else if (_nonlinearityType == "Softmax") { return torch::softmax(conv->forward(input), 1); }
    else {
      std::cout << "Incorrect nonlinearity type!\n";
    }
  }

private:
  std::vector<int> _kernel;
  std::vector<int> _stride;
  std::vector<int> _padding;
  std::vector<int> _dilation;
  std::string _nonlinearityType;
  torch::nn::Conv2d conv {nullptr};
};


class PoolingLayer : public BaseLayer {
public:
  PoolingLayer(const std::string& layerName, json& layerJson) {
    _kernel = layerJson["kernel"].get<std::vector<int>>();
    _stride = layerJson["stride"].get<std::vector<int>>();
    _subtype = layerJson["subtype"].get<std::string>();
    _nonlinearityType = layerJson["nonlinearity"];

    if (_subtype == "max") {
      max_pool = register_module(layerName, torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({_kernel[0], _kernel[1]})
                .stride({_stride[0], _stride[1]})));
    } else if (_subtype == "avg") {
      avg_pool = register_module(layerName, torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions({_kernel[0], _kernel[1]})
                .stride({_stride[0], _stride[1]})));
    }
  }

  torch::Tensor forward(torch::Tensor input, torch::Tensor label) override {
    if (_subtype == "max") {
      if (_nonlinearityType == "Linear") { return max_pool->forward(input); }
      else if (_nonlinearityType == "Sigmoid") { return torch::sigmoid(max_pool->forward(input)); }
      else if (_nonlinearityType == "Swish") { return torch::silu(max_pool->forward(input)); }
      else if (_nonlinearityType == "Tanh") { return torch::tanh(max_pool->forward(input)); }
      else if (_nonlinearityType == "ReLU") { return torch::relu(max_pool->forward(input)); }
      else if (_nonlinearityType == "ELU") { return torch::elu(max_pool->forward(input)); } 
      else if (_nonlinearityType == "LeakyReLU") { return torch::leaky_relu(max_pool->forward(input)); } 
      else if (_nonlinearityType == "SELU") { return torch::selu(max_pool->forward(input)); } 
      else if (_nonlinearityType == "CELU") { return torch::celu(max_pool->forward(input)); } 
      else if (_nonlinearityType == "GELU") { return torch::gelu(max_pool->forward(input)); } 
      else if (_nonlinearityType == "Softmax") { return torch::softmax(max_pool->forward(input), 1); }
      else {
        std::cout << "Incorrect nonlinearity type!\n";
      }
    } else if (_subtype == "avg") {
      if (_nonlinearityType == "Linear") { return avg_pool->forward(input); }
      else if (_nonlinearityType == "Sigmoid") { return torch::sigmoid(avg_pool->forward(input)); }
      else if (_nonlinearityType == "Swish") { return torch::silu(avg_pool->forward(input)); }
      else if (_nonlinearityType == "Tanh") { return torch::tanh(avg_pool->forward(input)); }
      else if (_nonlinearityType == "ReLU") { return torch::relu(avg_pool->forward(input)); }
      else if (_nonlinearityType == "ELU") { return torch::elu(avg_pool->forward(input)); } 
      else if (_nonlinearityType == "LeakyReLU") { return torch::leaky_relu(avg_pool->forward(input)); } 
      else if (_nonlinearityType == "SELU") { return torch::selu(avg_pool->forward(input)); } 
      else if (_nonlinearityType == "CELU") { return torch::celu(avg_pool->forward(input)); } 
      else if (_nonlinearityType == "GELU") { return torch::gelu(avg_pool->forward(input)); } 
      else if (_nonlinearityType == "Softmax") { return torch::softmax(avg_pool->forward(input), 1); }
      else {
        std::cout << "Incorrect nonlinearity type!\n";
      }
    }
  }

private:
  std::vector<int> _kernel;
  std::vector<int> _stride;
  std::string _subtype;
  std::string _nonlinearityType;
  torch::nn::MaxPool2d max_pool {nullptr};
  torch::nn::AvgPool2d avg_pool {nullptr};
};


// AUXILIARY LAYERS
class FlattenLayer : public BaseLayer {
public:
  FlattenLayer(const std::string& layerName, json& layerJson) {
    flatten = register_module(layerName, torch::nn::Flatten());
  }

  torch::Tensor forward(torch::Tensor input, torch::Tensor label) override {
    return flatten->forward(input);
  }

private:
  torch::nn::Flatten flatten {nullptr};
};


class ReshapeLayer : public BaseLayer {
public:
  ReshapeLayer(const std::string& layerName, json& layerJson) {
    _shape = layerJson["shape"].get<std::vector<int>>();
  }

  torch::Tensor forward(torch::Tensor input, torch::Tensor label) {
    return input.view({-1, _shape[0], _shape[1], _shape[2]});
  }

private:
  std::vector<int> _shape;
};

/*
class ArithmeticLayer : public BaseLayer {
public:
  ArithmeticLayer(const std::string& layerName, json& layerJson) {
    _Operationtype = layerJson["Operationtype"].get<std::string>();
  }

  torch::Tensor forward(torch::Tensor input1, torch::Tensor input2) {
    if (_Operationtype == "sum") {
      return (input1 + input2);
    } else if (_Operationtype == "sub") {
      return (input1 - input2);
    } else if (_Operationtype == "mul") {
      return (input1 * input2);
    } else if (_Operationtype == "div") {
      return (input1 / input2);
    } else {
      std::cout << "Incorrect layer type\n";
    }
  }

private:
  std::string _Operationtype;
};
*/

// LOSS LAYERS
class MAELossLayer : public BaseLayer {
public:
  MAELossLayer() = default;

  torch::Tensor forward(torch::Tensor prediction, torch::Tensor label) override {
    return mae_loss(prediction, label);
  }

private:
  torch::nn::L1Loss mae_loss;
};


class MSELossLayer : public BaseLayer {
public:
  MSELossLayer() = default;

  torch::Tensor forward(torch::Tensor prediction, torch::Tensor label) override {
    return mse_loss(prediction, label);
  }

private:
  torch::nn::MSELoss mse_loss;
};


class CrossEntropyLossLayer : public BaseLayer {
public:
  CrossEntropyLossLayer() = default;

  torch::Tensor forward(torch::Tensor prediction, torch::Tensor label) override {
    return ce_loss(prediction, label);
  }

private:
  torch::nn::CrossEntropyLoss ce_loss;
};

#endif // LAYER_H