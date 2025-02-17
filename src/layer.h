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

///
// COMMON LAYERS
///
class DenseLayer : public BaseLayer {
public:
  DenseLayer(const std::string& layerName, json& layerJson) {
    _nonlinearityType = layerJson["nonlinearity"];
    linear = register_module(layerName, torch::nn::Linear(torch::nn::LinearOptions(layerJson["shape"][0], layerJson["shape"][1]).bias(layerJson["bias"].get<bool>())));
  }

  torch::Tensor forward(torch::Tensor input, torch::Tensor label) override {
    if (_nonlinearityType == "Linear") {
      return linear->forward(input);
    } else if (_nonlinearityType == "ReLU") {
      return torch::relu(linear->forward(input));
    } else {
      std::cout << "Incorrect nonlinearity type!\n";
    }
  }

private:
  std::string _nonlinearityType;
  torch::nn::Linear linear {nullptr};
};

/*
class ConvLayer : public BaseLayer {
public:
  ConvLayer(const std::string& layerName, json& layerJson) {
    
  }

  torch::Tensor forward(torch::Tensor input) override {

  }

private:
  torch::nn::Conv3d conv3d {nullptr};
  torch::nn::Conv2d conv2d {nullptr};
  torch::nn::Conv1d conv1d {nullptr};
};
*/

///
// AUXILIARY LAYERS
///
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

///
// LOSS LAYERS
///

class MSELossLayer : public BaseLayer {
public:
  MSELossLayer() = default;

  torch::Tensor forward(torch::Tensor prediction, torch::Tensor label) override {
    return mse_loss(prediction, label);
  }

private:
  torch::nn::MSELoss mse_loss;
};

#endif // LAYER_H