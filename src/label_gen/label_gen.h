#ifndef LABEL_GEN
#define LABEL_GEN

#include <iostream>
#include <fstream>
#include <string>
#include <torch/torch.h>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

class BaseLabelGenerator {
public:
  virtual ~BaseLabelGenerator() {};

public:
  virtual torch::Tensor operator()(torch::Tensor labels) = 0;
};

#endif // LABEL_GEN