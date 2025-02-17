#ifndef BATCH_H
#define BATCH_H

#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <cstdint>
#include <memory>

#include <torch/torch.h>
#include <nlohmann/json.hpp>
using json = nlohmann::json;
using namespace torch::indexing;

using Data = std::pair<torch::Tensor, torch::Tensor>;
using Batch = std::vector<std::pair<torch::Tensor, torch::Tensor>>;

std::vector<Batch> loadBatches(const std::string& batches_path);

#endif // BATCH_H