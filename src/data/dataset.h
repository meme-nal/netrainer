#ifndef DATASET_H
#define DATASET_H

#include "batch.h"

#include <iostream>
#include <vector>
#include <torch/torch.h>

class TextDataset : public torch::data::datasets::Dataset<TextDataset> {
private:
  std::vector<Batch> _batches;
  std::vector<Data>  _data;

public:
  TextDataset(std::vector<Batch>& batches);
  torch::data::Example<> get(size_t index) override;
  std::optional<size_t> size() const override;
};

#endif // DATASET_H