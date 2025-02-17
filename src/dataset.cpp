#include "dataset.h"

TextDataset::TextDataset(std::vector<Batch>& batches): _batches(batches) {
  for (size_t bi {0}; bi < _batches.size(); ++bi) {
    for (size_t pi {0}; pi < _batches[bi].size(); ++pi) {
      _data.push_back(_batches[bi][pi]);
    }
  }
}

torch::data::Example<> TextDataset::get(size_t index) {
  if (index >= _data.size()) {
    throw std::out_of_range("Index is out of bounds");
  }
  return {_data[index].first, _data[index].second};
}

std::optional<size_t> TextDataset::size() const {
  return _data.size();
}