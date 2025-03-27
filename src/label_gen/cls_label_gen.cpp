#include "cls_label_gen.h"

void ClassLabelGenerator::get_labels_smoothing(torch::Tensor& labels, float eps) {
  int num_cls = labels.sizes()[1];
  for (size_t i {0}; i < labels.sizes()[0]; ++i) {
    for (size_t j {0}; j < num_cls; ++j) {
      if (labels[i][j].item<float>() == 0) {
        labels[i][j] = torch::tensor(eps / (num_cls - 1));
      } else {
        labels[i][j] = torch::tensor(labels[i][j].item<float>() - eps);
      }
    }
  }
}

torch::Tensor ClassLabelGenerator::operator()(torch::Tensor labels) {
  std::string path_to_rules = _label_gen_cfg["rules"].get<std::string>();

  json rules_json;
  std::ifstream rules_file(path_to_rules);

  if (!rules_file.is_open()) {
    std::cerr << "Could not open the file: " << path_to_rules << std::endl;
  }

  rules_file >> rules_json;
  rules_file.close();

  int num_cls = rules_json.size();
  int mbatch_size = labels.sizes()[0];

  torch::Tensor new_labels = torch::zeros({mbatch_size, num_cls});

  for (size_t i {0}; i < mbatch_size; ++i) {
    new_labels[i][labels[i].item<int>()] = torch::tensor(1.0);
  }

  if (_label_gen_cfg["label_smoothing"]["to_use"].get<bool>()) {
    get_labels_smoothing(new_labels, _label_gen_cfg["label_smoothing"]["eps"].get<float>());
  }

  return new_labels;

}