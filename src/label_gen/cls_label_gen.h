#ifndef CLS_LABEL_GEN
#define CLS_LABEL_GEN

#include "label_gen.h"

class ClassLabelGenerator : public BaseLabelGenerator {
public:
  ClassLabelGenerator(json& label_gen_cfg) { _label_gen_cfg = label_gen_cfg; }

public:
  torch::Tensor operator()(torch::Tensor labels) override;

private:
  json _label_gen_cfg;

private:
  void get_labels_smoothing(torch::Tensor& labels, float eps);
};

#endif // CLS_LABEL_GEN