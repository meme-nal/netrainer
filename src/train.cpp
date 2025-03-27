///
// THIS FILE IS USED TO TRAIN NEURAL NETWORKS ONLY
///

#include "data/batch.h"
#include "data/dataset.h"
#include "inputParser.h"
#include "nn.h"

#include <torch/torch.h>

#include <iostream>
#include <vector>
#include <memory>


int main(int argc, char** argv) {
  InputParser input(argc, argv);
  if (input.cmdOptionExists("-h")){
    std::cout << "Usage: nntrainer [OPTION] > [LOG]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -h" << '\t' << "print help information\n";
    std::cout << "  -c" << '\t' << "path to neural net config json file\n";
    std::cout << "  -b" << '\t' << "path to train batches\n"; 
    return 0;
  }
  if (!input.cmdOptionExists("-c") ||
      !input.cmdOptionExists("-b")) {
    return -1;
  }
  std::string path_to_nn_cfg = input.getCmdOption("-c");
  std::string path_to_batches = input.getCmdOption("-b");
  
  CommonOptions common = loadCommonOptions(path_to_nn_cfg);

  std::shared_ptr<NN> model = loadModel(path_to_nn_cfg);
  model->to(common._device);
  model->train();

  std::shared_ptr<torch::optim::Optimizer> optimizer = loadOptimizer(path_to_nn_cfg, model);
  std::shared_ptr<BaseLayer> criterion = loadCriterion(path_to_nn_cfg);
  std::shared_ptr<BaseLabelGenerator> label_generator = loadLabelGenerator(path_to_nn_cfg);

  std::cout << "Count of parameters: " << count_model_params(model) << "\n\n";

  std::vector<Batch> batches = loadBatches(path_to_batches);
  auto trainDataset = TextDataset(batches).map(torch::data::transforms::Stack<>());

  auto dataLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(trainDataset), common._mbatch_size);
  
  for (size_t epoch {0}; epoch < common._num_epochs; ++epoch) {
    size_t bi {0};
    float mean_loss {0};
    std::cout << "=== Epoch: " << epoch << " ===\n";
    for (auto& batch : *dataLoader) {
      optimizer->zero_grad();

      torch::Tensor features = batch.data.to(common._device);
      torch::Tensor labels   = batch.target.squeeze().to(common._device);

      if (label_generator) {
        labels = (*label_generator)(labels);
      }
      
      torch::Tensor prediction = model->forward(features).to(common._device);
      torch::Tensor loss = criterion->forward(prediction, labels);
      
      mean_loss += loss.item<float>();
      loss.backward();
      optimizer->step();

      std::cout << "loss | " << bi << " | : " << loss.item<float>() << '\n';
      ++bi;
    }
    std::cout << "=== Mean Loss: " << mean_loss / (bi + 1) << " ===\n\n";
    std::string path_to_model_states = common._to_save_path;
    std::filesystem::path model_state_dir = path_to_model_states + "/state_" + std::to_string(epoch);
    std::filesystem::create_directory(model_state_dir);

    std::string path_to_model_state = model_state_dir.string() + "/model_state.pt";
    torch::save(model, path_to_model_state);
  }

  return 0;
}
