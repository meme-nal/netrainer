///
// THIS FILE IS USED TO TRAIN NEURAL NETWORKS ONLY
///

#include "data/batch.h"
#include "data/dataset.h"
#include "inputParser.h"
#include "nn.h"

#include <torch/torch.h>
#include <torch/script.h>

#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <random>


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

  // start from epoch if setted
  Net model = loadModel(path_to_nn_cfg);
  model->to(common._device);

  //print_arch(model);
  model->print_modules();
  std::cout << "\n";

  //std::shared_ptr<torch::optim::Optimizer> optimizer = loadOptimizer(path_to_nn_cfg, model);
  std::shared_ptr<torch::optim::Adam> optimizer = std::make_shared<torch::optim::Adam>(model->parameters(), 0.001);
  //std::shared_ptr<torch::nn::Module> criterion = loadCriterion(path_to_nn_cfg);
  std::shared_ptr<CrossEntropyLossLayer> criterion = std::make_shared<CrossEntropyLossLayer>();
  std::shared_ptr<BaseLabelGenerator> label_generator = loadLabelGenerator(path_to_nn_cfg);

  //std::cout << "Count of parameters: " << count_model_params(model) << "\n\n";
  std::cout << "Count of parameters: " << count_parameters(model) << "\n\n";

  //auto params = model->parameters();
  //for (const auto& param : params) {
  //  std::cout << param << "\n";
  //}

  // split on train and test batches, add metric
  std::vector<Batch> batches = loadBatches(path_to_batches);

  size_t train_part = static_cast<int>((common._train_part / 100.f) * batches.size());
  size_t test_part = batches.size() - train_part;

  auto rng = std::default_random_engine {};
  std::shuffle(std::begin(batches), std::end(batches), rng);

  std::vector<Batch> train_batches;
  for (size_t i {0}; i < train_part; ++i) {
    train_batches.push_back(batches[i]);
  }
  
  std::vector<Batch> test_batches;
  for (size_t i {train_part}; i < train_part + test_part; ++i) {
    test_batches.push_back(batches[i]);
  }

  auto trainDataset = TextDataset(train_batches).map(torch::data::transforms::Stack<>());
  auto testDataset = TextDataset(test_batches).map(torch::data::transforms::Stack<>());

  auto trainDataLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(trainDataset), common._mbatch_size);
  auto testDataLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(testDataset), common._mbatch_size);
  
  for (size_t epoch {0}; epoch < common._num_epochs; ++epoch) {
    //std::cout << "=== Parameters ===\n";
    //auto params = model->parameters();
    //for (const auto& param : params) {
    //  std::cout << param << "\n";
    //  break;
    //}
    //std::cout << "\n";
    model->train();
    for (auto& param : model->parameters()) {
      if (!param.requires_grad()) {
        param.set_requires_grad(true);
      }
    }
    size_t bi_train {0};
    float mean_train_loss {0};
    std::cout << "=== Epoch: " << epoch << " ===\n";
    for (auto& batch : *trainDataLoader) {
      optimizer->zero_grad();

      torch::Tensor features = batch.data.to(common._device);
      torch::Tensor labels   = batch.target.squeeze().to(common._device);

      if (label_generator) {
        labels = (*label_generator)(labels);
      }

      torch::Tensor prediction = model->forward(features).to(common._device);
      std::vector<torch::Tensor> criterion_input = {prediction, labels};
      torch::Tensor loss = criterion->forward(criterion_input);
      //if (auto mae_criterion = criterion->as<MAELossLayer>()) {
      //  std::cout << "MAE\n";
      //  loss = mae_criterion->forward(criterion_input);
      //} else if (auto mse_criterion = criterion->as<MSELossLayer>()) {
      //  std::cout << "MSE\n";
      //  loss = mse_criterion->forward(criterion_input);
      //} else if (auto ce_criterion = criterion->as<CrossEntropyLossLayer>()) {
      //  std::cout << "CE\n";
      //  loss = ce_criterion->forward(criterion_input);
      //}
      
      mean_train_loss += loss.item<float>();
      loss.backward();

      std::cout << "AFTER BACKWARD: \n";
      for (const auto& pair : model->named_parameters()) {
        if (pair.value().grad().defined()) {
          std::cout << pair.key() << ": " << pair.value().grad().sum() << std::endl;
        }
      }
      std::cout << "\n\n";

      //std::cout << "CHECK PARAMS\n";
      //std::vector<torch::Tensor> params_before;
      //for (const auto& param : model->parameters()) {
      //  params_before.push_back(param.clone());
      //}

      optimizer->step();

      //for (size_t i=0; i < model->parameters().size(); ++i) {
      //  auto& param = model->parameters()[i];
      //  auto diff = (param - params_before[i]).abs().mean().item<float>();
      //  std::cout << "Parameter " << i << " change magnitude: " << diff << std::endl;
      //}
      //std::cout << "CHECK PARAMS DONE\n";

      std::cout << "loss | " << bi_train << " | : " << loss.item<float>() << '\n';
      ++bi_train;
    }
    std::cout << "=== Mean Train Loss: " << mean_train_loss / (bi_train + 1) << " ===\n\n";
    
    size_t bi_test {0};
    float mean_test_loss {0};
    std::cout << "=== Epoch: " << epoch << " ===\n";
    for (auto& batch : *testDataLoader) {
      model->eval();
      torch::Tensor features = batch.data.to(common._device);
      torch::Tensor labels   = batch.target.squeeze().to(common._device);
      
      if (label_generator) {
        labels = (*label_generator)(labels);
      }

      torch::Tensor prediction = model->forward(features).to(common._device);
      std::cout << "prediction: " << prediction.sizes() << '\n';
      std::cout << "labels: " << labels.sizes() << '\n';
      std::vector<torch::Tensor> criterion_input = {prediction, labels};
      torch::Tensor loss = criterion->forward(criterion_input);
      //if (auto criterion_tmp = criterion->as<MAELossLayer>()) {
      //  loss = criterion_tmp->forward(criterion_input);
      //} else if (auto criterion_tmp = criterion->as<MSELossLayer>()) {
      //  loss = criterion_tmp->forward(criterion_input);
      //} else if (auto criterion_tmp = criterion->as<CrossEntropyLossLayer>()) {
      //  loss = criterion_tmp->forward(criterion_input);
      //}
      
      std::cout << "loss | " << bi_test << " | : " << loss.item<float>() << '\n';

      mean_test_loss += loss.item<float>();
      ++bi_test;
    }
    std::cout << "=== Mean Test Loss: " << mean_test_loss / (bi_test + 1) << " ===\n\n";

    std::string to_terminal = std::to_string(epoch) + ": " + std::to_string(mean_test_loss / (bi_test + 1));
    print_to_terminal(to_terminal);

    std::string path_to_model_states = common._to_save_path;
    std::filesystem::path model_state_dir = path_to_model_states + "/state_" + std::to_string(epoch);
    std::filesystem::create_directory(model_state_dir);

    std::string path_to_model_state = model_state_dir.string() + "/model_state.pt";
    torch::save(model, path_to_model_state);
  }
  
  return 0;
}
