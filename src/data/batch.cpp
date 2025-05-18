#include "batch.h"

template<typename main_t, typename label_t>
Batch loadBatch(const std::string& batch_path,
    const size_t batch_size,
    std::vector<int64_t> main_data_shape,
    std::vector<int64_t> label_data_shape) {
  size_t main_data_count  = 1;
  size_t label_data_count = 1;
  
  for (const size_t& dim : main_data_shape) {
    main_data_count *= dim;
  } main_data_count *= batch_size;
  
  for (const size_t& dim : label_data_shape) {
    label_data_count *= dim;
  } label_data_count *= batch_size;
  
  std::vector<main_t> main_data(main_data_count);
  std::vector<label_t> label_data(label_data_count);

  std::ifstream batch_file(batch_path, std::ios::binary);
  if (!batch_file) {
    std::cerr << "Error opening file: " << batch_path << "\n\n";
  }
  
  batch_file.read(reinterpret_cast<char*>(main_data.data()), main_data_count * sizeof(main_t));
  batch_file.read(reinterpret_cast<char*>(label_data.data()), label_data_count * sizeof(label_t));

  batch_file.close();

  std::vector<torch::Tensor> labels(batch_size);
  for (size_t i {0}; i < batch_size; ++i) {
    std::vector<label_t> label;
    if (label_data_count / batch_size == 1) {
      label.push_back(label_data[i]);
    } else {
      for (size_t j {0}; j < (label_data_count / batch_size); ++j) {
        label.push_back(label_data[i*(label_data_count / batch_size) + j]);
      }
    }
         if (std::is_same<label_t,float>::value) { labels[i] = torch::tensor(label, torch::dtype(torch::kFloat32)).view(label_data_shape); }
    else if (std::is_same<label_t,int>::value) { labels[i] = torch::tensor(label, torch::dtype(torch::kInt32)).view(label_data_shape); }
    else if (std::is_same<label_t,uint8_t>::value) { labels[i] = torch::tensor(label, torch::dtype(torch::kByte)).view(label_data_shape); }
  }

  std::vector<torch::Tensor> mains(batch_size);
  for (size_t i {0}; i < batch_size; ++i) {
    std::vector<main_t> main;
    for (size_t j {0}; j < (main_data_count / batch_size); ++j) {
      main.push_back(main_data[i*(main_data_count / batch_size) + j]);
    }
         if (std::is_same<main_t,float>::value) { mains[i] = torch::tensor(main, torch::dtype(torch::kFloat32)).view(main_data_shape); }
    else if (std::is_same<main_t,int>::value) { mains[i] = torch::tensor(main, torch::dtype(torch::kFloat32)).view(main_data_shape); }
    //else if (std::is_same<main_t,int>::value) { mains[i] = torch::tensor(main, torch::dtype(torch::kInt32)).view(main_data_shape); }
    else if (std::is_same<main_t,uint8_t>::value) { mains[i] = torch::tensor(main, torch::dtype(torch::kFloat32)).view(main_data_shape); }
    //else if (std::is_same<main_t,uint8_t>::value) { mains[i] = torch::tensor(main, torch::dtype(torch::kByte)).view(main_data_shape); }
  }

  Batch batch(batch_size);
  for (size_t i {0}; i < batch_size; ++i) {
    batch[i] = Data(mains[i], labels[i]);
  }

  return batch;
}

std::vector<Batch> loadBatches(const std::string& batches_path) {
  std::vector<Batch> batches;
  std::vector<int64_t> main_data_shape;
  std::vector<int64_t> label_data_shape;
  std::string main_dtype;
  std::string label_dtype;
  size_t batch_size;

  std::filesystem::path batches_filepath = batches_path;
  std::filesystem::path metas = batches_path + "/meta";

  if (std::filesystem::exists(batches_filepath) && std::filesystem::is_directory(batches_filepath)) {
    for (const auto& filename : std::filesystem::directory_iterator(batches_filepath)) {
      if (!std::filesystem::is_directory(filename)) {
        json metaJson;
        std::string meta_path = metas.string() + "/meta_" + std::filesystem::path(filename).string().back() + ".json";
        std::ifstream metaFile(meta_path);        
        
        if (!metaFile.is_open()) {
          std::cerr << "Could not open the file: " << meta_path << std::endl;
        }

        metaFile >> metaJson;
        metaFile.close();

        batch_size = metaJson["batch_size"].get<size_t>();
        main_data_shape = metaJson["main_data_shape"].get<std::vector<int64_t>>();
        main_dtype = metaJson["main_dtype"].get<std::string>();
        label_dtype = metaJson["label_dtype"].get<std::string>();
        label_data_shape = metaJson["label_data_shape"].get<std::vector<int64_t>>();

        
        if (main_dtype == "uint8" && label_dtype == "uint8") {
          batches.push_back(loadBatch<uint8_t, uint8_t>(std::filesystem::path(filename).string(),
                                    batch_size, main_data_shape, label_data_shape));
        } else if (main_dtype == "uint8" && label_dtype == "float32") {
          batches.push_back(loadBatch<uint8_t, float>(std::filesystem::path(filename).string(),
                                    batch_size, main_data_shape, label_data_shape)); 
        } else if (main_dtype == "float32" && label_dtype == "float32") {
          batches.push_back(loadBatch<float, float>(std::filesystem::path(filename).string(),
                                    batch_size, main_data_shape, label_data_shape));
        } else if (main_dtype == "float32" && label_dtype == "int") { // not tested
          batches.push_back(loadBatch<float, int>(std::filesystem::path(filename).string(),
                                    batch_size, main_data_shape, label_data_shape));
        } else {
          std::cerr << "These types does not supported! " << main_dtype << ' ' << label_dtype << "\n\n";
        }
      }
    }

  } else {
    std::cerr << "Path does not exist.\n";
  }

  return batches;

}