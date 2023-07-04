#ifndef MODEL_HPP
#define MODEL_HPP

#include "json.hpp"
#include <fstream>
#include <iostream>
#include <torch/torch.h>
#include <vector>

using json = nlohmann::json;
using namespace std;

enum class op { conv, relu, maxpool, linear, flat, avgpool, lstm, tf };

class Layer {
public:
  op type;
  Layer(op type) : type(type){};
};

class Model {
public:
  int id;
  string name;
  Model(int id, string name) : id(id), name(name){};
  vector<long> input_shape;
  size_t size() { return layers->size(); }

  torch::Tensor forward_layer(int layer_id, torch::Tensor &tensor) {
    assert(layer_data.size() > layer_id);
    assert(layers->size() > layer_id);
    switch (layer_data[layer_id].type) {
    case op::conv:
      return layers->at<torch::nn::Conv2dImpl>(layer_id).forward(tensor);
    case op::relu:
      return layers->at<torch::nn::ReLUImpl>(layer_id).forward(tensor);
    case op::maxpool:
      return layers->at<torch::nn::MaxPool2dImpl>(layer_id).forward(tensor);
    case op::linear:
      return layers->at<torch::nn::LinearImpl>(layer_id).forward(tensor);
    case op::flat:
      return layers->at<torch::nn::FlattenImpl>(layer_id).forward(tensor);
    case op::avgpool:
      return layers->at<torch::nn::AdaptiveAvgPool2dImpl>(layer_id).forward(
          tensor);
    case op::lstm:
      return get<0>(layers->at<torch::nn::LSTMImpl>(layer_id).forward(tensor));
    case op::tf:
      torch::Tensor tgt = tensor.clone();
      return layers->at<torch::nn::TransformerImpl>(layer_id).forward(tensor,
                                                                      tgt);
    }
    return tensor;
  }

  torch::Tensor forward(torch::Tensor &tensor) {
    return layers->forward(tensor);
  }

  void to(torch::Device &device) { this->layers->to(device); }

  void add_conv_layer(int in_channels, int out_channels, int filter_size,
                      int stride, int padding) {
    auto options =
        torch::nn::Conv2dOptions(in_channels, out_channels, filter_size)
            .padding(padding)
            .bias(false);
    if (stride) {
      options.stride(stride);
    }
    this->layers->push_back(torch::nn::Conv2d(options));
    this->layer_data.push_back(Layer(op::conv));
  }
  void add_relu_layer() {
    this->layers->push_back(
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
    this->layer_data.push_back(Layer(op::relu));
  }

  void add_maxpool_layer(int size, int stride) {
    auto options = torch::nn::MaxPool2dOptions(size);
    if (stride) {
      options.stride(stride);
    }
    this->layers->push_back(torch::nn::MaxPool2d(options));
    this->layer_data.push_back(Layer(op::maxpool));
  }
  void add_linear_layer(int in_channels, int out_channels) {
    this->layers->push_back(torch::nn::Linear(
        torch::nn ::LinearOptions(in_channels, out_channels)));
    this->layer_data.push_back(Layer(op::linear));
  }
  void add_flat_layer() {
    this->layers->push_back(torch::nn::Flatten(
        torch::nn::FlattenOptions().start_dim(1).end_dim(3)));
    this->layer_data.push_back(Layer(op::flat));
  }
  void add_avgpool_layer(int size) {
    this->layers->push_back(torch::nn::AdaptiveAvgPool2d(
        torch::nn::AdaptiveAvgPool2dOptions({size, size})));
    this->layer_data.push_back(Layer(op::avgpool));
  }
  void add_lstm_layer(int input_size, int hidden_size) {
    this->layers->push_back(
        torch::nn::LSTM(torch::nn::LSTMOptions(input_size, hidden_size)
                            .num_layers(3)
                            .batch_first(false)
                            .bidirectional(true)));
    this->layer_data.push_back(Layer(op::lstm));
  }
  void add_tf_layer(int d_model, int nhead) {
    this->layers->push_back(
        torch::nn::Transformer(torch::nn::TransformerOptions(d_model, nhead)));
    this->layer_data.push_back(Layer(op::tf));
  }

private:
  torch::nn::Sequential layers;
  vector<Layer> layer_data;
};

int get_models_from_json(vector<Model> &models, string filename) {
  ifstream f(filename);
  json data = json::parse(f);
  for (int i = 0; i < data["Models"].size(); i++) {
    models.push_back(Model(i, data["Models"][i]["name"]));
    for (auto &num : data["Models"][i]["input_shape"]) {
      models[i].input_shape.push_back(num);
    }
    for (auto &layer : data["Models"][i]["layers"]) {
      if (layer["type"] == "conv") {
        models[i].add_conv_layer(
            layer["params"]["in_channels"], layer["params"]["out_channels"],
            layer["params"]["filter_size"], layer["params"]["stride"],
            layer["params"]["padding"]);
      } else if (layer["type"] == "relu") {
        models[i].add_relu_layer();
      } else if (layer["type"] == "maxpool") {
        models[i].add_maxpool_layer(layer["params"]["ksize"],
                                    layer["params"]["kstride"]);
      } else if (layer["type"] == "linear") {
        models[i].add_linear_layer(layer["params"]["in_channels"],
                                   layer["params"]["out_channels"]);
      } else if (layer["type"] == "flat") {
        models[i].add_flat_layer();
      } else if (layer["type"] == "avgpool") {
        models[i].add_avgpool_layer(layer["params"]["size"]);
      } else if (layer["type"] == "lstm") {
        models[i].add_lstm_layer(layer["params"]["input_size"],
                                 layer["params"]["hidden_size"]);
      } else if (layer["type"] == "tf") {
        models[i].add_tf_layer(layer["params"]["d_model"],
                               layer["params"]["nhead"]);
      }
    }
  }
  return models.size();
}
#endif // !MODEL_HPP
