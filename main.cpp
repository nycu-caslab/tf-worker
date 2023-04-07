#include <atomic>
#include <chrono>
#include <fstream>
#include <hiredis/hiredis.h>
#include <iostream>
#include <stdio.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/container/modulelist.h>
#include <torch/nn/modules/pooling.h>
#include <torch/torch.h>

#include "model.hpp"
#include "utils.hpp"

using namespace std;
using namespace chrono;
using json = nlohmann::json;

std::atomic_int status;

const int MEM_INIT = 0;
const int MEM_SENT = 1;
const int MEM_RECV = 2;
std::atomic_long st, ed;
torch::nn::Sequential Vgg16;

void worker(int worker_id, vector<torch::Tensor> &variables,
            vector<Model> &models, string redis) {
  LOGs("Start pooling redis", redis);

  redisContext *c = redisConnect(redis.c_str(), 6379);

  LOGs("Creating redis instance");

  if (c == NULL || c->err) {
    if (c) {
      LOGs("Error:", c->errstr);
    } else {
      LOGs("Can't allocate redis context");
    }
  }

  while (true) {
    std::string result;
    redisReply *reply;
    reply = (redisReply *)redisCommand(c, "BLPOP foo 0");

    result = reply->element[1]->str;
    LOGs("Worker", worker_id, ", Input: ", result);

    std::istringstream iss(result);
    freeReplyObject(reply);

    string cmd;
    iss >> cmd;
    if (cmd == "forward") {
      cout << "went";
      int model_id, layer_id;
      iss >> model_id >> layer_id;
      auto start = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();

      models[model_id].forward_layer(layer_id, variables[worker_id]);

      LOGs("Forwarded: Vgg16-", layer_id);
      auto end = std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::system_clock::now().time_since_epoch())
                     .count();
      LOGs(worker_id, ":", start, end);

      if (layer_id == 0)
        if (layer_id == 36) {
          st = start;
          LOGs("------------------------");
          ed = end;
          LOGs("Total:", ed - st);
        }

    } else if (cmd == "send") {
      int src, dst;
      iss >> src >> dst;
      variables[dst] = variables[src];
      status = MEM_SENT;
      LOGs("Sent ", src, "to", dst);
    } else if (cmd == "recv") {
      int src, dst;
      iss >> src >> dst;
      while (status != MEM_SENT) {
      }
      status = MEM_INIT;
      LOGs("Recv ", src, "to", dst);
    } else if (cmd == "create") {
      int batch_size;
      iss >> batch_size;
    }
  }
}

int main() {
  std::ofstream stream;

  LOGs("Process start");

  auto options = torch::TensorOptions().device(torch::kCPU);
  std::vector<torch::Tensor> variables(10);
  variables[0] = torch::rand({16, 3, 244, 244}, options);
  variables[1] = torch::rand({16, 3, 244, 244}, options);

  vector<Model> models;
  int n = get_models_from_json(models, "schema.json");
  for (int i = 0; i < n; i++) {
    cout << "Model " << i << ": " << models[i].size() << "\n";
    for (int j = 0; j < models[i].size(); j++) {
      cout << "Forwarding layer " << j << "\n";
      cout << "Shape: " << torch::_shape_as_tensor(variables[i]) << "\n";
      variables[i] = models[i].forward_layer(j, variables[i]);
    }
  }

  // torch::Device device(torch::kCUDA, 0);
  // Vgg16->to(device);

  status = 0;

  // std::thread t1(worker, 0, std::ref(variables), std::ref(models),
  //                getenv("REDIS0"));
  // std::thread t2(worker, 1, std::ref(variables), std::ref(models),
  //                getenv("REDIS1"));

  // t1.join();
  // t2.join();
  return 0;
}
