#include <atomic>
#include <c10/cuda/CUDACachingAllocator.h>
#include <chrono>
#include <fstream>
#include <hiredis/hiredis.h>
#include <iostream>
#include <stdio.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/container/modulelist.h>
#include <torch/nn/modules/pooling.h>
#include <torch/torch.h>
#include <unistd.h>

#include "model.hpp"
#include "utils.hpp"

using namespace std;
using namespace chrono;

std::atomic_int status;

const int MEM_INIT = 0;
const int MEM_SENT = 1;
const int MEM_RECV = 2;
std::atomic_long st, ed;
std::vector<torch::Tensor> variables(10);
torch::Device device(torch::kCPU);

void worker(int worker_id, vector<Model> &models, string redis,
            string redis_done) {
  LOGs("Start pooling redis", redis);

  redisContext *c = redisConnect(redis.c_str(), 6379);
  redisContext *done = redisConnect(redis_done.c_str(), 6379);

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
      int task_id, model_id, layer_id, variable_id;
      iss >> task_id >> model_id >> layer_id >> variable_id;
      if (layer_id == 0) {
        variables[variable_id] = torch::rand({16, 3, 244, 244}).to(device);
      }

      variables[variable_id] =
          models[model_id].forward_layer(layer_id, variables[variable_id]);

      std::string cmd = to_string(task_id) + " " + to_string(worker_id);
      reply = (redisReply *)redisCommand(done, "RPUSH done %s", cmd.c_str());

      LOGs("Forwarded: Vgg16-", layer_id);
    } else if (cmd == "create") {
      int batch_size;
      iss >> batch_size;
      variables[worker_id] = torch::rand({batch_size, 3, 244, 244}).to(device);
    }
  }
}

int main() {
  LOGs("Process start");

  if (torch::cuda::is_available()) {
    cout << "Using CUDA\n";
    device = torch::Device(torch::kCUDA);
  }

  vector<Model> models;
  int n = get_models_from_json(models, "schema.json");
  for (auto &model : models) {
    model.to(device);
  }

  std::thread t1(worker, 0, std::ref(models), getenv("REDIS0"),
                 getenv("REDISDONE"));
  std::thread t2(worker, 1, std::ref(models), getenv("REDIS1"),
                 getenv("REDISDONE"));

  t1.join();
  t2.join();
  return 0;
}
