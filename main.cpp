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

  variables[worker_id] = torch::rand({16, 3, 244, 244}).to(device);
  sleep(1);

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

      cout << torch::_shape_as_tensor(variables[variable_id]);

      variables[variable_id] =
          models[model_id].forward_layer(layer_id, variables[variable_id]);

      std::string cmd = to_string(task_id) + " " + to_string(worker_id);
      reply = (redisReply *)redisCommand(done, "RPUSH done %s", cmd.c_str());

      LOGs("Forwarded: Vgg16-", layer_id);
    } else if (cmd == "send") {
      int src, dst;
      iss >> src >> dst;
      variables[dst] = variables[src];
      while (status != MEM_INIT) {
      }
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
      variables[worker_id] = torch::rand({batch_size, 3, 244, 244}).to(device);
    }
  }
}

void test_torch(vector<Model> &models) {
  int n = models.size();
  for (int i = 0; i < n; i++) {
    // cout << "Model " << i << ": " << models[i].size() << "\n";
    for (int o = 0; o < 20; o++) {
      torch::Tensor variable = torch::rand({16, 3, 244, 244}).to(device);
      // if (torch::cuda::is_available()) {
      //   torch::cuda::synchronize();
      // }
      // auto start = std::chrono::duration_cast<std::chrono::microseconds>(
      //                  std::chrono::system_clock::now().time_since_epoch())
      //                  .count();
      for (int j = 0; j < models[i].size(); j++) {
        // cout << "Forwarding layer " << j << "\n";
        // cout << "Shape: " << torch::_shape_as_tensor(variables[i]) <<
        // "\n";
        variable = models[i].forward_layer(j, variable);
      }
      // if (torch::cuda::is_available()) {
      //   torch::cuda::synchronize();
      // }
      // auto end = std::chrono::duration_cast<std::chrono::microseconds>(
      //                std::chrono::system_clock::now().time_since_epoch())
      //                .count();
      // cout << end - start << "\n";
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

  status = 0;

  // for (int i = 0; i < n; i++) {
  //   for (int o = 0; o < 20; o++) {
  //     torch::Tensor variable = torch::rand({16, 3, 244, 244}).to(device);
  //     for (int j = 0; j < models[i].size(); j++) {
  //       variable = models[i].forward_layer(j, variable);
  //     }
  //   }
  // }

  // thread test1(test_torch, ref(models));
  // thread test2(test_torch, ref(models));
  // test1.join();
  // test2.join();

  // worker(0, variables, models, getenv("REDIS0"));
  std::thread t1(worker, 0, std::ref(models), getenv("REDIS0"),
                 getenv("REDISDONE"));
  std::thread t2(worker, 1, std::ref(models), getenv("REDIS1"),
                 getenv("REDISDONE"));

  t1.join();
  t2.join();
  return 0;
}
