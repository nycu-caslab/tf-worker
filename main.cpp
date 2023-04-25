#include <atomic>
#include <c10/cuda/CUDACachingAllocator.h>
#include <chrono>
#include <fstream>
#include <hiredis/hiredis.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/container/modulelist.h>
#include <torch/nn/modules/pooling.h>
#include <torch/torch.h>
#include <unistd.h>

static uint64_t *glob_var0;
static uint64_t *glob_var1;
static uint64_t *glob_var2;
static uint64_t *glob_var3;

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

void warmup(vector<Model> &models) {
  for (int i = 0; i < 8; i++) {
    torch::Tensor tensor = torch::rand({16, 3, 244, 244}).to(device);
    models[1].forward(tensor);
  }
}

void creator(char *redis) {
  if (torch::cuda::is_available()) {
    LOGs("Using CUDA\n");
    device = torch::Device(torch::kCUDA);
  }

  redisContext *c = redisConnect(redis, 6379);
  torch::Tensor input = torch::rand({32, 3, 224, 224}).to(device);
  while (true) {
    std::string result;
    redisReply *reply;
    reply = (redisReply *)redisCommand(c, "BLPOP creator 0");

    result = reply->element[1]->str;
    LOGs("Creator", ", Input: ", result);

    std::istringstream iss(result);
    freeReplyObject(reply);

    int variable_id;
    iss >> variable_id;
    variables[variable_id] = input.clone();
    if (variable_id == 0) {
      *glob_var0 = (uint64_t)variables[variable_id].data_ptr<float>();
      LOGs(glob_var0);
    } else if (variable_id == 1) {
      *glob_var1 = (uint64_t)variables[variable_id].data_ptr<float>();
      LOGs(glob_var1);
    } else if (variable_id == 2) {
      *glob_var2 = (uint64_t)variables[variable_id].data_ptr<float>();
      LOGs(glob_var2);
    } else if (variable_id == 3) {
      *glob_var3 = (uint64_t)variables[variable_id].data_ptr<float>();
      LOGs(glob_var3);
    }
    string cmd = to_string(variable_id) + " -1";
    redisCommand(c, "RPUSH done %s", cmd.c_str());
    LOGs("Creator", ", Created");
  }
}

void worker(int worker_id, char *redis) {
  // if (worker_id == 1) {
  //   setenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "10%", 1);
  // } else {
  //   setenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "90%", 1);
  // }
  if (torch::cuda::is_available()) {
    LOGs("Using CUDA\n");
    device = torch::Device(torch::kCUDA, 0);
  }

  vector<Model> models;
  int n = get_models_from_json(models, "schema.json");
  for (auto &model : models) {
    model.to(device);
  }

  redisContext *c = redisConnect(redis, 6379);

  warmup(models);
  redisCommand(c, "RPUSH initdone %s", to_string(worker_id).c_str());

  LOGs("Worker inited", worker_id);

  while (true) {
    std::string result;
    redisReply *reply;
    reply = (redisReply *)redisCommand(c, "BLPOP worker%d 0", worker_id);

    result = reply->element[1]->str;
    LOGs("Worker", worker_id, ", Input: ", result);

    std::istringstream iss(result);
    freeReplyObject(reply);

    string cmd;
    iss >> cmd;

    if (cmd == "forward") {
      int task_id, model_id, layer_id, variable_id;
      iss >> task_id >> model_id >> layer_id >> variable_id;

      LOGs("Start:", models[model_id].name, layer_id);
      if (layer_id == 0) {
        if (variable_id == 0) {
          LOGs(glob_var0);
          variables[variable_id] =
              torch::from_blob(glob_var0, {32, 3, 224, 224},
                               torch::TensorOptions().device(torch::kCUDA, 0));
        } else if (variable_id == 1) {
          LOGs(glob_var1);
          variables[variable_id] =
              torch::from_blob(glob_var1, {32, 3, 224, 224},
                               torch::TensorOptions().device(torch::kCUDA, 0));
        } else if (variable_id == 2) {
          LOGs(glob_var2);
          variables[variable_id] =
              torch::from_blob(glob_var2, {32, 3, 224, 224},
                               torch::TensorOptions().device(torch::kCUDA, 0));
        } else if (variable_id == 3) {
          LOGs(glob_var3);
          variables[variable_id] =
              torch::from_blob(glob_var3, {32, 3, 224, 224},
                               torch::TensorOptions().device(torch::kCUDA, 0));
        }
      }
      LOGs("Blobed:", models[model_id].name, layer_id);
      variables[variable_id] =
          models[model_id].forward_layer(layer_id, variables[variable_id]);

      std::string cmd = to_string(task_id) + " " + to_string(worker_id);
      reply = (redisReply *)redisCommand(c, "RPUSH done %s", cmd.c_str());

      LOGs("Forwarded:", models[model_id].name, layer_id);
    }
  }
}

int main() {
  LOGs("Process start");

  // setenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "10", 1);
  glob_var0 = (uint64_t *)mmap(NULL, sizeof *glob_var0, PROT_READ | PROT_WRITE,
                               MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  glob_var1 = (uint64_t *)mmap(NULL, sizeof *glob_var1, PROT_READ | PROT_WRITE,
                               MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  glob_var2 = (uint64_t *)mmap(NULL, sizeof *glob_var2, PROT_READ | PROT_WRITE,
                               MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  glob_var3 = (uint64_t *)mmap(NULL, sizeof *glob_var3, PROT_READ | PROT_WRITE,
                               MAP_SHARED | MAP_ANONYMOUS, -1, 0);

  // torch::Tensor tensor = torch::rand({16, 3, 244, 244}).to(torch::kCUDA);
  // auto p = tensor.data_ptr<float>();
  // cout << p << "\n";
  // auto options = torch::TensorOptions().device(torch::kCUDA);
  // auto ten2 = torch::from_blob(p, {16, 3, 244, 244}, options);

  int worker_id = fork();
  if (worker_id) {
    worker(0, getenv("REDIS"));
  } else {

    worker_id = fork();
    if (worker_id) {
      worker(1, getenv("REDIS"));
    } else {
      creator(getenv("REDIS"));
    }
  }

  // std::thread t1(worker, 0, getenv("REDIS"));
  // std::thread t2(worker, 1, getenv("REDIS"));
  // std::thread c1(creator, getenv("REDIS"));

  // t1.join();
  // t2.join();
  return 0;
}
