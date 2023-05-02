#include <atomic>

#include <c10/cuda/CUDACachingAllocator.h>
#include <chrono>
#include <cuda.h>
#include <errno.h>
#include <error.h>
#include <fstream>
#include <hiredis/hiredis.h>
#include <iostream>
#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/container/modulelist.h>
#include <torch/nn/modules/pooling.h>
#include <torch/torch.h>
#include <unistd.h>

#include "model.hpp"
#include "utils.hpp"
#define N_of_variables 10

using namespace std;
using namespace chrono;

const int MEM_INIT = 0;
const int MEM_SENT = 1;
const int MEM_RECV = 2;

std::atomic_long st, ed;
torch::Device device(torch::kCPU);

struct variable_t {
  cudaIpcMemHandle_t memHandle;
  int dim[4];
};

variable_t *variables;

void warmup(vector<Model> &models) {
  for (int i = 0; i < 8; i++) {
    torch::Tensor tensor = torch::rand({16, 3, 224, 224}).to(device);
    models[i % 2].forward(tensor);
  }
}

void creator(char *redis) {
  vector<torch::Tensor> tensors(N_of_variables);
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
    tensors[variable_id] = input.clone();
    cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&variables[variable_id].memHandle,
                        tensors[variable_id].data_ptr<float>());

    variables[variable_id].dim[0] = 32;
    variables[variable_id].dim[1] = 3;
    variables[variable_id].dim[2] = 224;
    variables[variable_id].dim[3] = 224;

    string cmd = to_string(variable_id) + " -1";
    redisCommand(c, "RPUSH done %s", cmd.c_str());
    LOGs("Creator", ", Created");
  }
}

void worker(int worker_id, char *redis) {
  if (worker_id == 1) {
    setenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "10%", 1);
  } else {
    setenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "90%", 1);
  }

  if (torch::cuda::is_available()) {
    LOGs("Using CUDA\n");
    device = torch::Device(torch::kCUDA, 0);
  }

  int variable = -1;
  torch::Tensor tensor;

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

      if (variable != variable_id) {
        variable = variable_id;
        float *ptr;
        cudaIpcOpenMemHandle(
            (void **)&ptr,
            *(cudaIpcMemHandle_t *)&variables[variable].memHandle,
            cudaIpcMemLazyEnablePeerAccess);
        tensor = torch::from_blob(ptr, {32, 3, 224, 224},
                                  torch::TensorOptions().device(torch::kCUDA));
        LOGs("Blobed:", models[model_id].name, layer_id);
      }

      tensor = models[model_id].forward_layer(layer_id, tensor);
      if (layer_id + 1 == models[model_id].size()) {
        variable = -1;
      }

      std::string cmd = to_string(task_id) + " " + to_string(worker_id);
      reply = (redisReply *)redisCommand(c, "RPUSH done %s", cmd.c_str());

      LOGs("Forwarded:", models[model_id].name, layer_id);
    }
  }
}

int old() {
  LOGs("Process start");

  // setenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "10", 1);

  float *a = nullptr;
  variables = (variable_t *)mmap(NULL, sizeof(*variables) * N_of_variables,
                                 PROT_READ | PROT_WRITE,
                                 MAP_SHARED | MAP_ANONYMOUS, 0, 0);

  memset((void *)variables, 0, sizeof(*variables) * N_of_variables);

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

  return 0;
}

int main() { old(); }
