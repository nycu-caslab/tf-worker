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

torch::Device get_device() {
  if (torch::cuda::is_available()) {
    return torch::Device(torch::kCUDA);
  } else {
    return torch::Device(torch::kCPU);
  }
}

vector<long> get_shape_from_dim(int dim[4]) {
  vector<long> result;
  for (int i = 0; i < 4; i++) {
    if (dim[i])
      result.push_back(dim[i]);
  }
  return result;
}

struct variable_t {
  cudaIpcMemHandle_t memHandle;
  int dim[4];
  int pos;
};

variable_t *variables;

void warmup(vector<Model> &models) {
  torch::Device device = get_device();
  torch::Tensor tensor;
  for (int i = 0; i < 8; i++) {
    if (models[i % models.size()].name == "lstm") {
      tensor = torch::rand({16, 3, 50176}).to(device);
      models[i % models.size()].forward_layer(0, tensor);
    } else {
      tensor = torch::rand({16, 3, 224, 224}).to(device);
      models[i % models.size()].forward(tensor);
    }
    torch::cuda::synchronize();
  }
}

void creator(char *redis) {
  vector<torch::Tensor> tensors(N_of_variables);
  torch::Device device = get_device();

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
    variables[variable_id].pos = -1;

    string cmd = to_string(variable_id) + " -1";
    redisCommand(c, "RPUSH done %s", cmd.c_str());
    LOGs("Creator", ", Created");
  }
}

void worker(int worker_id, char *redis) {
  if (worker_id == 0) {
    setenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", getenv("WORKER0"), 1);
  } else {
    setenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", getenv("WORKER1"), 1);
  }

  torch::Device device = get_device();

  vector<Model> models;

  int n = get_models_from_json(models, "schema.json");
  if (string(getenv("CASE")) == "0") {
    models = {models[0], models[1]};
  } else if (string(getenv("CASE")) == "1") {
    models = {models[0], models[2]};
  } else if (string(getenv("CASE")) == "2") {
    models = {models[1], models[2]};
  }

  for (auto &model : models) {
    model.to(device);
  }

  redisContext *c = redisConnect(redis, 6379);

  warmup(models);

  redisCommand(c, "RPUSH initdone %s", to_string(worker_id).c_str());
  torch::Tensor tensor;

  LOGs("Worker inited", worker_id);

  while (true) {
    std::string result;
    redisReply *reply;
    reply = (redisReply *)redisCommand(c, "BLPOP worker%d 0", worker_id);

    result = reply->element[1]->str;
    // LOGs("Worker", worker_id, ", Input: ", result);

    std::istringstream iss(result);
    freeReplyObject(reply);

    string cmd;
    iss >> cmd;

    float *ptr;

    if (cmd == "forward") {
      int task_id, model_id, layer_id, variable_id;
      iss >> task_id >> model_id >> layer_id >> variable_id;

      // LOGs("Start:", models[model_id].name, layer_id, variable_id);
      // LOGs("Variable:", variables[variable_id].pos);

      if (layer_id == 0 && models[model_id].name == "lstm") {
        variables[variable_id].dim[1] = 3;
        variables[variable_id].dim[2] = 50176;
        variables[variable_id].dim[3] = 0;
      }

      if (variables[variable_id].pos % 10 != worker_id) {
        if (variables[variable_id].pos >= 10) {
          // LOGs("Over");
        }
        if (variables[variable_id].pos != -1) {
          variables[variable_id].pos = 10;
        } else {
          variables[variable_id].pos = 0;
        }
        variables[variable_id].pos += worker_id;

        int blob = cudaIpcOpenMemHandle(
            (void **)&(ptr),
            *(cudaIpcMemHandle_t *)&variables[variable_id].memHandle,
            cudaIpcMemLazyEnablePeerAccess);
        tensor = torch::from_blob(
            ptr, get_shape_from_dim(variables[variable_id].dim),
            torch::TensorOptions().device(device));
        // LOGs("Blobed");
      }

      // LOGs(torch::_shape_as_tensor(local_variables[variable]));
      // LOGs(local_variables[variable].data_ptr<float>());
      ptr = tensor.data_ptr<float>();
      tensor = models[model_id].forward_layer(layer_id, tensor);
      torch::cuda::synchronize();
      // LOGs(local_variables[variable].data_ptr<float>());
      // LOGs(torch::_shape_as_tensor(local_variables[variable]));
      // LOGs("fowwared", worker_id);

      if (ptr != tensor.data_ptr<float>()) {
        LOGs(cudaIpcGetMemHandle(
            (cudaIpcMemHandle_t *)&variables[variable_id].memHandle,
            tensor.data_ptr<float>()));
      }

      torch::Tensor shape = torch::_shape_as_tensor(tensor);

      for (int i = 0; i < tensor.dim(); i++) {
        variables[variable_id].dim[i] = tensor.sizes()[i];
      }

      for (int i = tensor.dim(); i < 4; i++) {
        variables[variable_id].dim[i] = 0;
      }

      std::string cmd = to_string(task_id) + " " + to_string(worker_id);
      reply = (redisReply *)redisCommand(c, "RPUSH done %s", cmd.c_str());

      // LOGs("Forwarded:", models[model_id].name, layer_id);
    }
  }
}

int main() {
  LOGs("Process start");

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
