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
  int version;
  bool moved;
};

variable_t *variables;

void warmup(vector<Model> &models) {
  torch::Device device = get_device();
  for (int i = 0; i < 8; i++) {
    torch::Tensor tensor = torch::rand({16, 3, 224, 224}).to(device);
    models[i % 2].forward(tensor);
  }
}

void creator(char *redis) {
  vector<torch::Tensor> tensors(N_of_variables);
  torch::Device device = get_device();

  redisContext *c = redisConnect(redis, 6379);
  torch::Tensor input = torch::rand({32, 3, 224, 224}).to(device);
  for (int i = 0; i < N_of_variables; i++) {
    variables[i].version = 0;
  }

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
    variables[variable_id].moved = true;
    variables[variable_id].version++;

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

  int variable = -1;
  torch::Device device = get_device();

  vector<Model> models;

  int n = get_models_from_json(models, "schema.json");
  for (auto &model : models) {
    model.to(device);
  }

  redisContext *c = redisConnect(redis, 6379);

  warmup(models);

  redisCommand(c, "RPUSH initdone %s", to_string(worker_id).c_str());

  LOGs("Worker inited", worker_id);
  vector<torch::Tensor> local_variables(N_of_variables);
  vector<int> variable_versions(N_of_variables, -1);

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

    vector<float *> ptrs(10, nullptr);

    float *ptr;
    // for (int i = 0; i < N_of_variables; i++) {
    //   if (variables[i].version != variable_versions[i])
    //     local_variables[i] = torch::Tensor();
    // }
    if (cmd == "forward") {
      int task_id, model_id, layer_id, variable_id;
      iss >> task_id >> model_id >> layer_id >> variable_id;

      LOGs("Start:", models[model_id].name, layer_id, variable, variable_id);
      LOGs("Variable:", variables[variable_id].pos,
           variables[variable_id].moved);

      if (variable_versions[variable_id] != variables[variable_id].version ||
          (variables[variable_id].pos != worker_id &&
           variables[variable_id].moved) ||
          variable != variable_id) {
        variable = variable_id;
        if (variables[variable].pos != worker_id &&
            (variable_versions[variable_id] != variables[variable_id].version ||
             variables[variable].moved)) {
          LOGs("path0");
          float *ptr;
          int blob = cudaIpcOpenMemHandle(
              (void **)&(ptr),
              *(cudaIpcMemHandle_t *)&variables[variable].memHandle,
              cudaIpcMemLazyEnablePeerAccess);
          local_variables[variable] =
              torch::from_blob(ptr, get_shape_from_dim(variables[variable].dim),
                               torch::TensorOptions().device(device));
          // LOGs("Blobed0:", blob, models[model_id].name, "Layer: ", layer_id);
          // cudaIpcCloseMemHandle(ptr);
        } else {
          // LOGs("path1");
          // local_variables[variable] = torch::from_blob(
          //     ptrs[variable], get_shape_from_dim(variables[variable].dim),
          //     torch::TensorOptions().device(device));
          // LOGs("Blobed1:", models[model_id].name, "Layer ", layer_id);
        }
      }
      variable = variable_id;
      variable_versions[variable_id] = variables[variable_id].version;
      if (variables[variable].dim[2] == 0) {
        local_variables[variable] = at::reshape(
            local_variables[variable],
            {variables[variable].dim[0], variables[variable].dim[1]});
      }

      variables[variable].pos = worker_id;

      // LOGs(local_variables[variable].data_ptr<float>());
      // LOGs(torch::_shape_as_tensor(local_variables[variable]));
      ptrs[variable] = local_variables[variable].data_ptr<float>();
      local_variables[variable] =
          models[model_id].forward_layer(layer_id, local_variables[variable]);
      // LOGs(local_variables[variable].data_ptr<float>());
      // LOGs(torch::_shape_as_tensor(local_variables[variable]));
      // LOGs("fowwared", worker_id);

      if (ptrs[variable] != local_variables[variable].data_ptr<float>()) {
        variables[variable].moved = true;
        // LOGs(cudaIpcGetMemHandle(
        //     (cudaIpcMemHandle_t *)&variables[variable_id].memHandle,
        //     local_variables[variable].data_ptr<float>()));
      } else {

        variables[variable].moved = false;
      }
      for (int i = 0; i < 4; i++) {
        variables[variable].dim[i] = 0;
      }

      torch::Tensor shape = torch::_shape_as_tensor(local_variables[variable]);
      for (int i = 0; i < shape.sizes()[0]; i++) {
        variables[variable].dim[i] = shape[i].item<int>();
      }

      if (layer_id + 1 == models[model_id].size()) {
        variable = -1;
      }

      std::string cmd = to_string(task_id) + " " + to_string(worker_id);
      reply = (redisReply *)redisCommand(c, "RPUSH done %s", cmd.c_str());

      LOGs("Forwarded:", models[model_id].name, layer_id);
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
