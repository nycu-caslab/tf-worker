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
long scheduler_all = 0, scheduler_n = 0;
bool overhead = true;

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
    Model &model = models[i % models.size()];
    tensor = torch::rand(model.input_shape).to(device);
    long long move_time = 0;
    for (int j = 0; j < model.size(); j++) {
      torch::cuda::synchronize();
      long long move_start = chrono::duration_cast<chrono::microseconds>(
                                 chrono::steady_clock::now().time_since_epoch())
                                 .count();
      tensor.to(torch::kCPU);
      tensor.to(device);
      torch::cuda::synchronize();
      long long move_end = chrono::duration_cast<chrono::microseconds>(
                               chrono::steady_clock::now().time_since_epoch())
                               .count();
      move_time += move_end - move_start;
      tensor = model.forward_layer(j, tensor);
    }
    LOGs(i % models.size(), move_time, model.size());
    torch::cuda::synchronize();
  }
}

void creator(char *redis) {
  vector<torch::Tensor> tensors(N_of_variables);
  torch::Device device = get_device();

  vector<Model> models;

  int n = get_models_from_json(models, "schema.json");

  redisContext *c = redisConnect(redis, 6379);

  vector<torch::Tensor> inputs = {
      torch::rand(models[0].input_shape).to(device),
      torch::rand(models[1].input_shape).to(device)};

  while (true) {
    std::string result;
    redisReply *reply;
    reply = (redisReply *)redisCommand(c, "BLPOP creator 0");

    result = reply->element[1]->str;
    // LOGs("Creator", ", Input: ", result);

    std::istringstream iss(result);
    freeReplyObject(reply);

    int variable_id;
    iss >> variable_id;
    int model_id = (variable_id + N_of_variables / 2) / N_of_variables;
    tensors[variable_id] = inputs[model_id].clone();
    cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&variables[variable_id].memHandle,
                        tensors[variable_id].data_ptr<float>());

    for (int i = 0; i < 4; i++) {
      variables[variable_id].dim[i] = 0;
    }

    for (int i = 0; i < models[model_id].input_shape.size(); i++) {
      variables[variable_id].dim[i] = models[model_id].input_shape[i];
    }
    variables[variable_id].pos = -1;

    string cmd = to_string(variable_id) + " -1";
    redisCommand(c, "RPUSH done %s", cmd.c_str());
    // LOGs("Creator", ", Created");
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

      if (variables[variable_id].pos % 10 != worker_id) {
        long scheduler_start;
        if (variables[variable_id].pos >= 10) {
          // LOGs("Over");
        }
        if (variables[variable_id].pos != -1) {
          variables[variable_id].pos = 10;
          if (overhead) {
            scheduler_start =
                chrono::duration_cast<chrono::microseconds>(
                    chrono::steady_clock::now().time_since_epoch())
                    .count();
          }
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
        if (overhead && variables[variable_id].pos >= 10) {
          long scheduler_end =
              chrono::duration_cast<chrono::microseconds>(
                  chrono::steady_clock::now().time_since_epoch())
                  .count();
          scheduler_all += scheduler_end - scheduler_start;
          scheduler_n++;
          if (scheduler_n == 20) {
            LOGs("Overhead", scheduler_all);
          }
        }
      }

      // LOGs(torch::_shape_as_tensor(local_variables[variable]));
      // LOGs(local_variables[variable].data_ptr<float>());
      // LOGs(torch::_shape_as_tensor(tensor));
      ptr = tensor.data_ptr<float>();
      tensor = models[model_id].forward_layer(layer_id, tensor);
      torch::cuda::synchronize();
      // LOGs(local_variables[variable].data_ptr<float>());
      // LOGs(torch::_shape_as_tensor(tensor));
      // LOGs("fowwared", worker_id);

      if (ptr != tensor.data_ptr<float>()) {
        cudaIpcGetMemHandle(
            (cudaIpcMemHandle_t *)&variables[variable_id].memHandle,
            tensor.data_ptr<float>());
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
