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

#include "json.hpp"
#include "utils.hpp"
using namespace std;
using namespace chrono;

std::atomic_int status;

const int MEM_INIT = 0;
const int MEM_SENT = 1;
const int MEM_RECV = 2;
std::atomic_long st, ed;
torch::nn::Sequential Vgg16;

void worker(int worker_id, vector<torch::Tensor> &variables, string redis) {
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
    cout << "gogogo";

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

      // Vgg16[layer_id]->forward(variables[worker_id]);

      LOGs("Forwarded: Vgg16-", layer_id);
      auto end = std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::system_clock::now().time_since_epoch())
                     .count();
      LOGs(worker_id, ":", start, end);
      cout << "gone";

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

  auto options = torch::TensorOptions().device(torch::kCUDA, 0);
  std::vector<torch::Tensor> variables(10);
  variables[0] = torch::rand({16, 3, 244, 244}, options);
  variables[1] = torch::rand({16, 3, 244, 244}, options);

  Vgg16 = torch::nn::Sequential(
      // Block 1
      torch::nn::Conv2d(
          torch::nn::Conv2dOptions(3, 64, 3).stride(1).bias(false)),
      torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
      torch::nn::Conv2d(
          torch::nn::Conv2dOptions(64, 64, 3).stride(1).bias(false)),
      torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
      torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)),

      // Block 2
      torch::nn::Conv2d(
          torch::nn::Conv2dOptions(64, 128, 3).stride(1).bias(false)),
      torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
      torch::nn::Conv2d(
          torch::nn::Conv2dOptions(128, 128, 3).stride(1).bias(false)),
      torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
      torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)),

      // Block 3
      torch::nn::Conv2d(
          torch::nn::Conv2dOptions(128, 256, 3).stride(1).bias(false)),
      torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
      torch::nn::Conv2d(
          torch::nn::Conv2dOptions(256, 256, 3).stride(1).bias(false)),
      torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
      torch::nn::Conv2d(
          torch::nn::Conv2dOptions(256, 256, 3).stride(1).bias(false)),
      torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
      torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)),

      // Block 4
      torch::nn::Conv2d(
          torch::nn::Conv2dOptions(256, 512, 3).stride(1).bias(false)),
      torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
      torch::nn::Conv2d(
          torch::nn::Conv2dOptions(512, 512, 3).stride(1).bias(false)),
      torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
      torch::nn::Conv2d(
          torch::nn::Conv2dOptions(512, 512, 3).stride(1).bias(false)),
      torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
      torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)),

      // Block 5
      torch::nn::Conv2d(
          torch::nn::Conv2dOptions(512, 512, 3).stride(1).bias(false)),
      torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
      torch::nn::Conv2d(
          torch::nn::Conv2dOptions(512, 512, 3).stride(1).bias(false)),
      torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
      torch::nn::Conv2d(
          torch::nn::Conv2dOptions(512, 512, 3).stride(1).bias(false)),
      torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
      torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2))

      //  Flat(root),
      //  FC(root, session, 512 * 7 * 7, 4096),
      //  torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
      //  FC(root, session, 4096, 4096),
      //  torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
      //  FC(root, session, 4096, 10),
  );
  torch::Device device(torch::kCUDA, 0);
  Vgg16->to(device);
  std::cout << Vgg16->size() << std::endl;

  status = 0;

  std::thread t1(worker, 0, std::ref(variables), getenv("REDIS0"));
  std::thread t2(worker, 1, std::ref(variables), getenv("REDIS1"));

  t1.join();
  t2.join();
  return 0;
}
