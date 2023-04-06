#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/queue_interface.h"
#include <atomic>
#include <chrono>
#include <fstream>
#include <hiredis/hiredis.h>
#include <iostream>
#include <stdio.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>

#include "layer.hpp"
#include "tensorflow/core/platform/status.h"
#include "utils.hpp"

using namespace tensorflow;
using namespace tensorflow::ops;
std::vector<Op *> Vgg16;
std::atomic_int status;

const int MEM_INIT = 0;
const int MEM_SENT = 1;
const int MEM_RECV = 2;
std::atomic_long st, ed;

void worker(int worker_id, Scope &root, ClientSession &session,
            std::vector<Output> &variables,
            std::vector<std::vector<int>> &send_recv, QueueEnqueue &enqueue,
            QueueDequeue &dequeue, QueueSize &size, string redis) {
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
      int model_id, layer_id;
      iss >> model_id >> layer_id;
      auto start = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
      variables[worker_id] =
          Vgg16[layer_id]->forward(session, variables[worker_id]);
      auto end = std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::system_clock::now().time_since_epoch())
                     .count();
      LOGs("Forwarded: Vgg16-", layer_id);
      LOGs(worker_id, ":", start, end);

      if (layer_id == 0)
        st = start;
      if (layer_id == 36) {
        LOGs("------------------------");
        ed = end;
        LOGs("Total:", ed - st);
      }

    } else if (cmd == "push") {
      int variable_id;
      iss >> variable_id;
      std::vector<Tensor> outputs;
      TF_CHECK_OK(
          session.Run({}, {variables[variable_id]}, {enqueue}, &outputs));
      TF_CHECK_OK(session.Run({}, {size}, {}, &outputs));
      int size_value = *outputs[0].scalar<int>().data();
      LOGs("Enqueued ", variable_id, "size: ", size_value);
    } else if (cmd == "pop") {
      int variable_id;
      iss >> variable_id;
      std::vector<Tensor> outputs;
      TF_CHECK_OK(session.Run({}, {}, {dequeue.operation}, &outputs));
      variables[variable_id] = dequeue.components[0];
      TF_CHECK_OK(session.Run({variables[variable_id]}, &outputs));
      std::cout << outputs[0].shape();
      TF_CHECK_OK(session.Run({}, {size}, {}, &outputs));
      int value_3 = *outputs[0].scalar<int>().data();
      LOGs("Dequeued ", variable_id, "size: ", value_3);
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

      std::vector<Tensor> outputs;
      TensorShape sp({batch_size, 224, 224, 3});
      auto input = Variable(root, sp, DT_FLOAT);
      variables[worker_id] = Assign(
          root, input, RandomNormal(root, {batch_size, 224, 224, 3}, DT_FLOAT));
      TF_CHECK_OK(session.Run({}, {variables[0]}, &outputs));
    }
  }
}

int main() {
  std::ofstream stream;
  stream.open("tf-worker/log.txt");
  stream.close();

  LOGs("Process start");

  Scope root = Scope::NewRootScope();
  SessionOptions config;
  config.config.mutable_gpu_options()->set_allow_growth(true);
  ClientSession session(root, config);

  FIFOQueue::Attrs fifoqueue_attr;
  FIFOQueue queue(root, {DT_FLOAT}, fifoqueue_attr.SharedName("test"));
  QueueEnqueue enqueue(root.WithOpName("enqueue"), queue, {Output()});
  QueueDequeue dequeue(root.WithOpName("dequeue"), queue, {DT_FLOAT});
  QueueSize size(root.WithOpName("size"), queue);

  std::vector<Output> variables(10);

  Vgg16 = {
      // Block 1
      new Conv(root, session, 3, 3, 64, 1),
      new Activation(root),
      new Conv(root, session, 3, 64, 64, 1),
      new Activation(root),
      new Pool(root, 2, 2),

      // Block 2
      new Conv(root, session, 3, 64, 128, 1),
      new Activation(root),
      new Conv(root, session, 3, 128, 128, 1),
      new Activation(root),
      new Pool(root, 2, 2),

      // Block 3
      new Conv(root, session, 3, 128, 256, 1),
      new Activation(root),
      new Conv(root, session, 3, 256, 256, 1),
      new Activation(root),
      new Conv(root, session, 3, 256, 256, 1),
      new Activation(root),
      new Pool(root, 2, 2),

      // Block 4
      new Conv(root, session, 3, 256, 512, 1),
      new Activation(root),
      new Conv(root, session, 3, 512, 512, 1),
      new Activation(root),
      new Conv(root, session, 3, 512, 512, 1),
      new Activation(root),
      new Pool(root, 2, 2),

      // Block 5
      new Conv(root, session, 3, 512, 512, 1),
      new Activation(root),
      new Conv(root, session, 3, 512, 512, 1),
      new Activation(root),
      new Conv(root, session, 3, 512, 512, 1),
      new Activation(root),
      new Pool(root, 2, 2),

      new Flat(root),
      new FC(root, session, 512 * 7 * 7, 4096),
      new Activation(root),
      new FC(root, session, 4096, 4096),
      new Activation(root),
      new FC(root, session, 4096, 10),
  };
  std::cout << Vgg16.size() << std::endl;

  auto send_recv =
      std::vector<std::vector<int>>(2, std::vector<int>(2, MEM_INIT));
  status = 0;

  std::thread t1(worker, 0, std::ref(root), std::ref(session),
                 std::ref(variables), std::ref(send_recv), std::ref(enqueue),
                 std::ref(dequeue), std::ref(size), getenv("REDIS0"));
  std::thread t2(worker, 1, std::ref(root), std::ref(session),
                 std::ref(variables), std::ref(send_recv), std::ref(enqueue),
                 std::ref(dequeue), std::ref(size), getenv("REDIS1"));

  t1.join();
  t2.join();
  return 0;
}
