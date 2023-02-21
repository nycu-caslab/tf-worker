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

int main() {
  std::ofstream stream;
  stream.open("tf-worker/log.txt");
  stream.close();

  LOGs("Process start");

  Scope root = Scope::NewRootScope();
  SessionOptions config;
  config.config.mutable_gpu_options()->set_allow_growth(true);
  ClientSession session(root, config);

  redisContext *c = redisConnect(getenv("REDIS"), 6379);

  LOGs("Creating redis instance");

  if (c == NULL || c->err) {
    if (c) {
      LOGs("Error:", c->errstr);
    } else {
      LOGs("Can't allocate redis context");
    }
  }

  FIFOQueue::Attrs fifoqueue_attr;
  FIFOQueue queue(root, {DT_FLOAT}, fifoqueue_attr.SharedName("test"));
  QueueEnqueue enqueue(root.WithOpName("enqueue"), queue, {Output()});
  QueueDequeue dequeue(root.WithOpName("dequeue"), queue, {DT_FLOAT});
  QueueSize size(root.WithOpName("size"), queue);

  std::vector<Tensor> outputs;
  TensorShape sp({1, 224, 224, 3});
  auto input = Variable(root, sp, DT_FLOAT);
  std::vector<Output> variables(10);
  variables[0] =
      Assign(root, input, RandomNormal(root, {1, 224, 224, 3}, DT_FLOAT));
  TF_CHECK_OK(session.Run({}, {variables[0]}, &outputs));

  std::vector<Op *> Vgg16 = {
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

  LOGs("Start pooling redis");

  while (true) {
    std::string result;
    redisReply *reply;
    reply = (redisReply *)redisCommand(c, "BLPOP foo 0");

    result = reply->element[1]->str;
    LOGs("Input: ", result);

    std::istringstream iss(result);
    freeReplyObject(reply);

    string cmd;
    iss >> cmd;
    if (cmd == "forward") {
      int variable_id, layer_id;
      iss >> variable_id >> layer_id;
      variables[variable_id] =
          Vgg16[layer_id]->forward(session, variables[variable_id]);
      LOGs("Forwarded: Vgg16-", layer_id);
    } else if (cmd == "push") {
      int variable_id;
      iss >> variable_id;
      std::vector<Tensor> outputs;
      TF_CHECK_OK(
          session.Run({}, {variables[variable_id]}, {enqueue}, &outputs));
      TF_CHECK_OK(session.Run({}, {size}, {}, &outputs));
      int value_3 = *outputs[0].scalar<int>().data();
      LOGs("Enqueued ", variable_id, "size: ", value_3);
    } else if (cmd == "pop") {
      int variable_id;
      iss >> variable_id;
      LOGs("POP", variable_id);
      std::vector<Tensor> outputs;
      TF_CHECK_OK(session.Run({}, {variables[variable_id]}, {dequeue.operation},
                              &outputs));
      TF_CHECK_OK(session.Run({}, {size}, {}, &outputs));
      int value_3 = *outputs[0].scalar<int>().data();
      LOGs("Dequeued ", variable_id, "size: ", value_3);
    }
  }
  return 0;
}
