#include <fstream>
#include <hiredis/hiredis.h>
#include <iostream>
#include <stdio.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>

#include "layer.hpp"
#include "utils.hpp"
using namespace tensorflow;
using namespace tensorflow::ops;

int main() {

  std::ofstream stream;
  stream.open("tf-worker/log.txt");
  stream.close();

  LOGs("Process start");

  Scope root = Scope::NewRootScope();
  ClientSession session(root);

  redisContext *c = redisConnect(getenv("REDIS"), 6379);

  LOGs("Creating redis instance");

  if (c == NULL || c->err) {
    if (c) {
      LOGs("Error:", c->errstr);
    } else {
      LOGs("Can't allocate redis context");
    }
  }

  LOGs("Start pooling redis");

  while (true) {
    string result;
    redisReply *reply;
    reply = (redisReply *)redisCommand(c, "BLPOP foo 0");

    result = reply->element[1]->str;
    LOGs("Input: ", result);

    std::istringstream iss(result);

    int in_channels, out_channels, filter_size, stride;
    iss >> in_channels >> out_channels >> filter_size >> stride;

    freeReplyObject(reply);
    TensorShape sp({1, 3, 224, 224});
    auto input = Variable(root.WithOpName("I"), sp, DT_FLOAT);
    std::vector<Tensor> outputs;
    auto assignedI = Assign(root.WithOpName("I_assign"), input,
                            RandomInit(root, 1, 3, 224, 224));

    TF_CHECK_OK(session.Run({}, {assignedI}, &outputs));
    LOGs(outputs[0].shape());
    Conv conv(root, filter_size, in_channels, out_channels, stride);
    conv.forward(session, input);

    LOGs(outputs[0].shape());
    // LOGs(outputs[0].tensor<float, 4>());
  }
  return 0;
}
