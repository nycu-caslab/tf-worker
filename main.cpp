#include <hiredis/hiredis.h>
#include <stdio.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>

#include <fstream>
#include <iostream>

#include "layer.hpp"
using namespace tensorflow;
using namespace tensorflow::ops;

#define LOGs(...) log2file(__LINE__, __FILE__, __VA_ARGS__)

template <typename... Args>
void log2file(int line, const char *fileName, Args &&...args) {
  std::ofstream stream;
  stream.open("tf-worker/log.txt", std::ofstream::out | std::ofstream::app);
  stream << fileName << "(" << line << ") : ";
  (stream << ... << std::forward<Args>(args)) << '\n';
}

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
    LOGs("Waiting");
    reply = (redisReply *)redisCommand(c, "BLPOP foo 0");

    result = reply->element[1]->str;
    LOGs("Result: ", result);

    int in_channels = 1, out_channels = std::stoi(result), filter_size = 2;
    freeReplyObject(reply);
    TensorShape sp({filter_size, filter_size, in_channels, out_channels});
    auto input = Variable(root.WithOpName("I"), sp, DT_FLOAT);
    std::vector<Tensor> outputs;
    auto assignedI =
        Assign(root.WithOpName("I_assign"), input,
               XavierInit(root, in_channels, out_channels, filter_size));

    LOGs("Run session");
    TF_CHECK_OK(session.Run({}, {assignedI}, &outputs));
    LOGs("Create conv layer");
    Conv conv(root, filter_size, in_channels, out_channels);
    LOGs("Forward");
    conv.forward(session, input);

    LOGs(outputs[0].shape());
    LOGs(outputs[0].tensor<float, 4>());
  }
  return 0;
}
