#include <hiredis/hiredis.h>
#include <stdio.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>

#include <iostream>

#include "layer.hpp"
using namespace tensorflow;
using namespace tensorflow::ops;

int main() {
  Scope root = Scope::NewRootScope();
  ClientSession session(root);

  redisContext *c = redisConnect(getenv("REDIS"), 6379);
  if (c == NULL || c->err) {
    if (c) {
      printf("Error: %s\n", c->errstr);
    } else {
      printf("Can't allocate redis context\n");
    }
  }
  while (true) {
    string result;
    redisReply *reply;
    while (true) {
      reply = (redisReply *)redisCommand(c, "LPOP foo");
      if (reply->type != REDIS_REPLY_NIL)
        break;
    }
    result = reply->str;

    int in_channels = 1, out_channels = atoi(reply->str), filter_size = 2;
    TensorShape sp({filter_size, filter_size, in_channels, out_channels});
    auto input = Variable(root.WithOpName("I"), sp, DT_FLOAT);
    std::vector<Tensor> outputs;
    auto assignedI =
        Assign(root.WithOpName("I_assign"), input,
               XavierInit(root, in_channels, out_channels, filter_size));
    TF_CHECK_OK(session.Run({}, {assignedI}, &outputs));
    Conv conv(root, filter_size, in_channels, out_channels);
    conv.forward(session, input);
  }
  return 0;
}
