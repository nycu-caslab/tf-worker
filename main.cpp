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

  Scope root = Scope::NewRootScope();
  ClientSession session(root);
  int in_channels = 3, out_channels = 64, filter_size = 3, stride = 1;

  std::vector<Tensor> outputs;

  TensorShape sp({1, 224, 224, 3});
  auto input = Variable(root, sp, DT_FLOAT);
  Output assignedI =
      Assign(root, input, RandomNormal(root, {1, 224, 224, 3}, DT_FLOAT));
  TF_CHECK_OK(session.Run({}, {assignedI}, &outputs));

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

  auto begin = std::chrono::high_resolution_clock::now();

  for (auto &layer : Vgg16) {
    assignedI = layer->forward(session, assignedI);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
                   .count()
            << "ns" << std::endl;

  return 0;
}
