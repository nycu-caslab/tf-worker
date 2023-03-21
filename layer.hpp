#include <iostream>
#include <stdio.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/nn_ops.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <vector>

#include "utils.hpp"
using namespace tensorflow;
using namespace tensorflow::ops;

class Op {
public:
  Op(Scope &root) : root(root) {}
  virtual Output forward(ClientSession &sesion, Input input) = 0;

private:
  Scope &root;
};

class Flat : public Op {
public:
  Flat(Scope &root) : root(root), Op(root) {}

  Output forward(ClientSession &session, Input input) {
    auto res = Reshape(root, input, {-1, 8});
    std::vector<Tensor> outputs;
    TF_CHECK_OK(session.Run({res}, &outputs));
    return res;
  }

private:
  Scope &root;
};

class FC : public Op {
public:
  FC(Scope &root, ClientSession &session, int in_channels, int out_channels)
      : root(root), in_channels(in_channels), out_channels(out_channels),
        weight(Variable(root, {out_channels, in_channels}, DT_FLOAT)),
        Op(root) {
    auto assignedW =
        Assign(root, weight,
               RandomNormal(root, {out_channels, in_channels}, DT_FLOAT));
    std::vector<Tensor> outputs;
    TF_CHECK_OK(session.Run({}, {assignedW}, &outputs));
  }

  Output forward(ClientSession &session, Input input) {
    std::vector<Tensor> outputs;
    auto res = MatMul(root, weight, input);
    TF_CHECK_OK(session.Run({res}, &outputs));
    return res;
  }

private:
  Scope &root;
  Variable weight;
  int in_channels;
  int out_channels;
};

class Activation : public Op {
public:
  Activation(Scope &root) : root(root), Op(root) {}
  Output forward(ClientSession &session, Input input) {
    auto res = Relu(root, input);
    std::vector<Tensor> outputs;
    TF_CHECK_OK(session.Run({res}, &outputs));
    return res;
  }

private:
  Scope &root;
};

class Pool : public Op {
public:
  Pool(Scope &root, int ksize, int stride)
      : root(root), ksize(ksize), stride(stride), Op(root) {}

  Output forward(ClientSession &session, Input input) {
    auto res = MaxPool(root, input, {1, ksize, ksize, 1},
                       {1, stride, stride, 1}, "SAME");
    std::vector<Tensor> outputs;
    TF_CHECK_OK(session.Run({res}, &outputs));
    return res;
  }

private:
  Scope &root;
  int ksize;
  int stride;
};

class Conv : public Op {
public:
  Conv(Scope &root, ClientSession &session, int filter_size = 2,
       int in_channels = 1, int out_channels = 1, int stride = 1)
      : root(root), filter_size(filter_size), in_channels(in_channels),
        out_channels(out_channels), Op(root),
        filter(Variable(root.WithOpName("B"),
                        {filter_size, filter_size, in_channels, out_channels},
                        DT_FLOAT)),
        stride(stride) {
    auto assignedW =
        Assign(root.WithOpName("W_assign"), filter,
               RandomNormal(
                   root, {filter_size, filter_size, in_channels, out_channels},
                   DT_FLOAT));
    std::vector<Tensor> outputs;
    TF_CHECK_OK(session.Run({}, {assignedW}, &outputs));
  }

  Output forward(ClientSession &session, Input input) {
    auto conv =
        Conv2D(root, input, filter, {stride, stride, stride, stride}, "SAME");
    std::vector<Tensor> outputs;
    TF_CHECK_OK(session.Run({conv}, &outputs));
    std::cout << outputs[0].shape() << "\n";
    return conv.output;
  }

private:
  Scope &root;
  Variable filter;
  int in_channels;
  int out_channels;
  int filter_size;
  int stride;
};
