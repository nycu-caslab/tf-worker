#include <iostream>
#include <stdio.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>

#include "utils.hpp"
using namespace tensorflow;
using namespace tensorflow::ops;

Input RandomInit(Scope scope, int d1, int d2, int d3, int d4) {
  float std;
  Tensor t;
  std = sqrt(6.f / (d1 * d2 * (d3 + d4)));
  Tensor ts(DT_INT64, {4});
  auto v = ts.vec<int64_t>();
  v(0) = d1;
  v(1) = d2;
  v(2) = d3;
  v(3) = d4;
  t = ts;
  auto rand = RandomUniform(scope, t, DT_FLOAT);
  return rand;
}

Input XavierInit(Scope scope, int in_chan, int out_chan, int filter_size) {
  float std;
  Tensor t;
  if (filter_size == 0) { // Dense
    std = sqrt(6.f / (in_chan + out_chan));
    Tensor ts(DT_INT64, {2});
    auto v = ts.vec<int64_t>();
    v(0) = in_chan;
    v(1) = out_chan;
    t = ts;
  } else { // Conv
    std = sqrt(6.f / (filter_size * filter_size * (in_chan + out_chan)));
    Tensor ts(DT_INT64, {4});
    auto v = ts.vec<int64_t>();
    v(0) = filter_size;
    v(1) = filter_size;
    v(2) = in_chan;
    v(3) = out_chan;
    t = ts;
  }
  auto rand = RandomUniform(scope, t, DT_FLOAT);
  return Multiply(scope, Sub(scope, rand, 0.5f), std * 2.f);
}

class Conv {
public:
  Conv(Scope &root, int filter_size = 2, int in_channels = 1,
       int out_channels = 1, int stride = 1)
      : filter_size(filter_size), in_channels(in_channels),
        out_channels(out_channels), root(root),
        filter(Variable(root.WithOpName("B"),
                        {filter_size, filter_size, in_channels, out_channels},
                        DT_FLOAT)),
        stride(stride) {}
  auto forward(ClientSession &session, Variable input) {
    auto conv = Conv2D(root, input, filter,
                       gtl::ArraySlice<int>({stride, stride, stride, stride}),
                       StringPiece("SAME"));
    LOGs("a");
    auto assignedW = Assign(
        root.WithOpName("W_assign"), filter,
        RandomInit(root, filter_size, filter_size, in_channels, out_channels));
    std::vector<Tensor> outputs;
    LOGs("a");
    TF_CHECK_OK(session.Run({}, {assignedW}, &outputs));
    LOGs("a");
    TF_CHECK_OK(session.Run({conv}, &outputs));
    LOGs("a");
    return outputs;
  }

private:
  Scope root;
  Variable filter;
  int in_channels;
  int out_channels;
  int filter_size;
  int stride;
};
