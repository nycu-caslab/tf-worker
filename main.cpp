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

    int in_channels = 1, out_channels = 1, filter_size = 2;
    TensorShape sp({filter_size, filter_size, in_channels, out_channels});
    ClientSession session(root);
    auto input = Variable(root.WithOpName("I"), sp, DT_FLOAT);
    std::vector<Tensor> outputs;
    auto assignedI =
        Assign(root.WithOpName("I_assign"), input,
               XavierInit(root, in_channels, out_channels, filter_size));
    TF_CHECK_OK(session.Run({}, {assignedI}, &outputs));
    Conv conv(root, filter_size, in_channels, out_channels);
    conv.forward(session, input);
    Conv conv2(root, filter_size, in_channels, out_channels);
    conv2.forward(session, input);
    return 0;
}
