#include <stdio.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>

#include <iostream>
using namespace tensorflow;
using namespace tensorflow::ops;

Input XavierInit(Scope scope, int in_chan, int out_chan, int filter_side) {
    float std;
    Tensor t;
    if (filter_side == 0) {  // Dense
        std = sqrt(6.f / (in_chan + out_chan));
        Tensor ts(DT_INT64, {2});
        auto v = ts.vec<int64_t>();
        v(0) = in_chan;
        v(1) = out_chan;
        t = ts;
    } else {  // Conv
        std = sqrt(6.f / (filter_side * filter_side * (in_chan + out_chan)));
        Tensor ts(DT_INT64, {4});
        auto v = ts.vec<int64_t>();
        v(0) = filter_side;
        v(1) = filter_side;
        v(2) = in_chan;
        v(3) = out_chan;
        t = ts;
    }
    auto rand = RandomUniform(scope, t, DT_FLOAT);
    return Multiply(scope, Sub(scope, rand, 0.5f), std * 2.f);
}

class Conv {
   public:
    Conv(Scope& root, int filter_size = 2, int in_channels = 1,
         int out_channels = 1)
        : filter_size(filter_size),
          in_channels(in_channels),
          out_channels(out_channels),
          root(root),
          filter(Variable(root.WithOpName("B"),
                          {filter_size, filter_size, in_channels, out_channels},
                          DT_FLOAT)) {}
    auto forward(ClientSession& session, Variable input) {
        auto conv = Conv2D(root, input, filter, {1, 1, 1, 1}, "SAME");
        auto assignedW =
            Assign(root.WithOpName("W_assign"), filter,
                   XavierInit(root, in_channels, out_channels, filter_size));
        std::vector<Tensor> outputs;
        TF_CHECK_OK(session.Run({}, {assignedW}, &outputs));
        std::cout << outputs[0].shape() << "\n";
        std::cout << outputs[0].tensor<float, 4>() << "\n";
         TF_CHECK_OK(session.Run({conv}, &outputs));
         std::cout << outputs[0].shape() << "\n";
         std::cout << outputs[0].tensor<float, 4>() << "\n";
        return outputs;
    }

   private:
    Scope root;
    Variable filter;
    int in_channels;
    int out_channels;
    int filter_size;
};
